"""Stage 03 -- train OceanEmbed on dataset.npz.

    python scripts/03_train.py [--epochs N] [--encoder cnn|vit] [--backend auto|torch|numpy]

Auto-detects the backend: the torch CNN/ViT if torch really works, otherwise the
NumPy MLP fallback. Either way the best-by-validation checkpoint is written and
04_evaluate.py can read it.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from oceanembed import ensure_dirs, load_config, resolve
from oceanembed.backend import describe, torch_available
from oceanembed.dataset import apply_scalers


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--encoder", type=str, default=None, choices=["cnn", "vit"])
    ap.add_argument("--backend", type=str, default="auto", choices=["auto", "torch", "numpy"])
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--weight-decay", type=float, default=None)
    ap.add_argument("--channel-dropout", type=float, default=None,
                    help="probability of blanking a whole input channel per sample. "
                         "Teaches the model to cope when a satellite is down or a "
                         "product has a gap -- something an assimilation system "
                         "cannot do without a full rerun.")
    ap.add_argument("--seed", type=int, default=None,
                    help="override the config seed -- use several to check a result is not noise")
    ap.add_argument("--device", type=str, default="auto",
                    help="auto | cpu | cuda | cuda:0 -- auto uses the GPU when one is usable")
    args = ap.parse_args()

    cfg = load_config()
    ensure_dirs(cfg)
    if args.encoder:
        cfg["model"]["encoder"] = args.encoder
    epochs = args.epochs or int(cfg["train"]["epochs"])
    batch = args.batch_size or int(cfg["train"]["batch_size"])
    lr = args.lr or float(cfg["train"]["lr"])
    wd = args.weight_decay if args.weight_decay is not None else float(cfg["train"]["weight_decay"])
    seed = args.seed if args.seed is not None else int(cfg["train"]["seed"])
    cdrop = (args.channel_dropout if args.channel_dropout is not None
             else float(cfg["train"].get("channel_dropout", 0.0)))

    dpath = resolve(cfg["paths"]["dataset"])
    if not dpath.exists():
        raise SystemExit(f"missing {dpath} -- run scripts/02_build_dataset.py first")
    d = np.load(dpath, allow_pickle=True)

    xm, xs, ym, ys = d["xm"], d["xs"], d["ym"], d["ys"]
    trX, trY = apply_scalers(d["trX"], d["trY"], xm, xs, ym, ys)
    vaX, vaY = apply_scalers(d["vaX"], d["vaY"], xm, xs, ym, ys)
    depths = d["depths"]
    n_ch, P = trX.shape[1], trX.shape[2]

    w = np.asarray(cfg["train"]["depth_weights"], dtype=np.float32)
    if w.size != depths.size:
        w = np.ones(depths.size, dtype=np.float32)

    use_torch = torch_available() if args.backend == "auto" else (args.backend == "torch")
    print(describe())
    print(f"train {trX.shape}  val {vaX.shape}  channels {n_ch}  patch {P}  depths {depths.size}")
    print(f"epochs {epochs}  batch {batch}  lr {lr}  wd {wd}  encoder {cfg['model']['encoder']}"
          f"  channel_dropout {cdrop}")

    t0 = time.time()
    out_dir = resolve(cfg["paths"]["outputs"])

    if use_torch:
        import torch

        from oceanembed.model import build_model, weighted_mse

        torch.manual_seed(seed)
        np.random.seed(seed)

        # Pick the device: GPU when one is genuinely usable, else CPU. --device
        # overrides. A CPU-only torch build reports no CUDA even on a GPU box,
        # so say so loudly instead of silently training 30x slower.
        if args.device == "auto":
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            dev = torch.device(args.device)
        if dev.type == "cuda":
            torch.cuda.manual_seed_all(seed)
            print(f"device: cuda -> {torch.cuda.get_device_name(0)} "
                  f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")
        else:
            note = ""
            if not torch.cuda.is_available() and "+cpu" in torch.__version__:
                note = "  (torch is a CPU-only build -- reinstall with CUDA to use a GPU)"
            print(f"device: cpu{note}")

        model = build_model(cfg, in_ch=n_ch, n_depths=int(depths.size)).to(dev)
        n_par = sum(p.numel() for p in model.parameters())
        print(f"model: {cfg['model']['encoder']} encoder, {n_par:,} parameters")

        # The whole dataset is small (a few hundred MB), so keep it resident on
        # the device and slice batches directly. That removes the per-batch host
        # transfer and the DataLoader overhead, which dominate at this size.
        tr_x = torch.from_numpy(trX).to(dev)
        tr_y = torch.from_numpy(trY).to(dev)
        va_x = torch.from_numpy(vaX).to(dev)
        va_y = torch.from_numpy(vaY).to(dev)
        wt = torch.from_numpy(w).to(dev)
        n_train = tr_x.shape[0]

        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

        best, history = np.inf, []
        ckpt = resolve(cfg["paths"]["checkpoint"])
        for ep in range(1, epochs + 1):
            model.train()
            tot, seen = 0.0, 0
            perm = torch.randperm(n_train, device=dev)
            for s in range(0, n_train, batch):
                idx = perm[s:s + batch]
                xb, yb = tr_x[idx], tr_y[idx]
                if cdrop > 0:
                    # blank whole channels, not pixels: a missing satellite loses
                    # an entire variable. Inputs are standardised, so zero is the
                    # channel mean -- the honest "no information" value.
                    m = (torch.rand(xb.shape[0], xb.shape[1], 1, 1, device=dev)
                         >= cdrop).float()
                    xb = xb * m
                opt.zero_grad(set_to_none=True)
                loss = weighted_mse(model(xb), yb, wt)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                tot += float(loss.detach()) * xb.shape[0]; seen += xb.shape[0]
            sched.step()
            model.eval()
            with torch.no_grad():
                vl = float(weighted_mse(model(va_x), va_y, wt).detach())
            tr = tot / max(seen, 1)
            history.append({"epoch": ep, "train_loss": tr, "val_loss": vl})
            flag = ""
            if vl < best:
                best = vl
                torch.save(
                    {
                        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "cfg_model": cfg["model"], "patch": cfg["patch"],
                        "in_ch": n_ch, "n_depths": int(depths.size),
                        "xm": xm, "xs": xs, "ym": ym, "ys": ys, "depths": depths,
                        "backend": "torch", "epoch": ep, "val_loss": vl,
                    },
                    ckpt,
                )
                flag = "  <- best"
            print(f"  epoch {ep:3d}/{epochs}  train {tr:.5f}  val {vl:.5f}{flag}")
        saved = ckpt
    else:
        from oceanembed.model_numpy import NumpyOceanEmbed

        model = NumpyOceanEmbed(
            in_dim=int(np.prod(trX.shape[1:])), n_depths=int(depths.size),
            embed_dim=int(cfg["model"]["embed_dim"]), hidden=int(cfg["model"]["head_hidden"]), seed=seed,
        )
        n_par = sum(v.size for v in model.p.values())
        print(f"model: numpy MLP fallback, {n_par:,} parameters")
        history = model.fit(trX, trY, vaX, vaY, epochs=epochs, batch_size=batch, lr=lr,
                            weight_decay=wd, weights=w, seed=seed)
        best = min(h["val_loss"] for h in history)
        saved = resolve(cfg["paths"]["checkpoint_npy"])
        model.save(saved)
        np.savez(
            str(saved).replace(".npz", "_scalers.npz"),
            xm=xm, xs=xs, ym=ym, ys=ys, depths=depths,
        )

    dt = time.time() - t0
    (out_dir / "train_history.json").write_text(
        json.dumps({"backend": "torch" if use_torch else "numpy",
                    "device": str(dev) if use_torch else "cpu",
                    "encoder": cfg["model"]["encoder"] if use_torch else "numpy_mlp",
                    "epochs": epochs, "best_val_loss": float(best),
                    "channel_dropout": cdrop,
                    "seconds": round(dt, 1), "history": history}, indent=2),
        encoding="utf-8",
    )

    print(f"\nbest val loss : {best:.5f}")
    print(f"trained in    : {dt:.1f} s")
    print(f"checkpoint    : {saved}")
    print(f"history       : {out_dir / 'train_history.json'}")


if __name__ == "__main__":
    main()
