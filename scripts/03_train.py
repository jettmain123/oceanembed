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
    args = ap.parse_args()

    cfg = load_config()
    ensure_dirs(cfg)
    if args.encoder:
        cfg["model"]["encoder"] = args.encoder
    epochs = args.epochs or int(cfg["train"]["epochs"])
    batch = args.batch_size or int(cfg["train"]["batch_size"])
    lr = args.lr or float(cfg["train"]["lr"])
    wd = float(cfg["train"]["weight_decay"])
    seed = int(cfg["train"]["seed"])

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
    print(f"epochs {epochs}  batch {batch}  lr {lr}  wd {wd}  encoder {cfg['model']['encoder']}")

    t0 = time.time()
    out_dir = resolve(cfg["paths"]["outputs"])

    if use_torch:
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        from oceanembed.model import build_model, weighted_mse

        torch.manual_seed(seed)
        np.random.seed(seed)
        dev = torch.device("cpu")
        model = build_model(cfg, in_ch=n_ch, n_depths=int(depths.size)).to(dev)
        n_par = sum(p.numel() for p in model.parameters())
        print(f"model: {cfg['model']['encoder']} encoder, {n_par:,} parameters")

        tr_ds = TensorDataset(torch.from_numpy(trX), torch.from_numpy(trY))
        va_x = torch.from_numpy(vaX)
        va_y = torch.from_numpy(vaY)
        dl = DataLoader(tr_ds, batch_size=batch, shuffle=True, num_workers=0, drop_last=False)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        wt = torch.from_numpy(w)

        best, history = np.inf, []
        ckpt = resolve(cfg["paths"]["checkpoint"])
        for ep in range(1, epochs + 1):
            model.train()
            tot, seen = 0.0, 0
            for xb, yb in dl:
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
                        "state_dict": model.state_dict(),
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
                    "encoder": cfg["model"]["encoder"] if use_torch else "numpy_mlp",
                    "epochs": epochs, "best_val_loss": float(best),
                    "seconds": round(dt, 1), "history": history}, indent=2),
        encoding="utf-8",
    )

    print(f"\nbest val loss : {best:.5f}")
    print(f"trained in    : {dt:.1f} s")
    print(f"checkpoint    : {saved}")
    print(f"history       : {out_dir / 'train_history.json'}")


if __name__ == "__main__":
    main()
