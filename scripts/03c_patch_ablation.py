"""Stage 03c -- the field-of-view ablation.

    python scripts/03c_patch_ablation.py [--sizes 1,3,5,7,9,11] [--epochs 25]

Answers the one question the whole architecture rests on: does letting the model
see the NEIGHBOURING ocean actually help, and how much neighbourhood is useful?

Method matters here. Rebuilding dataset.npz at each patch size would also change
WHICH cells are sampled -- a 9x9 patch needs 4 cells of margin from the coast, a
1x1 needs none -- so the variants would be scored on different holdout sets and
the comparison would be worthless.

Instead we build the dataset ONCE at the largest size and centre-crop it. Every
variant then sees identical samples, identical targets and identical training
days. The only thing that changes is how far the model can see. That makes the
difference attributable to field of view and nothing else.

Writes outputs/patch_ablation.json.
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
from oceanembed.dataset import apply_scalers, invert_target
from oceanembed.evaluate import scorecard


def centre_crop(X: np.ndarray, out: int) -> np.ndarray:
    """Crop (N,C,P,P) down to (N,C,out,out) around the centre cell."""
    P = X.shape[2]
    if out == P:
        return X
    if out > P:
        raise ValueError(f"cannot crop {P} up to {out} -- rebuild the dataset larger")
    lo = (P - out) // 2
    return np.ascontiguousarray(X[:, :, lo:lo + out, lo:lo + out])


def train_one(cfg, trX, trY, vaX, vaY, arX, arY, depths, w, P, epochs, seed, device):
    import torch

    from oceanembed.model import build_model, weighted_mse

    torch.manual_seed(seed)
    np.random.seed(seed)
    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    cfg = json.loads(json.dumps(cfg))          # deep copy so we can set patch size
    cfg["patch"]["size"] = P
    model = build_model(cfg, in_ch=trX.shape[1], n_depths=int(depths.size)).to(dev)

    tr_x = torch.from_numpy(trX).to(dev); tr_y = torch.from_numpy(trY).to(dev)
    va_x = torch.from_numpy(vaX).to(dev); va_y = torch.from_numpy(vaY).to(dev)
    wt = torch.from_numpy(w).to(dev)
    n = tr_x.shape[0]
    batch = int(cfg["train"]["batch_size"])

    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"]["lr"]),
                            weight_decay=float(cfg["train"]["weight_decay"]))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best, best_state = np.inf, None
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, device=dev)
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            opt.zero_grad(set_to_none=True)
            loss = weighted_mse(model(tr_x[idx]), tr_y[idx], wt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            vl = float(weighted_mse(model(va_x), va_y, wt).detach())
        if vl < best:
            best = vl
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(arX).to(dev)).cpu().numpy()
    n_par = sum(p.numel() for p in model.parameters())
    return pred, best, n_par


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=str, default="1,3,5,7,9,11")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--device", type=str, default="auto")
    args = ap.parse_args()

    cfg = load_config()
    ensure_dirs(cfg)
    dpath = resolve(cfg["paths"]["dataset"])
    if not dpath.exists():
        raise SystemExit(f"missing {dpath} -- run scripts/02_build_dataset.py first")

    d = np.load(dpath, allow_pickle=True)
    xm, xs, ym, ys = d["xm"], d["xs"], d["ym"], d["ys"]
    trX_full, trY = apply_scalers(d["trX"], d["trY"], xm, xs, ym, ys)
    vaX_full, vaY = apply_scalers(d["vaX"], d["vaY"], xm, xs, ym, ys)
    arX_full, _ = apply_scalers(d["arX"], None, xm, xs, ym, ys)
    arY = d["arY"]
    depths = d["depths"]
    P_built = trX_full.shape[2]

    w = np.asarray(cfg["train"]["depth_weights"], dtype=np.float32)
    if w.size != depths.size:
        w = np.ones(depths.size, dtype=np.float32)
    seed = int(cfg["train"]["seed"])

    sizes = [int(v) for v in args.sizes.split(",")]
    bad = [p for p in sizes if p > P_built]
    if bad:
        raise SystemExit(
            f"dataset was built at patch {P_built}, cannot test {bad}. "
            f"Set patch.size to at least {max(bad)} in configs/config.yaml and re-run 02."
        )

    print(f"dataset built at patch {P_built}; identical samples reused for every size")
    print(f"train {trX_full.shape[0]}  val {vaX_full.shape[0]}  holdout {arX_full.shape[0]}")
    print(f"epochs {args.epochs} each\n")

    results = []
    for P in sizes:
        t0 = time.time()
        trX = centre_crop(trX_full, P)
        vaX = centre_crop(vaX_full, P)
        arX = centre_crop(arX_full, P)
        pred_n, val_loss, n_par = train_one(
            cfg, trX, trY, vaX, vaY, arX, arY, depths, w, P, args.epochs, seed, args.device
        )
        pred = invert_target(pred_n, ym, ys)
        card = scorecard(arY, pred, depths, label=f"patch_{P}")
        dt = time.time() - t0
        row = {
            "patch": P,
            "params": int(n_par),
            "val_loss": float(val_loss),
            "mean_corr": card["mean_corr"],
            "mean_rmse": card["mean_rmse"],
            "mixed_layer_rmse": card["bands"]["mixed_layer_0_50m"]["mean_rmse"],
            "thermocline_rmse": card["bands"]["thermocline_75_300m"]["mean_rmse"],
            "deep_rmse": card["bands"]["deep_500_1000m"]["mean_rmse"],
            "seconds": round(dt, 1),
            "per_depth": card["per_depth"],
        }
        results.append(row)
        print(f"  patch {P:2d}  mean RMSE {row['mean_rmse']:.4f}  "
              f"thermocline {row['thermocline_rmse']:.4f}  corr {row['mean_corr']:.4f}  "
              f"({dt:.0f}s, {n_par:,} params)")

    base = results[0]
    print(f"\n{'patch':>6} {'mean RMSE':>10} {'vs patch 1':>11} {'thermocline':>12} "
          f"{'vs patch 1':>11} {'corr':>7}")
    print("-" * 64)
    for r in results:
        i_all = 100 * (base["mean_rmse"] - r["mean_rmse"]) / base["mean_rmse"]
        i_th = 100 * (base["thermocline_rmse"] - r["thermocline_rmse"]) / base["thermocline_rmse"]
        print(f"{r['patch']:6d} {r['mean_rmse']:10.4f} {i_all:10.1f}% "
              f"{r['thermocline_rmse']:12.4f} {i_th:10.1f}% {r['mean_corr']:7.4f}")

    best = min(results, key=lambda r: r["mean_rmse"])
    out = resolve(cfg["paths"]["outputs"]) / "patch_ablation.json"
    out.write_text(json.dumps({
        "note": "identical samples across sizes -- dataset built once at patch "
                f"{P_built} and centre-cropped, so only the field of view differs",
        "dataset_patch": int(P_built), "epochs": args.epochs,
        "best_patch": best["patch"], "results": results,
    }, indent=2), encoding="utf-8")

    print(f"\nbest patch size: {best['patch']} (mean RMSE {best['mean_rmse']:.4f})")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
