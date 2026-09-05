# OceanEmbed

**Subsurface ocean temperature reconstruction from surface satellite observations.**
SIH 2026 · INCOIS problem statement #01.

Reconstructs temperature at **15 standard depths (0-1000 m)** across the North
Indian Ocean (5-30 N, 45-105 E) at **0.25 deg / daily** resolution, using only
surface fields. Trained on GLORYS reanalysis, validated against independent ARGO.

```
SST · SSS · SLA · currents · winds
        │  harmonize -> 0.25 deg / daily common grid
        ▼
   N x N spatial patch per cell  (+ lat, lon, sin/cos day-of-year)
        ▼
   Embedding engine (CNN or ViT-tiny)  ->  latent z (128-d)
        ▼
   Reconstruction head (MLP + depth attention)  ->  T at 15 depths
        ▼
   train on GLORYS   |   validate on independent ARGO
```

The **N x N patch** is the point of the architecture: subsurface structure
depends on the eddy field around a cell, not just the value at it. A point-only
model cannot see that, which is exactly what the baseline comparison measures.

---

## Quick start (no downloads, runs today)

```bash
pip install -r requirements.txt

python scripts/00_make_synthetic.py     # physically-plausible synthetic cube
python scripts/01_harmonize.py          # -> data/processed/harmonized.nc
python scripts/02_build_dataset.py      # -> data/processed/dataset.npz
python scripts/02c_export_table.py      # -> data/processed/samples.parquet (flat table)
python scripts/03_train.py              # torch auto-detected, NumPy fallback otherwise
python scripts/03b_baseline.py          # per-depth RandomForest control
python scripts/04_evaluate.py           # -> outputs/scorecard.json + 3 figures
streamlit run scripts/05_demo.py        # interactive PoC
```

Every stage prints what it wrote and its dimensions. The synthetic path is the
safety net -- it always works, so there is always something to demo.

## Swapping in real data

Same schemas, so nothing downstream changes:

```bash
# 1. download (see scripts/download/README.md)
# 2. fill in the EDIT BLOCK at the top of 01_harmonize_real.py
python scripts/01_harmonize_real.py --dry-run    # inspects files, writes nothing
python scripts/01_harmonize_real.py              # -> real harmonized.nc
python scripts/02_build_dataset.py
python scripts/02b_argo_colocate.py              # real ARGO becomes the holdout
python scripts/03_train.py && python scripts/03b_baseline.py && python scripts/04_evaluate.py
```

---

## Data contracts

Everyone codes to these. Match the contract and your file drops in without
touching anyone else's code.

**A -- canonical names (never rename)**
`sst, sss, sla, ucur, vcur, uwind, vwind` · `temp` (time,depth,lat,lon) · `land_mask` (1=land)

**B -- `data/processed/harmonized.nc`**
coords `time` (daily), `depth` (15 levels), `lat`, `lon` (0.25 deg);
surface vars (time,lat,lon), `temp` (time,depth,lat,lon), `land_mask` (lat,lon).
*Everything downstream reads only this file.* Data people own everything up to
here; model people own everything after.

**C -- `data/processed/dataset.npz`**
`trX(N,C,P,P) trY(N,15)` · `vaX,vaY` · `arX,arY,arM` (independent holdout + meta)
· `xm,xs,ym,ys` (scalers, **train split only**) · `depths`

**D -- standard depths** `[0,5,10,20,30,50,75,100,125,150,200,300,500,700,1000]`

**E -- `outputs/scorecard.json`** per-depth corr/RMSE/bias + band means.
Baseline writes `baseline_scorecard.json` in the same shape.

---

## Layout

```
configs/config.yaml       domain, depths, patch, model, training -- single source of truth
oceanembed/
  synthetic.py            physically-plausible cube; T(z) depends on the NEIGHBOURHOOD
  harmonize.py            regrid, depth-interp, daily resample, Kelvin detection
  loaders.py              real NetCDF -> canonical names (PRODUCT_VARS table)
  dataset.py              patch extraction, scalers, chronological splits
  model.py                CNN / ViT encoders, depth-attention head  (torch)
  model_numpy.py          NumPy MLP fallback with Adam -- runs without torch
  backend.py              torch_available() -- runs a real tensor op
  inference.py            Predictor.load() -- one checkpoint loader for both backends
  evaluate.py             per-depth metrics, scorecard, comparison tables
scripts/                  00..05 pipeline stages, plus the real-data variants
  download/README.md      Copernicus / PODAAC / INCOIS instructions
```

## Design notes

- **Scalers are fitted on the train split only** and saved with the checkpoint,
  so evaluation cannot leak.
- **Splits are chronological.** Earliest days train, then val, and the last block
  of days is the holdout -- independent in time, never used for model selection.
- **SST Kelvin is auto-detected** (converted only if the median exceeds 100), so
  OSTIA and already-converted files both work.
- **GLORYS depth levels are interpolated** onto the 15 standard levels.
- **Land and NaN are handled explicitly**: any sample whose patch or target
  contains a NaN is dropped, and the percent missing is reported.
- **Training uses a GPU automatically** when one is available, and prints the
  device it chose. Note that `pip install torch` gives a CPU-only build on
  Windows; for an NVIDIA GPU install from the PyTorch CUDA index instead
  (`--index-url https://download.pytorch.org/whl/cu130`). Override with
  `--device cpu|cuda`.
- **The NumPy fallback is real.** `backend.torch_available()` executes a matmul
  and a linear layer, so a torch that imports but is broken still falls back
  cleanly and the pipeline finishes.

## Results (synthetic holdout: 7200 profiles over 18 days never seen in training)

CNN encoder, patch 9, 120 epochs.

| | OceanEmbed | RF baseline (point-only) |
|---|---|---|
| mean correlation | **0.986** | 0.936 |
| mean RMSE | **0.218 degC** | 0.619 degC |
| mixed layer 0-50 m RMSE | **0.182** | 0.271 |
| thermocline 75-300 m RMSE | **0.323** | 1.170 |
| deep 500-1000 m RMSE | **0.084** | 0.215 |

**64.7% lower mean RMSE than the point-only baseline**, and 64-77% lower at every
depth between 50 m and 500 m.

The field-of-view ablation is the result that matters most: with identical
samples and only the patch size changed, going from a 1x1 patch (centre cell
only) to 9x9 cuts mean RMSE by 48% and thermocline RMSE by 58%. That isolates the
spatial contribution cleanly, which a CNN-vs-RandomForest comparison cannot.
See `EXPERIMENTS.md`.

Two honest points to keep in any write-up:

- The baseline still edges us in the **top 10 m** (0.119 vs 0.164 degC). Near the
  surface, temperature is essentially the surface temperature, so a point model
  recovers it almost exactly. The 0.045 degC gap is negligible, and from 20 m
  down we win everywhere.
- **Skill decays below 500 m**, where the surface stops constraining temperature.
  That is physics, not a defect, and we report it rather than quoting one
  flattering basin-wide number.

Numbers vary by roughly +/-0.01 degC between runs even with the seed fixed, so
quote your own `outputs/scorecard.json`. Full detail there, plus three figures:
`skill_vs_depth.png`, `example_profiles.png`, `spatial_map.png`. See
`FINAL_PLAN.md` for roles and the timeline, `EXPERIMENTS.md` for the ablations.
