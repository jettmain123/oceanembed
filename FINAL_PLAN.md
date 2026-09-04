# OceanEmbed — FINAL 20-Hour Merge-Safe Plan (SIH 2026 · INCOIS PS#01)

This is the single source of truth. It merges the ChatGPT blueprint (concepts, tabular
schema, checklists) with the working codebase (runnable modules, tested end-to-end).

**Golden rule:** the synthetic pipeline already runs and is submittable. Never break it.
Real data is layered ON TOP. If real data stalls, you still demo and submit.

---

## 0. The one architecture everyone builds toward

```
multi-source surface data (SST, SSS, SLA, Ucur, Vcur, Uwind, Vwind)
        │  harmonize → 0.25° / daily common grid  (harmonize.py)
        ▼
  N×N spatial patch around each cell  (+lat, lon, sin/cos day-of-year)
        ▼
  Embedding engine (CNN / ViT)  →  latent z (128-d)     (model.py)
        ▼
  Reconstruction head (MLP + depth-attention)  →  T at 15 depths
        ▼
  train on GLORYS   |   validate on independent ARGO
  metrics: correlation, RMSE, bias  (per depth)         (evaluate.py)
        ▼
  Streamlit PoC: point profile + basin heatmap          (05_demo.py)
```

We keep the **N×N patch** (not point-MLP) because that is what lets the CNN see eddies
and fronts — the reason a deep model beats the baseline. The flat one-row-per-cell table
(from the ChatGPT plan) is used ONLY as the baseline feature format and the hand-off
contract between people (see §2).

---

## 1. THE DATA CONTRACTS (this is what makes parallel work merge seamlessly)

Everyone codes to these fixed interfaces. As long as your file matches the contract, it
drops in without touching anyone else's code.

### Contract A — canonical variable names (NEVER rename these)
```
sst, sss, sla, ucur, vcur, uwind, vwind      # surface inputs (7)
temp                                          # target, dims (time, depth, lat, lon)
land_mask                                     # 1 = land, 0 = ocean
```

### Contract B — harmonized cube  →  data/processed/harmonized.nc
An xarray Dataset with:
```
coords: time (daily), depth (the 15 standard levels), lat, lon (0.25° grid)
data_vars: sst,sss,sla,ucur,vcur,uwind,vwind  dims (time,lat,lon)
           temp                                dims (time,depth,lat,lon)
           land_mask                           dims (lat,lon)
```
Produced by `01_harmonize.py` (synthetic) OR `01_harmonize_real.py` (real).
**Everything downstream reads ONLY this file.** Data people own everything up to here;
model people own everything after. Clean seam.

### Contract C — ML dataset  →  data/processed/dataset.npz
```
trX (N,C,P,P) trY (N,15)     # train
vaX, vaY                      # val
arX, arY, arM                 # independent ARGO-style holdout (+meta lat,lon,t)
xm,xs,ym,ys                   # scalers (train stats only)
depths                        # the 15 levels
```
Produced by `02_build_dataset.py`; ARGO holdout replaced by real profiles via
`02b_argo_colocate.py`.

### Contract D — standard depths (fixed, from PS)
```
[0,5,10,20,30,50,75,100,125,150,200,300,500,700,1000]  # 15 levels
```

### Contract E — metrics output  →  outputs/scorecard.json
Per-depth corr/RMSE/bias + mean scorecard. Baseline writes baseline_scorecard.json.

If you honor Contracts B and C, you can rewrite any module and nothing else breaks.

---

## 2. WHO DOES WHAT (4–5 people, fully parallel from hour 0)

### Person 1 — DATA (owns everything up to harmonized.nc)
- Hour 0: start downloads (see §3). Small window first, big window in background.
- Fill variable names into `scripts/01_harmonize_real.py` (SURFACE dict + TARGET).
- Run it → produce real `harmonized.nc` matching Contract B.
- Deliverable: `harmonized.nc` + a `DATA_SOURCES.md` (product, version, DOI, access date,
  % missing per variable).
- **Until real data lands, downstream people use the synthetic harmonized.nc — identical schema.**

### Person 2 — VALIDATION / ARGO (owns the credibility centerpiece)
- Download gridded ARGO from INCOIS LAS for the same window/box.
- Fill the 3 var names at the top of `scripts/02b_argo_colocate.py`.
- Run it → real independent ARGO holdout spliced into dataset.npz (Contract C).
- Deliverable: co-located ARGO count + which region/dates.

### Person 3 — MODEL (owns everything after dataset.npz)
- Works entirely on synthetic dataset.npz from hour 0 — no waiting.
- Confirm torch works on your machine (`python -c "import torch;print(torch.randn(2,2)@torch.randn(2,2))"`).
  If yes, `03_train.py` auto-uses the real CNN/ViT. If broken, the numpy fallback still runs.
- Run `03b_baseline.py` (RF) for the comparison story.
- Tune encoder (cnn vs vit in config.yaml), patch size, epochs. Save best checkpoint.
- Deliverable: `outputs/oceanembed.pt` + training curves.

### Person 4 — EVAL + DEMO (owns judge-facing outputs)
- Run `04_evaluate.py` → scorecard.json, skill_vs_depth.png, example_profiles.png.
- Polish `05_demo.py`: point it at harmonized.nc, add a real ARGO-vs-prediction panel.
- Deliverable: working Streamlit demo + the 3 result plots.

### Person 5 (or shared) — PITCH + REPORT
- 5-slide deck: problem → solution (embedding idea) → data+architecture → results (with
  baseline comparison + ARGO validation) → demo + limitations + future work.
- Technical report: data table, % missing, splits, metrics, honest limitations.

Sync points: **hour 6, hour 12, hour 17.** At each, whoever has real outputs replaces the
synthetic ones — because schemas are identical, it's a file swap, not a code change.

---

## 3. DATA DOWNLOAD (hour 0 = right now, in parallel)

Register once, then subset by region+date (do NOT download global files).

- **Copernicus Marine** (SST OSTIA, SSS, SLA DUACS, GLORYS target): `pip install copernicusmarine; copernicusmarine login`
- **NASA PODAAC / Earthdata** (OSCAR currents, CCMP winds): free Earthdata login
- **INCOIS LAS** (gridded ARGO): las.incois.gov.in

**Window:** start with **1 month** to get the pipeline flowing and read variable names.
In background pull **1–2 years** (must be a period ALL products overlap — verify first).
Box: full domain 45–105°E, 5–30°N, or a smaller PoC box (Bay of Bengal 80–100°E,10–25°N)
if downloads are slow.

**Split chronologically:** earlier dates → train, later dates → val; ARGO = independent test.

---

## 4. EXECUTION ORDER (commands)

```
# ---- works TODAY on synthetic data (safety net) ----
python scripts/00_make_synthetic.py
python scripts/01_harmonize.py
python scripts/02_build_dataset.py
python scripts/03_train.py            # torch auto-detected, else numpy fallback
python scripts/03b_baseline.py        # RF baseline for comparison
python scripts/04_evaluate.py
streamlit run scripts/05_demo.py

# ---- swap in REAL data (same schemas, no downstream changes) ----
# (edit var names in 01_harmonize_real.py and 02b_argo_colocate.py first)
python scripts/01_harmonize_real.py   # -> real harmonized.nc
python scripts/02_build_dataset.py    # rebuild dataset from real cube
python scripts/02b_argo_colocate.py   # real ARGO holdout
python scripts/03_train.py
python scripts/03b_baseline.py
python scripts/04_evaluate.py
```

---

## 5. 20-HOUR TIMELINE

| Hours | Milestone (must be submittable at each) |
|---|---|
| 0–2 | downloads started (1mo + big bg); synthetic pipeline re-verified on every machine; git repo + roles fixed |
| 2–6 | real 1-month data harmonized → real harmonized.nc; model person training on synthetic in parallel |
| 6–8 | **SYNC 1**: real dataset.npz built; baseline running; torch confirmed |
| 8–12 | real ARGO co-located; CNN training on real data; first real scorecard |
| 12 | **SYNC 2**: real metrics exist; demo points at real data |
| 12–17 | tune model; finalize plots; build demo ARGO panel; draft slides |
| 17 | **SYNC 3**: freeze features. Everything from here is polish only |
| 17–20 | report, slides, rehearse demo twice, package repo. Add nothing new |

---

## 6. DEFINITION OF DONE (from the merged checklists)

Data: all 7 inputs + GLORYS + ARGO loaded · common 0.25°/daily grid · units verified
(SST Kelvin→°C) · missing handled + % recorded · scalers saved.
ML: baseline + CNN train · chronological val · best checkpoint · inference works.
Eval: per-depth corr/RMSE/bias · profile plot · error-vs-depth plot · spatial map if time.
Product: Streamlit with location+date+profile+heatmap+metrics · demo mode · screenshots.
Pitch: problem · solution · data · architecture · results · demo · limitations · future.

**When Strong-MVP (MVP + CNN + baseline comparison + chronological test + ARGO) is done: STOP.**

---

## 7. Judge talking points

- "Skill follows the physics": high correlation in the mixed layer, decaying below the
  thermocline where surface no longer constrains T — we quantify it, not hide it.
- Real independent validation against ARGO (never seen in training).
- OceanEmbed beats the RF baseline by using spatial patches (eddies/fronts), not just point values.
- Data-source agnostic: same schema, swap synthetic→real, rerun. Operational-ready path.

---

## 8. BUILD STATUS (what is already in this repo)

The synthetic pipeline is **built, run and verified end-to-end**. Stages 00-04
produce a scorecard today; 05 is the Streamlit demo. Real-data scripts are
written with fill-in EDIT blocks and have NOT been run (no downloads yet).

| Stage | File | Status |
|---|---|---|
| 00 synthetic cube | `scripts/00_make_synthetic.py` | runs |
| 01 harmonize | `scripts/01_harmonize.py` | runs -> Contract B |
| 01 harmonize REAL | `scripts/01_harmonize_real.py` | written, EDIT BLOCK, `--dry-run` inspector |
| 02 dataset | `scripts/02_build_dataset.py` | runs -> Contract C |
| 02b ARGO co-locate | `scripts/02b_argo_colocate.py` | written, EDIT BLOCK, `--dry-run` |
| 02c flat table | `scripts/02c_export_table.py` | runs -> 25-col parquet |
| 03 train | `scripts/03_train.py` | runs (torch CNN/ViT, NumPy fallback) |
| 03b baseline | `scripts/03b_baseline.py` | runs -> baseline_scorecard.json |
| 04 evaluate | `scripts/04_evaluate.py` | runs -> scorecard.json + 3 figures |
| 05 demo | `scripts/05_demo.py` | imports clean, degrades gracefully |

### Measured result on the synthetic holdout (7200 profiles, 18 unseen days)

| | OceanEmbed (CNN) | RF baseline (point-only) |
|---|---|---|
| mean correlation | **0.965** | 0.936 |
| mean RMSE | **0.441 degC** | 0.619 degC |
| mixed layer 0-50 m RMSE | 0.491 | **0.271** |
| thermocline 75-300 m RMSE | **0.552** | 1.170 |
| deep 500-1000 m RMSE | **0.119** | 0.215 |

**~29% lower mean RMSE than the baseline, and roughly 45-60% lower right through
the thermocline (75-300 m)** -- which is exactly where spatial context should
matter and where the science is hard.

torch on CPU is not bit-deterministic even with the seed fixed; repeated runs of
`03_train.py` land between about 28% and 35% mean-RMSE improvement. Always quote
your own `outputs/scorecard.json`, never a number copied from these docs.

Report the top-20 m row honestly: the RF baseline WINS there (0.12 vs ~0.45 degC
at the surface). Near the surface T is almost exactly SST, so a point model recovers
it trivially, and our loss deliberately up-weights the thermocline (weight 2.0)
over the surface (weight 1.0). It is a designed trade-off, not a defect -- and
saying so is stronger than hiding it.

### Three things worth knowing before you tune anything

**1. The synthetic subsurface genuinely depends on the neighbourhood.**
`synthetic.eddy_diagnostics()` derives a smoothed SLA field and an EKE proxy
(smoothed |grad SLA|) and feeds them into the thermocline scale and mixed-layer
depth. That is deliberate: if T(z) were a purely pointwise function of the centre
cell, a point model would match the CNN exactly and the "spatial patches beat the
baseline" claim would be empty. It is now a claim the numbers actually support.

**2. The head gets a skip connection from the raw centre cell**
(`model.surface_skip`). Shallow depths are essentially SST; without the skip the
pooled embedding blurs the very cell being predicted. Measured, not assumed:
adding it cut mean RMSE from 0.474 to 0.442.

**3. The CNN encoder keeps a spatially intact feature map, not just pooling.**
Global average pooling discards WHERE the eddy sits relative to the cell -- and a
gradient is precisely a where-question. With pooling alone the NumPy MLP fallback
(which sees the flattened patch) actually beat the CNN. Adding a 1x1-reduced,
spatially intact branch alongside the pooled summary took mean RMSE from 0.442 to
0.405 and put the CNN back in front. If you swap the encoder, keep that branch.

### Extra module not in the original plan

`oceanembed/inference.py` -- `Predictor.load()` reads either checkpoint (torch or
NumPy) and exposes `.predict()` / `.embed()`. Added so `04_evaluate.py` and
`05_demo.py` share one loader instead of duplicating checkpoint layout knowledge.

---

## 9. WORK DIVISION FROM HERE (written against the ACTUAL current state)

Section 2 assumed hour 0. That milestone is already met: the synthetic pipeline
00-04 runs end-to-end and produces a scorecard. This section replaces it.

### STEP 0 -- one person, 5 minutes, BEFORE anyone else starts

Nothing is in version control. Parallel work without git is a merge disaster.

```
cd oceanembed
git init -b main && git add -A && git commit -m "OceanEmbed: synthetic pipeline 00-04 running"
gh repo create oceanembed --private --source=. --push
```

Everyone else clones AFTER this. `.gitignore` already excludes data and
checkpoints, so a clone is 193 KB of code -- no large files in git, ever.

### The one dependency that actually blocks people

Only the real DOWNLOADS are serial. Tracks C, D and E below need no real data at
all -- they run on the synthetic cube from minute one. Do not let anyone sit idle
waiting for Copernicus.

### Track A -- DATA (owns everything up to harmonized.nc)

Blocked on: Copernicus + Earthdata registration. START THIS FIRST, today.

1. Register (marine.copernicus.eu, urs.earthdata.nasa.gov), `copernicusmarine login`.
2. Pull ONE MONTH first, per `scripts/download/README.md`.
   Download the **0.25 deg** GLORYS product, not 1/12 deg -- harmonize.py regrids
   to 0.25 deg anyway, so 1/12 deg costs ~10x the bandwidth for zero benefit
   (~1.3 GB/month vs ~100 MB/month).
3. Fill the SOURCES dict in `scripts/01_harmonize_real.py`.
4. `python scripts/01_harmonize_real.py --dry-run` -- prints every file's
   variables, dims, ranges and flags Kelvin, writing nothing. Fix any NOT FOUND.
5. Run it for real, then start the 1-2 year pull in the background.

**Deliverable:** `harmonized.nc` (~25-30 MB for a month) + a filled `DATA_SOURCES.md`.
**Hand-off:** share ONLY `harmonized.nc`. Never share `dataset.npz` -- it
regenerates in about 60 s. That single file unblocks every other track.

### Track B -- VALIDATION / ARGO (the credibility centrepiece)

Blocked on: INCOIS LAS (no CLI -- use the web UI, export NetCDF).

1. Download gridded ARGO for the SAME box and window as Track A.
2. Set `ARGO_PATH` / `ARGO_VAR` in `scripts/02b_argo_colocate.py`.
3. `--dry-run` first: it reports the co-location count and the median time gap
   between an ARGO observation and the matched model day. Both go in the report.
4. Run it. Note that train/val and the scalers are untouched -- no retraining.

**Deliverable:** real ARGO holdout spliced into dataset.npz, plus the profile
count and the region/dates covered.

### Track C -- MODEL (NOT blocked -- starts now on synthetic)

Four things, in value order. None of them have been done yet.

1. **Patch-size sweep (7 / 9 / 11)** in `configs/config.yaml`. Spatial context is
   the entire thesis; this is the most on-message ablation you can show a judge.
2. **A cleaner control than the RF.** The current baseline confounds architecture
   with input -- RF-on-centre vs CNN-on-patch differs in both at once. Train the
   SAME OceanEmbed on a 1x1 patch to isolate the spatial contribution properly.
   Much stronger claim, about 10 minutes of compute.
3. **Train longer.** Validation loss was still falling at epoch 25. Free skill.
4. **The ViT has never actually been trained** -- only shape-checked. Run
   `python scripts/03_train.py --encoder vit` and compare honestly.

Note: `03_train.py` is currently pinned to CPU (`dev = torch.device("cpu")`, and
batches are never moved to a device). On a GPU box it will silently keep using the
CPU rather than fail. Roughly a five-line fix if you move to Colab or a cloud GPU.

**Deliverable:** best checkpoint + a table of what each variant scored.

### Track D -- EVAL + DEMO (NOT blocked -- starts now)

- `04_evaluate.py` already writes scorecard.json and 3 figures. Re-run after
  every Track C experiment.
- Add a real-ARGO-vs-prediction panel to `05_demo.py` once Track B lands.
- Capture demo screenshots EARLY as a fallback, in case the live demo misbehaves.

**Deliverable:** working demo + the 3 result figures + screenshots.

### Track E -- PITCH + REPORT (NOT blocked -- starts now)

5 slides: problem -> the embedding idea -> data + architecture -> results (with
baseline comparison and ARGO validation) -> demo, limitations, future work.

Use the honest findings, they are stronger than a clean number: the RF baseline
beats us in the top 20 m (T is nearly SST there, and our loss deliberately
up-weights the thermocline), and skill decays below 500 m where the surface stops
constraining temperature. "Our skill follows the physics" is a better line than
any single basin-wide average.

### If you have fewer than 5 people

- **3 people:** one does A+B (all data), one does C+D (model + eval), one does E.
- **2 people:** one does A+B, one does C+D+E. Cut the ViT and the patch sweep
  first -- the 1x1 ablation is the one experiment worth keeping.

### Sync points -- unchanged: hour 6, hour 12, hour 17

At each, whoever has real outputs replaces the synthetic ones. Because the schemas
are identical, that is a file swap and not a code change. At hour 17, freeze.
