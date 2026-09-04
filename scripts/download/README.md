# Downloading the real data

Three accounts, all free. Register first, then subset by region and date --
**never download global files**, they are terabytes.

Domain: **5-30 N, 45-105 E**. Start with **one month** to read the variable names
and get the pipeline flowing; pull 1-2 years in the background afterwards.

Put everything under `data/raw/<product>/`. The globs in
`scripts/01_harmonize_real.py` (EDIT BLOCK) already expect those folders.

---

## 1. Copernicus Marine -- SST, SSS, SLA, and the GLORYS target

```bash
pip install copernicusmarine
copernicusmarine login          # one-time, stores credentials
```

Register at <https://marine.copernicus.eu>.

```bash
# ---- SST (OSTIA L4, 0.05 deg daily) -> data/raw/ostia/     NOTE: KELVIN
copernicusmarine subset \
  --dataset-id METOFFICE-GLO-SST-L4-NRT-OBS-SST-V2 \
  --variable analysed_sst \
  --minimum-longitude 45 --maximum-longitude 105 \
  --minimum-latitude 5 --maximum-latitude 30 \
  --start-datetime 2021-01-01 --end-datetime 2021-01-31 \
  --output-directory data/raw/ostia

# ---- Sea level anomaly (DUACS L4, 0.25 deg daily) -> data/raw/duacs/
copernicusmarine subset \
  --dataset-id cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.25deg_P1D \
  --variable sla --variable ugosa --variable vgosa \
  --minimum-longitude 45 --maximum-longitude 105 \
  --minimum-latitude 5 --maximum-latitude 30 \
  --start-datetime 2021-01-01 --end-datetime 2021-01-31 \
  --output-directory data/raw/duacs

# ---- Sea surface salinity (multi-observation L4) -> data/raw/sss/
copernicusmarine subset \
  --dataset-id cmems_obs-mob_glo_phy-sss_my_multi_P1D \
  --variable sos \
  --minimum-longitude 45 --maximum-longitude 105 \
  --minimum-latitude 5 --maximum-latitude 30 \
  --start-datetime 2021-01-01 --end-datetime 2021-01-31 \
  --output-directory data/raw/sss

# ---- GLORYS12V1 reanalysis: THE TRAINING TARGET -> data/raw/glorys/
# thetao on the native 50 levels; harmonize.py interpolates onto the 15 standard depths
copernicusmarine subset \
  --dataset-id cmems_mod_glo_phy_my_0.083deg_P1D-m \
  --variable thetao \
  --minimum-longitude 45 --maximum-longitude 105 \
  --minimum-latitude 5 --maximum-latitude 30 \
  --minimum-depth 0 --maximum-depth 1100 \
  --start-datetime 2021-01-01 --end-datetime 2021-01-31 \
  --output-directory data/raw/glorys
```

Dataset IDs change between Copernicus releases. If one 404s, search the catalogue
and keep the *variable* name -- that is what the loaders match on.

---

## 2. NASA PODAAC / Earthdata -- currents and winds

Register at <https://urs.earthdata.nasa.gov>, then either use the web subsetter
or `podaac-data-subscriber`:

```bash
pip install podaac-data-subscriber
# ---- OSCAR surface currents (0.25 deg) -> data/raw/oscar/   vars: u, v
podaac-data-downloader -c OSCAR_L4_OC_third-deg -d data/raw/oscar \
  -sd 2021-01-01T00:00:00Z -ed 2021-01-31T23:59:59Z \
  -b="45,5,105,30"

# ---- CCMP winds (0.25 deg, 6-hourly -> harmonize.py resamples to daily)
podaac-data-downloader -c CCMP_WINDS_10M6HR_L4_V3.1 -d data/raw/ccmp \
  -sd 2021-01-01T00:00:00Z -ed 2021-01-31T23:59:59Z \
  -b="45,5,105,30"
```

ERA5 `u10`/`v10` from the Copernicus Climate Data Store is a fine substitute for
CCMP -- `loaders.PRODUCT_VARS` already knows both names.

---

## 3. INCOIS Live Access Server -- gridded ARGO (the independent validation)

<http://las.incois.gov.in>

Web UI: choose the gridded ARGO temperature product, set the box to
45-105 E / 5-30 N, set the date range to match everything else, and export
NetCDF into `data/raw/argo/`.

This is the dataset that makes the results credible: the model never sees it in
training, and `scripts/02b_argo_colocate.py` splices it in as the holdout.

---

## Choosing the window

All products must overlap. Verify before pulling a year:

```bash
python - <<'PY'
import glob, xarray as xr
for d in ("ostia","duacs","sss","oscar","ccmp","glorys","argo"):
    fs = sorted(glob.glob(f"data/raw/{d}/*.nc"))
    if not fs: print(f"{d:8s} -- nothing downloaded"); continue
    ds = xr.open_mfdataset(fs, combine="by_coords")
    t = ds[[c for c in ds.coords if "time" in c.lower()][0]]
    print(f"{d:8s} {str(t.values.min())[:10]} .. {str(t.values.max())[:10]}  {list(ds.data_vars)[:4]}")
PY
```

Then split chronologically: earlier dates train, later dates validate, ARGO is
the independent test.

---

## After downloading

```bash
python scripts/01_harmonize_real.py --dry-run   # prints variables + ranges, writes nothing
# fix any "NOT FOUND" entries in the EDIT BLOCK, then:
python scripts/01_harmonize_real.py
python scripts/02_build_dataset.py
python scripts/02b_argo_colocate.py --dry-run
python scripts/02b_argo_colocate.py
python scripts/03_train.py
python scripts/03b_baseline.py
python scripts/04_evaluate.py
```

Record product name, version, DOI and access date in `DATA_SOURCES.md`, plus the
percent-missing table that `01_harmonize_real.py` prints at the end.
