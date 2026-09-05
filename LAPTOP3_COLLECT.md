# LAPTOP 3 -- DATA COLLECTION

**Your months: January, April, July and October 2024**

Yes, October 2024 again. The first pull used the near-real-time products, and
everything else now uses the reprocessed ones. Re-downloading it on the same
stream as the other eleven months removes the only inconsistency in the dataset.
It is one extra month and it is worth it.

You download those months, harmonize them yourself, and upload ONE file per
month. Nothing else moves between laptops.

---

## STOP -- read this before running anything

Two things changed from the October run. Both matter a lot at three-year scale.

### 1. Use `subset`, NOT `get`

`copernicusmarine get` downloads the **whole global file** and you throw 99% of
it away afterwards. `copernicusmarine subset` cuts the box out **on the server**.

| per day | `get` (global) | `subset` (our box) |
|---|---|---|
| OSTIA | ~150 MB | ~0.4 MB |
| GLORYS 1/12 deg, all variables | ~1.3 GB | ~20 MB |
| GLORYS 0.25 deg, thetao only | ~1.3 GB | **~2 MB** |

Three years with `get` is well over a terabyte. With `subset` the whole job is a
few GB. This is the difference between finishing and not finishing.

### 2. Ask for one variable and one depth range

GLORYS files carry about ten variables (`mlotst`, `zos`, `bottomT`, `siconc`,
`so`, `uo` ...). We use exactly one: `thetao`. Add `--variable thetao` and
`--maximum-depth 1100` and the file shrinks roughly tenfold again.

---

## Setup (once)

Accounts, both free:
- https://marine.copernicus.eu
- https://urs.earthdata.nasa.gov

```
git clone https://github.com/jettmain123/oceanembed.git
cd oceanembed
python -m venv .venv
source .venv/bin/activate       # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install copernicusmarine "boto3<1.36.0" podaac-data-subscriber
copernicusmarine login
```

Earthdata credentials for the PODAAC downloader:

```
cat > ~/.netrc << 'EOF'
machine urs.earthdata.nasa.gov
login YOUR_EARTHDATA_USERNAME
password YOUR_EARTHDATA_PASSWORD
EOF
chmod 600 ~/.netrc
```

```
mkdir -p data/raw/{ostia,duacs,sss,oscar,ccmp,glorys}
```

**Do not edit `configs/config.yaml`.** Every laptop must use the same grid or the
cubes will not merge.

---

## Verified dataset coverage

Checked against the Copernicus catalogue, not assumed. These are the
**reprocessed / multi-year** streams, and all four cover Jan 2022 to Oct 2024:

| dataset | coverage |
|---|---|
| `METOFFICE-GLO-SST-L4-REP-OBS-SST` | 1981-10 .. 2026-03 |
| `cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D` | 1993-01 .. 2026-01 |
| `cmems_obs-mob_glo_phy-sss_my_multi_P1D` | 1993-01 .. 2024-12 |
| `cmems_mod_glo_phy_my_0.083deg_P1D-m` | 1993-01 .. 2026-06 |

The near-real-time streams do **not** reach back far enough -- OSTIA NRT starts
2024-01-17 and DUACS NRT starts 2022-10-04 -- which is why we use the
reprocessed ones throughout. `cmems_mod_glo_phy_my_0.25deg_P1D-m` does not
exist; 1/12 degree is the only GLORYS in this stream, roughly 0.5 GB per month
for `thetao` alone.

## Download your months

One command per month. It works out the last day itself, subsets server-side,
and prints a summary telling you if anything failed:

```
bash scripts/download/download_month.sh 2022-01
```

Repeat for each of your months.

---

## Check, harmonize, upload -- ONE MONTH AT A TIME

Do not download all your months and harmonize at the end. Do one month fully,
upload it, then start the next. That way a problem shows up after 20 minutes
instead of after a day, and the training laptop can start early.

```
python scripts/00_check_real_data.py
```

Fix anything marked FAIL; it names the product and the reason. Then:

```
python scripts/01_harmonize_real.py
```

Read the profile it prints. It **must** fall from roughly 28-30 degC at the
surface to about 5-9 degC at 1000 m. If it rises anywhere, stop and report it --
do not upload.

```
mv data/processed/harmonized.nc harmonized_$M.nc
```

Upload `harmonized_$M.nc` (~23 MB) to the shared Drive folder
`oceanembed/harmonized/`. That file is your entire deliverable. Never upload raw
downloads or `dataset.npz`.

Then clear the raw folders before the next month, so `01_harmonize_real.py`
only ever sees one month at a time:

```
rm -f data/raw/*/*.nc
```

## Then post in the group

- which month you finished
- the missing-data percentages `01_harmonize_real.py` printed
- the surface and 1000 m values from the profile

## If something goes wrong

**"no files match"** -- a download failed silently. Check the folder actually
has `.nc` files.

**A product does not cover your month** -- report it. Do not substitute.

**The profile rises with depth** -- stop, something is wrong with that month's
GLORYS. Report it.

**Downloads are enormous** -- you are using `get` instead of `subset`, or you
left out `--variable thetao`.
