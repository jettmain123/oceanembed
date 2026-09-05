# LAPTOP 2 -- DATA COLLECTION

**Your months: January, April, July and October 2023**

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

## Download everything -- one command

```
python scripts/download/collect.py --laptop 2
```

**On Windows you must use this Python version, not the .sh one.** Typing `bash`
in PowerShell launches WSL, which is a separate Linux system that cannot see
your Windows virtual environment -- every download then fails with
"command not found" and the run looks broken when it is only in the wrong place.
The Python script runs in whichever Python you launch it with, so that cannot
happen. It also works on macOS and Linux, so everyone can use it.

It checks your logins before starting and tells you exactly what is missing,
rather than failing six times with an empty file.

The shell version still exists if you prefer it on macOS:
`bash scripts/download/laptop2.sh`

That is the whole job. For each of your months it downloads all six products,
checks them, harmonizes them, verifies the profile decreases with depth, and
stages the cube in `to_upload/`. Expect a few hours, mostly GLORYS.

**Safe to stop and rerun.** Months already finished are skipped, so a closed
laptop or a dropped connection costs you one month, not the whole run. If
something fails it says which month and why, and keeps going with the rest.

Per-month logs land in `logs/` if you need detail.

When it finishes, upload everything in `to_upload/` to the shared Drive folder
`oceanembed/harmonized/`. Those files are your entire deliverable -- never
upload raw downloads or `dataset.npz`.

### If you would rather do one month at a time

```
bash scripts/download/download_month.sh 2022-04
python scripts/00_check_real_data.py
python scripts/01_harmonize_real.py
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
