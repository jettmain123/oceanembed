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

## FIRST: check your months actually exist in these products

Copernicus splits most variables into a near-real-time stream and a reprocessed
one, and they cover different periods. October 2024 came from the NRT streams.
Older months may only exist in the reprocessed ones.

Run this before downloading anything:

```
for id in METOFFICE-GLO-SST-L4-NRT-OBS-SST-V2           cmems_obs-sl_glo_phy-ssh_nrt_allsat-l4-duacs-0.25deg_P1D           cmems_obs-mob_glo_phy-sss_my_multi_P1D           cmems_mod_glo_phy_my_0.25deg_P1D-m ; do
  echo "=== $id"
  copernicusmarine describe --dataset-id $id 2>/dev/null | grep -iE '"(start|end)"|coordinate_id.: .time' | head -6
done
```

**If a product does not cover your months, say so in the group chat. Do NOT
silently switch to a different product** -- the two streams disagree slightly,
and mixing them across months puts artificial jumps into the training data that
the model will happily learn as if they were real ocean signal.

If no single product covers all three years, we shorten the window instead. A
consistent 18 months beats an inconsistent 3 years.

---

## Download your months

One `subset` call per product per month. Replace `YYYY-MM` and the month's last
day. Everything lands in the same folders regardless of month.

```
M=2023-01           # <-- change this for each of your months
LAST=31             # last day of that month (28/29/30/31)
BOX="--minimum-longitude 45 --maximum-longitude 105 --minimum-latitude 5 --maximum-latitude 30"
DATES="--start-datetime ${M}-01 --end-datetime ${M}-${LAST}"

# SST -- OSTIA (Kelvin; harmonize converts it)
copernicusmarine subset -i METOFFICE-GLO-SST-L4-NRT-OBS-SST-V2   --variable analysed_sst $BOX $DATES   -o data/raw/ostia --output-filename ostia_$M.nc

# Sea level anomaly -- DUACS
copernicusmarine subset -i cmems_obs-sl_glo_phy-ssh_nrt_allsat-l4-duacs-0.25deg_P1D   --variable sla $BOX $DATES   -o data/raw/duacs --output-filename duacs_$M.nc

# Salinity
copernicusmarine subset -i cmems_obs-mob_glo_phy-sss_my_multi_P1D   --variable sos $BOX $DATES   -o data/raw/sss --output-filename sss_$M.nc

# GLORYS -- the training target. 0.25 deg, thetao only, capped at 1100 m.
# We regrid to 0.25 deg anyway, so 1/12 deg costs ten times the bandwidth for
# nothing. Confirm the 0.25 deg id in the catalogue; they get renamed.
copernicusmarine subset -i cmems_mod_glo_phy_my_0.25deg_P1D-m   --variable thetao $BOX $DATES   --minimum-depth 0 --maximum-depth 1100   -o data/raw/glorys --output-filename glorys_$M.nc

# Currents -- OSCAR
podaac-data-downloader -c OSCAR_L4_OC_FINAL_V2.0 -d data/raw/oscar   -sd ${M}-01T00:00:00Z -ed ${M}-${LAST}T23:59:59Z -b="45,5,105,30"

# Winds -- CCMP (6-hourly; harmonize averages it to daily)
podaac-data-downloader -c CCMP_WINDS_10M6HR_L4_V3.1 -d data/raw/ccmp   -sd ${M}-01T00:00:00Z -ed ${M}-${LAST}T23:59:59Z -b="45,5,105,30"
```

Expect roughly 150 MB per month across all six products. If GLORYS alone is
coming down at hundreds of MB per month, you are on the 1/12 degree product or
you forgot `--variable thetao`.

If a download dies partway, just run it again -- `subset` overwrites cleanly, and
the PODAAC downloader skips files it already has.

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
