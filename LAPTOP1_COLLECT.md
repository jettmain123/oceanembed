# LAPTOP 1 -- DATA COLLECTION

**Your months: December 2023, February 2024**

Download those months, harmonize them, upload one file. Nothing else.

## Why you download whole months, not a region

Every day needs all seven products covering the whole box, so splitting the map
between laptops would mean stitching regions together -- the pipeline does not do
that, and one laptop failing would break every single day. Splitting by MONTH
means your cube is complete and valid on its own, and a laptop dropping out costs
months, not the whole dataset.

## Before you start (once)

Accounts -- both are free and you need both:
- https://marine.copernicus.eu  (SST, SSS, SLA, GLORYS)
- https://urs.earthdata.nasa.gov  (currents, winds)

```
git clone https://github.com/jettmain123/oceanembed.git
cd oceanembed
pip install -r requirements.txt
pip install copernicusmarine podaac-data-subscriber
copernicusmarine login
```

**Do not edit `configs/config.yaml`.** Every laptop must use the identical grid
or the cubes will not merge.

## IMPORTANT: use the 0.25 degree GLORYS

The October pull used GLORYS at 1/12 degree, which is 21 MB per day. We shrink
everything to 0.25 degree anyway, so that resolution is thrown away immediately.
The 0.25 degree product is about 2 MB per day -- ten times smaller, identical
result.

Search the Copernicus catalogue for the 0.25 degree GLORYS reanalysis (the id
looks like `cmems_mod_glo_phy_my_0.25deg_P1D-m`; they rename these, so confirm
it in the catalogue). The variable is still `thetao`.

If you genuinely cannot find it, use 1/12 degree but tell the team -- it changes
the download time a lot.

## Use the SAME product for every month

Copernicus publishes a near-real-time stream and a reprocessed one, and they do
not agree exactly. Mixing them across months puts artificial jumps in the data
that the model will happily learn.

October 2024 used the NRT streams (`nrt_global_allsat_phy_l4_*` for SLA,
`mercatorglorys12v1_*` for GLORYS). **Use the same ones.** If a month you are
assigned is not available on that stream, say so rather than silently switching.

## Download your months

For each month, run the six commands in `scripts/download/README.md` with your
own `--start-datetime` and `--end-datetime`. Everything goes into the same
folders regardless of month:

```
data/raw/ostia/   data/raw/sss/    data/raw/duacs/
data/raw/oscar/   data/raw/ccmp/   data/raw/glorys/
```

Roughly 5 MB per day, so a month is about 150 MB.

## Check, harmonize, upload

```
python scripts/00_check_real_data.py
```

Fix anything marked FAIL. It should end with READY. Then:

```
python scripts/01_harmonize_real.py
```

Read the profile it prints at the end. It **must** fall from about 28-30 degC at
the surface to roughly 5-9 degC at 1000 m. If it rises anywhere, stop and report
it -- do not upload.

Rename your cube so we can tell them apart, then upload it to the shared Google
Drive folder `oceanembed/harmonized/`:

```
mv data/processed/harmonized.nc harmonized_dec2023_feb2024.nc
```

Upload `harmonized_dec2023_feb2024.nc` (about 46 MB) to Drive.
That single file is your entire deliverable -- do not upload raw downloads, and
do not upload `dataset.npz`.

## Then tell the team

Post in the group:

- which months you completed
- the missing-data percentages that `01_harmonize_real.py` printed
- the surface and 1000 m values from the profile

## If something goes wrong

**A month is missing from the NRT stream** -- report it, do not substitute a
different product.

**`00_check_real_data.py` says FAIL** -- read the line. It names the product and
the reason. Most often a folder is empty because a download quietly failed;
re-run that product's command.

**The profile rises with depth** -- stop. Something is wrong with the GLORYS
download for that month. Report it.

**Download is very slow** -- confirm you are on the 0.25 degree GLORYS. That is
usually the cause.
