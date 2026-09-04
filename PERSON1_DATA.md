# PERSON 1 - REAL DATA

**Your job:** download the real ocean data and turn it into one file the others
can use.

**Start today.** The websites need you to register, and approval is not always
instant. Everyone else can work without you, but nobody can show REAL results
until you finish. You are the long pole.

Do `START_HERE.md` first if you have not.

---

## Step 1 - Register on three websites

Do all three now, before downloading anything.

1. **Copernicus Marine** - https://marine.copernicus.eu
   Gives: sea surface temperature, saltiness, sea height, and GLORYS (the deep
   temperature data we train the model on).
2. **NASA Earthdata** - https://urs.earthdata.nasa.gov
   Gives: ocean currents (OSCAR) and wind (CCMP).
3. **INCOIS** - http://las.incois.gov.in
   Gives: ARGO float data. This is the most important one for our credibility,
   because it is real measurements from real floating instruments in the sea.

## Step 2 - Install the download tools

```
pip install copernicusmarine podaac-data-subscriber
```

```
copernicusmarine login
```

It will ask for the username and password you just registered with.

## Step 3 - Download ONE MONTH first

Do not download a year yet. One month is enough to check that everything works.
A year takes hours and would waste that time if a variable name is wrong.

All the exact commands are in `scripts/download/README.md`. Copy them from
there. Put each product in its own folder:

```
data/raw/ostia/     sea surface temperature
data/raw/sss/       saltiness
data/raw/duacs/     sea height
data/raw/oscar/     currents
data/raw/ccmp/      wind
data/raw/glorys/    deep temperature (what we train on)
data/raw/argo/      ARGO floats (what we test on)
```

Many `.nc` files inside each folder is normal and fine. Usually one file per
day. Do not rename them or merge them.

### One important money-saving change

`scripts/download/README.md` mentions the GLORYS product at 1/12 degree
resolution. That is about **1.3 GB per month**.

Download the **0.25 degree** version instead - roughly **100 MB per month**.

Why: our code shrinks everything to 0.25 degree anyway before using it. So
downloading the sharper version means downloading 10 times more data and then
throwing the extra detail away. Look in the Copernicus catalogue for the 0.25
degree GLORYS reanalysis (the id looks something like
`cmems_mod_glo_phy_my_0.25deg_P1D-m`, but Copernicus renames these, so search
the catalogue rather than trusting that exact text). The variable inside is
still called `thetao`.

Keep the depth limit at 0 to 1100 m. We only need down to 1000 m.

## Step 4 - Tell the code what your files are called

Open `scripts/01_harmonize_real.py`. Near the top there is a clearly marked
section called `EDIT BLOCK`. It looks like this:

```
SOURCES = {
    "sst":   ("data/raw/ostia/*.nc",  None),
    "sss":   ("data/raw/sss/*.nc",    None),
    ...
}
```

The left side (`"sst"`, `"sss"`) is our internal name - **never change those**.
The first thing in brackets is where your files are. The `None` means "figure
out the variable name automatically".

Usually you can leave it all alone. Only change something if step 5 complains.

## Step 5 - Check before running (this writes nothing)

```
python scripts/01_harmonize_real.py --dry-run
```

This opens every file and prints what is inside: the variable names, the size,
and the range of values. It creates nothing, so it is completely safe to run as
many times as you like.

Look for the word `NOT FOUND`. If you see it, that product's variable has an
unexpected name. The printout lists the names that ARE in the file - put the
correct one in the EDIT BLOCK where the `None` is, then run `--dry-run` again.

Keep going until nothing says NOT FOUND.

## Step 6 - Run it for real

```
python scripts/01_harmonize_real.py
```

This makes `data/processed/harmonized.nc`. It puts all seven surface
measurements plus the deep temperature onto one common grid, converts
temperature from Kelvin to Celsius automatically, and makes everything daily.

At the end it prints a table of how much data is missing. **Copy that table into
`DATA_SOURCES.md`.** The judges will ask, and it is annoying to reconstruct later.

## Step 7 - Send the file to the others

Send **only** this one file:

```
data/processed/harmonized.nc
```

It should be around 25-30 MB for one month. Google Drive or WhatsApp is fine.

Do not send `dataset.npz` - it is 175 MB and the others can rebuild it in about
a minute from the file you just sent.

## Step 8 - ARGO (do this too, it matters most)

ARGO has no download command, so use the website: http://las.incois.gov.in

Pick the gridded ARGO temperature product. Set the area to 45-105 East and 5-30
North. Set the dates to match what you downloaded in step 3. Export as NetCDF
into `data/raw/argo/`.

Then:

```
python scripts/02b_argo_colocate.py --dry-run
```

```
python scripts/02b_argo_colocate.py
```

Write down the two numbers it prints: how many ARGO profiles matched, and the
median gap in days. Both go in the report.

Why this matters: GLORYS is a computer model of the ocean. ARGO is real
instruments floating in real water. Testing against ARGO proves our model works
on reality, not just on another computer's guess.

## Step 9 - Start the big download in the background

Now that one month works, repeat step 3 with a longer date range (1 to 2 years)
and let it run overnight. Make sure the dates you pick exist in ALL products.

---

## You are done when

- `data/processed/harmonized.nc` exists and Person 2 and Person 3 have it
- `DATA_SOURCES.md` is filled in with product names, dates, and the missing-data
  table
- ARGO is downloaded and `02b_argo_colocate.py` has run

## If something goes wrong

**"no files match"** - your folder is empty or the path is wrong. Check the
folder actually contains `.nc` files.

**"could not find a variable for sst"** - the file uses a different name. The
error prints every name in the file. Put the right one in the EDIT BLOCK.

**Temperature looks like 300 instead of 27** - that is Kelvin, and it is normal.
The code converts it automatically. Nothing to do.

**Download is extremely slow** - shrink the area. Bay of Bengal only
(80-100 East, 10-25 North) is an acceptable smaller version. Tell the others,
because they must use the same area.
