# START HERE

Read this first, then open the file for your job.

### Current phase: collecting a full seasonal cycle

- Laptop 1 -> `LAPTOP1_COLLECT.md`  (Dec 2023, Feb 2024)
- Laptop 2 -> `LAPTOP2_COLLECT.md`  (Apr 2024, Jun 2024)
- Laptop 3 -> `LAPTOP3_COLLECT.md`  (Aug 2024, Sep 2024)
- Laptop 4 -> `LAPTOP4_TRAIN.md`    (merges, trains, reports -- the RTX 4080)

Each collecting laptop downloads WHOLE MONTHS of all seven products, harmonizes
them itself, and uploads ONE file (~23 MB per month) to the shared Google Drive
folder `oceanembed/harmonized/`. Nothing else is ever transferred.

Later: `WEBSITE.md` for the frontend, `PERSON3_RESULTS.md` for slides.
Current results on real data: `REAL_DATA_RESULTS.md`.

## What this project does

We take satellite data from the ocean SURFACE (temperature, saltiness, sea
height, currents, wind) and predict the temperature DEEP UNDERWATER, at 15
depths from 0 m down to 1000 m. Satellites only see the surface, so this fills
in what they cannot see.

## Everyone does this first (about 5 minutes)

Step 1 - get the code:

```
git clone https://github.com/jettmain123/oceanembed.git
cd oceanembed
```

Step 2 - install what it needs:

```
pip install -r requirements.txt
```

Step 3 - make the practice data and check everything works:

```
python scripts/00_make_synthetic.py
python scripts/01_harmonize.py
python scripts/02_build_dataset.py
```

This takes about 3 minutes and creates fake-but-realistic ocean data on your
laptop. It is NOT the real data. It exists so everyone can work immediately
instead of waiting for downloads.

If those three commands finish without an error, you are ready.

## Important: the big data files are NOT on GitHub

GitHub refuses any file bigger than 100 MB, and our data files are 100-175 MB.
That is fine, because the code CREATES the data. That is what step 3 above does.

So: never try to push `.nc`, `.npz` or `.parquet` files. They are already
ignored by git.

## How the work fits together

```
Person 1  Copernicus downloads (SST, SSS, SLA, GLORYS)  ->  harmonized.nc
Person 4  PODAAC downloads (currents, winds) + ARGO     ->  sends to Person 1
             then switches to building the website
Person 2  trains the model                (never waits)
Person 3  results, slides, report         (never waits)
```

Persons 1 and 4 download from DIFFERENT servers with DIFFERENT logins, so they
genuinely run in parallel. Splitting Copernicus across two machines would just
throttle one account.

Only Person 1 has to wait for anything. Person 2 and Person 3 start now using
the practice data, and simply re-run their commands later when the real data
arrives. Nothing in their work changes, because the real data has exactly the
same shape as the practice data.

## Moving data between laptops

Only two transfers happen, and only raw files ever need to move once.

```
L2 (PODAAC: currents + winds)
      |  transfer 1: raw .nc folders, ONCE
      v
L1 (Copernicus: SST, SSS, SLA, GLORYS)
      |  runs 01_harmonize_real.py  ->  harmonized.nc
      |  transfer 2: harmonized.nc only
      v
L3 (the 4080)  builds dataset.npz locally and trains
```

**Never transfer `dataset.npz`.** It is the biggest file in the project and L3
rebuilds it from `harmonized.nc` in under a minute.

Sizes, per 100 days of data:

| moving | what | size per 100 days |
|---|---|---|
| L2 -> L1 | raw OSCAR + CCMP | ~180 MB |
| L1 -> L3 | harmonized.nc | ~75 MB |
| never | dataset.npz | ~700 MB |

How to move it: Google Drive or OneDrive for anything under a few GB. On the
same wifi it is faster to run `python -m http.server 8000` in the folder on the
sending laptop and download from `http://<their-ip>:8000` on the other. A USB
stick beats both for multi-GB transfers.

`harmonized.nc` fits in GitHub (100 MB limit) up to roughly 130 days of data.
Past that, use Drive.

### Download scattered months, not consecutive days

Two full years of daily files is ~3.7 GB downloaded and ~550 MB of
`harmonized.nc` to move around. You do not need it.

Consecutive days are nearly identical -- 15 October and 16 October tell the model
almost the same thing. What breaks the climatology shortcut is SPANNING SEASONS,
not filling in every day.

So pull **6-8 separate months spread across two years** (e.g. Jan, Mar, May,
Jul, Sep, Nov). Each one is a single command exactly like the October pull you
already did. That gives roughly 240 days covering every season, for about a
third of the download, transfer and training cost of 730 consecutive days.

Note for whoever trains: with scattered months the chronological split holds out
the LAST month, so the test becomes "does it work on a season it never saw".
That is a harder and more honest test than holding out five days in the middle
of a month -- expect lower numbers, and say why.

## The one file that gets shared

When Person 1 finishes, they send ONE file to Person 2 and Person 3:

```
data/processed/harmonized.nc     (about 25-30 MB for one month of data)
```

Send it over Google Drive or WhatsApp. Whoever receives it puts it in their own
`data/processed/` folder, replacing the practice version, then re-runs their
commands. That is the whole handover.

## Saving your work

Work on your own branch so you do not overwrite each other:

```
git checkout -b person1     (or person2 / person3)
git add -A
git commit -m "what you did"
git push -u origin person1
```
