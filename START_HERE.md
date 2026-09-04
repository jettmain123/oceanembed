# START HERE

Read this first, then open the file for your job.

- Person 1 -> `PERSON1_DATA.md` (gets the real ocean data)
- Person 2 -> `PERSON2_MODEL.md` (improves the model)
- Person 3 -> `PERSON3_RESULTS.md` (results, demo, slides)

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
Person 1  downloads real data  ->  makes harmonized.nc  ->  sends it to others
Person 2  improves the model   (works right away, does not wait for Person 1)
Person 3  makes results/slides (works right away, does not wait for Person 1)
```

Only Person 1 has to wait for anything. Person 2 and Person 3 start now using
the practice data, and simply re-run their commands later when the real data
arrives. Nothing in their work changes, because the real data has exactly the
same shape as the practice data.

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
