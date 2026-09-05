# LAPTOP 4 -- TRAINING (the RTX 4080)

You collect nothing. You wait for cubes, merge them, train, and report numbers.

## The collection plan

Twelve months spread across three years -- one per season, per year:

| year | months | who |
|---|---|---|
| 2022 | Jan, Apr, Jul, Oct | Laptop 1 |
| 2023 | Jan, Apr, Jul, Oct | Laptop 2 |
| 2024 | Jan, Apr, Jul | Laptop 3 |
| 2024 | **Oct** | already have it |

About 360 days covering every season in every year.

**Why four months a year and not all 36 months.** Consecutive days are nearly
identical -- 15 April tells the model almost the same thing as 16 April. What
breaks the climatology shortcut is spanning SEASONS and YEARS, not filling every
date. Four scattered months per year gets essentially the same information for a
third of the download, transfer and training cost. If collection turns out to be
fast, adding more months is easy; the merge script does not care.

**Why a year each rather than a season each.** If a laptop drops out we lose one
year and keep two full seasonal cycles. Splitting by season instead would mean
losing, say, every winter.

**The product-consistency risk.** Copernicus runs a near-real-time stream and a
reprocessed one, and they disagree slightly. October 2024 came from NRT. If the
same product does not cover 2022 as well, the collectors have been told to report
it rather than substitute -- mixing streams would put artificial jumps in the
data that the model would learn as real signal. **A consistent 18 months beats an
inconsistent 3 years.** Expect to hear about this; decide as a team.

## As cubes arrive

Put them in `data/incoming/` (any that have arrived -- you do not need all of
them to start):

```
mkdir -p data/incoming
```

```
python scripts/01c_merge_harmonized.py data/incoming/*.nc
```

That concatenates along time, drops duplicate dates, reconciles the land mask and
prints the merged profile. **Check the profile falls with depth** before going on.

Then:

```
python scripts/02_build_dataset.py --max-per-day 2000
```

```
python scripts/03_train.py --seed 42
```

```
python scripts/03b_baseline.py && python scripts/04_evaluate.py
```

Roughly two minutes end to end on the 4080.

## The number that decides everything

From `04_evaluate.py`, the **skill above climatology** table.

October alone gave:

| depth | skill vs climatology |
|---|---|
| 0-150 m | +0.48 to +0.68 |
| 200 m | +0.44 |
| 300 m | **-0.34** |
| 1000 m | **-1.01** |
| mean | **+0.284** |

Negative means the model is worse than simply predicting the average profile for
that location -- so below 300 m, the surface data is currently adding nothing.

**Watch the deep rows as months arrive.**

- Deep skill rises toward or past 0 -> more data is working. Ask for more months.
- Deep skill stays negative with 6 months of seasonal coverage -> the surface
  genuinely does not constrain 1000 m, which is a real physical finding. Stop
  downloading, report skill only where it exists, and say why.

Either answer is worth having. Do not keep downloading on hope.

## Re-run these too

Each time the dataset grows:

```
python scripts/03c_patch_ablation.py --sizes 1,3,5,9,11 --epochs 40
```

The field-of-view result is the core architecture claim. October gave 11%
(1x1 -> 9x9). See whether more seasons strengthen it.

```
python scripts/03b_baseline.py --features position
```

The climatology control. October: position alone scored 0.201 at depth while
adding every satellite field gave 0.203 -- the surface contributed nothing there.

## Retune as the data grows

Current settings (40 epochs, weight decay 1e-2) were tuned for 22 training days
and are deliberately heavily regularised. With 150+ training days you can afford
more capacity and longer training -- try `--epochs 120` and weight decay back at
1e-3, and compare. Watch for the train/val gap widening again.

## Keep every result

`04_evaluate.py` overwrites `scorecard.json` each run. After each dataset size:

```
cp outputs/scorecard.json outputs/scorecard_<n>months.json
```

Skill against months is a graph worth showing: it demonstrates the method
improving with data rather than being tuned into looking good.

## Files that matter

- `REAL_DATA_RESULTS.md` -- what October gave, and the caveats
- `outputs/scorecard.json` -- every number, per depth
- `outputs/skill_vs_depth.png` -- the main figure
