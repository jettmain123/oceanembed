# Real data results -- October 2024, North Indian Ocean

First run on real satellite and reanalysis data. Read this instead of
`EXPERIMENTS.md`, which covers the synthetic dry run.

**Data:** 31 days (2024-10-01 to 10-31), 0.25 deg, 5-30N / 45-105E.
OSTIA SST, Copernicus SSS, DUACS SLA, OSCAR currents, CCMP winds; GLORYS thetao
as the target. Chronological split: 22 train / 4 val / 5 holdout days.
44,000 train and 10,000 holdout profiles. Trained on an RTX 4080 in ~37 s.

---

## Headline

Model and baseline both see **the seven surface fields and nothing else**.

| | OceanEmbed | RF baseline | improvement |
|---|---|---|---|
| mean correlation | **0.875** | 0.806 | |
| mean RMSE | **0.639 degC** | 0.826 | **22.6%** |
| mixed layer 0-50 m | **0.534** | 0.600 | 11% |
| thermocline 75-300 m | **0.900** | 1.225 | **26.5%** |
| deep 500-1000 m | **0.328** | 0.481 | 32% |

Best gains are at 200-500 m (36-48%), through the lower thermocline. The
baseline is marginally ahead only in the top 10 m, where temperature is
essentially the surface temperature and any method recovers it.

## Field-of-view ablation on real data

Identical samples, only the patch size changed (see `03c_patch_ablation.py`):

| patch | mean RMSE | thermocline | vs 1x1 |
|---|---|---|---|
| 1 (centre only) | 0.633 | 0.854 | - |
| 5 | 0.619 | 0.837 | 2.1% |
| **9** | **0.563** | **0.775** | **11.0%** |
| 11 | 0.580 | 0.790 | 8.3% |

Spatial context helps on real data, and 9x9 is still the best size. But the gain
is **11%, not the 48% we measured on synthetic data**. The synthetic figure was
inflated because the generator was built with a neighbourhood dependence -- we
were partly measuring our own assumption. 11% is the real number. Quote that one.

---

## THE IMPORTANT CAVEAT: one month is mostly climatology

This is the finding that should shape how every number here is read.

We ran a control: a RandomForest given **only latitude, longitude and day of
year** -- no ocean data whatsoever.

| feature set | mean RMSE | thermocline | deep 500-1000 m |
|---|---|---|---|
| position + date ONLY (no ocean data) | 0.700 | 1.006 | **0.201** |
| the 7 surface fields ONLY | 0.826 | 1.225 | 0.481 |
| both together | 0.555 | 0.761 | **0.203** |

Two things fall out of this:

**1. Below 500 m, the surface data adds nothing at all.** Position alone scores
0.201; adding every satellite field gives 0.203. All of the apparent deep-water
skill -- the impressive 0.96+ correlations -- is the model memorising a static
map, not inferring anything from the surface. That is physically reasonable: the
surface genuinely does not constrain the deep ocean over one month.

**2. With position included, the RF beats OceanEmbed** (0.555 vs ~0.60-0.63).
Trees are very good at memorising a lat/lon lookup table, and with only 31 days
the deep ocean barely changes, so memorising is close to optimal. Our model gains
little from position (0.639 without, ~0.63 with), because it is doing something
else: reading the surface state.

So the honest framing is:

> Given position, the task collapses into recalling a spatial climatology, and a
> RandomForest wins. Given only surface observations -- the actual problem -- our
> model beats it by 23%, and by 27% through the thermocline.

That is why `patch.add_coords` is now **false** on the real path: it keeps the
comparison like-for-like and stops climatology masquerading as skill.

**This is not a weakness to hide. It is a finding, and it is a better slide than
a big correlation number.** Any team quoting 0.96 correlation at 1000 m from one
month of data is quoting climatology and probably does not know it.

---

## Overfitting

22 training days is very little, and the days are highly correlated.

| epochs | train loss | val loss |
|---|---|---|
| 40 | 0.0209 | 0.0735 |
| 80 | 0.0096 | 0.0656 |
| 120 | 0.0072 | 0.0672 |

A 9x train/val gap by epoch 120. Settings that worked on 87 synthetic days
(120 epochs, weight decay 1e-4) overfit badly here. The config is now
**40 epochs, weight decay 1e-2**, which was the best of the six combinations
swept. Larger models and longer training both made it worse.

---

## What would actually improve this

**More months.** This is the one that matters, and it is a download, not a code
change. With one month the deep ocean is nearly static, so climatology explains
most of the variance and there is little dynamic signal to learn. With 6-12
months the model has to explain seasonal and mesoscale change, position stops
being a shortcut, and the comparison gets far more meaningful.

Everything else -- architecture, tuning, more parameters -- is second order
against that.

## Reproducing

```
python scripts/01_harmonize_real.py
python scripts/02_build_dataset.py --max-per-day 2000
python scripts/03_train.py --seed 42
python scripts/03b_baseline.py
python scripts/03b_baseline.py --features position     # the climatology control
python scripts/03c_patch_ablation.py --sizes 1,3,5,9,11 --epochs 40
python scripts/04_evaluate.py
```

Run-to-run spread on real data is roughly +/-0.03 degC mean RMSE, wider than the
+/-0.01 seen on synthetic. Quote your own `outputs/scorecard.json`.

---

# UPDATE: 10 months of 2022 (304 days)

Laptop 1 collected 2022-01-01 to 2022-10-31 continuously -- 304 days, one
consistent reprocessed stream. Trained on 219 days, held out the last 46.

**The numbers got worse, and that is the point.**

| | Oct 2024 (1 month) | 2022 Jan-Oct (10 months) |
|---|---|---|
| training days | 22 | 219 |
| holdout | 5 days, same month | 46 days, an unseen season |
| mean correlation | 0.868 | 0.800 |
| mean RMSE | 0.646 degC | 0.952 degC |
| thermocline RMSE | 0.881 | 1.265 |
| **skill vs climatology** | **+0.246** | **+0.074** |

## Why the drop is the honest number, not a regression

In the October run, training ended 22 October and the holdout was 27-31 October.
Five days later, same season, same monsoon phase. That is barely a generalisation
test at all.

With ten months, training covers January to early August and the holdout is
mid-September to October. The model has to predict a **season it never saw**.
That is the test that tells you whether the thing would work operationally, and
it is much harder.

**So the earlier +0.246 was optimistic.** Quote +0.07 to +0.10. It is the number
that survives contact with an honest split.

## What we tried, and what the data said

**Longer training, less regularisation.** The old settings (40 epochs, weight
decay 1e-2) were tuned for 22 days and badly underfit 219 -- validation loss was
still falling at the last epoch. 150 epochs at weight decay 1e-3 lifted skill
from +0.074 to +0.102 and took us from losing to the RandomForest to beating it
slightly. Run-to-run spread is about +/-0.03, so treat those as the same
ballpark.

**Giving the model the date made it WORSE.** Adding sin/cos(day-of-year) dropped
skill from +0.102 to **-0.246**. The reason is clean: training covers January to
August, so the day-of-year values for September and October are inputs the model
has never seen. Handing it the date turns the date into an out-of-distribution
feature and it extrapolates badly.

This flips once the record covers complete years. `patch.coord_vars` in the
config keeps the switch, with the measurement written next to it.

## What still holds

- **Below 300 m the model remains worse than climatology** (-0.27 at 300 m,
  -1.24 at 700 m). Ten times the data did not fix it. That increasingly looks
  physical rather than a data-volume problem: over these timescales the surface
  does not constrain the deep ocean, and no amount of satellite data will change
  that.
- **Above 200 m there is genuine skill**, +0.18 to +0.50, strongest at 125-150 m
  in the thermocline where the interesting variability lives.
- The RandomForest still wins in the **top 30 m**, where temperature is nearly
  the surface value and a point model recovers it almost exactly.

## What to collect next

Full calendar years, not more of the same months. The single biggest remaining
gap is that the model has never seen a November or December. Once training spans
complete years, the seasonal-extrapolation problem above disappears and the
day-of-year channels should start helping rather than hurting.


---

# THE HEADLINE: validated against real ARGO floats

Everything above is measured against GLORYS, which is a computer model. This is
measured against **real instruments in real water** -- 20,000 profiles from
gridded ARGO, restricted with `--holdout-only` to dates the model never trained
on.

| | RMSE vs ARGO | bias |
|---|---|---|
| **OceanEmbed** | **1.55 degC** | +0.93 |
| **GLORYS itself** | **1.50 degC** | +0.80 |

**We are 0.05 degC -- about 3% -- from GLORYS when both are judged against real
floats.**

That is the claim worth making. Our raw error against ARGO looks poor next to the
0.91 degC we score against GLORYS, but almost none of it is ours: GLORYS
disagrees with ARGO by 1.50 degC on its own, and we inherit that. We reproduce a
supercomputer reanalysis to within 3% of its own accuracy, in milliseconds, from
satellite data alone.

> "Our reconstruction is within 3% of GLORYS's own agreement with ARGO floats --
> we match a supercomputer reanalysis using only satellite inputs, in
> milliseconds."

## The caveats to state before anyone asks

- Both we and GLORYS run about **+0.8 to +0.9 degC warm** against this ARGO
  product. That bias is GLORYS's, and we have faithfully learned it. We cannot
  be better than what we were trained on.
- The INCOIS gridded ARGO is itself an **objective analysis**, not raw floats, so
  part of that 1.5 degC is the gridding rather than either model being wrong.
- **GLORYS assimilates ARGO.** So this is not a fully independent test of the
  physics -- it is an honest end-to-end test of the surrogate. Say so first
  rather than being asked.

## Reproducing

```
python scripts/02b_argo_colocate.py --dry-run        # shows which dates are clean
python scripts/02b_argo_colocate.py --holdout-only   # splice in only those
python scripts/04_evaluate.py
```

`--holdout-only` matters. Of 108 ARGO dates in the 2022 record, only 6 fall on
holdout days; the rest land on days the model trained on. Their measurements
would still be independent, but the surface patches would not be, and the score
would flatter us.

After this, `dataset.npz` holds the ARGO holdout, so `04_evaluate` reports ARGO
numbers. Re-run `02_build_dataset.py` to get the GLORYS holdout back.
