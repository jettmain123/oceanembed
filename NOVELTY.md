# What we do that GLORYS does not

We train on GLORYS, so we cannot be more accurate than GLORYS -- at best we
approximate it quickly. Every claim below is therefore about something GLORYS
does not give you, not something we do better.

All numbers measured on the October 2024 holdout: 10,000 profiles, 5 days the
model never saw.

---

## 1. We output the number forecasters act on, not a temperature field

GLORYS gives you T(z). INCOIS issues cyclone advisories from **tropical cyclone
heat potential** -- the heat stored above the 26 degC isotherm, which is what
determines whether a storm intensifies, rather than sea surface temperature.

```
TCHP = rho * cp * integral from 0 to D26 of (T(z) - 26) dz
```

Two spots can share an identical satellite SST image and behave completely
differently: a 100 m warm layer keeps feeding a cyclone, a 20 m one lets the
storm churn cold water up and weaken itself.

| product | correlation | RMSE | reference mean |
|---|---|---|---|
| TCHP | 0.816 | 15.3 kJ/cm2 | 68.0 kJ/cm2 |
| D26 (26 degC isotherm depth) | 0.790 | 12.5 m | 69.1 m |
| MLD (mixed layer depth) | 0.719 | 10.1 m | 36.8 m |

**The number that matters most** is not RMSE but whether we would raise the same
alarm. At the 50 kJ/cm2 rapid-intensification threshold:

| | |
|---|---|
| precision | **0.951** |
| recall | **0.942** |
| F1 | **0.946** |

We agree with GLORYS on the cyclone-relevant call about 94% of the time. At the
higher 80 and 100 kJ/cm2 thresholds it degrades (F1 0.67 and 0.55) -- report
that, it is where more training data should help most.

**A convenient accident of physics.** D26 in the North Indian Ocean sits at
50-150 m, so TCHP is an integral over exactly the depth band where the model has
genuine skill (+0.44 to +0.68 above 200 m). Our known weakness below 300 m does
not touch it.

`scripts/07_products.py` · `oceanembed/products.py` · live on the website as its
own map layer with the 50 kJ/cm2 threshold flagged.

---

## 2. It still works when a satellite is down

Products have gaps. SMAP salinity is patchy, altimeters miss swaths, sensors get
decommissioned. An assimilation system needs its full input stack and a
supercomputer rerun. A learned model can simply degrade.

We train with `channel_dropout: 0.15` -- each input variable is blanked entirely,
15% of the time. Measured by removing one input at a time from the holdout:

| missing input | trained normally | trained with dropout |
|---|---|---|
| sea level (SLA) | **+59.0% worse** | **+19.2% worse** |
| salinity (SSS) | +57.5% | +14.1% |
| SST | +7.9% | +3.7% |
| currents (u) | +10.4% | +6.7% |
| winds | ~0% | ~0% |

**Three to four times more robust, and slightly MORE accurate with everything
present** (mean RMSE 0.639 to 0.631) -- blanking channels also regularises a
model that was overfitting 22 days of training data.

Two findings worth saying out loud:

- **Sea level anomaly is the single most important input.** Physically right:
  SLA is a direct expression of thermocline displacement, which is what sets the
  subsurface structure.
- **Winds contribute essentially nothing** over a one-month record. Also
  defensible -- the ocean's response to wind is cumulative and lagged, and we
  feed the model instantaneous wind. Worth revisiting with more months.

`scripts/03d_sensor_robustness.py`

---

## 3. Subsurface anomaly detection, daily

Marine heatwave monitoring is almost entirely surface-based, because nobody has
daily subsurface fields. Subsurface warm anomalies are what stress fisheries and
coral, and they are poorly observed.

We compute a per-location climatology from the training days and report the
departure from it at every depth:

| depth | anomaly correlation |
|---|---|
| 0-20 m | 0.72-0.79 |
| 100-150 m | **0.83-0.85** |
| 300 m | 0.42 |
| 1000 m | 0.24 |

Detecting a warm anomaly 100 m down, daily, from satellites alone is something
surface-only monitoring cannot do at all. Note that skill is HIGHEST at 100-150 m
-- the thermocline, where the interesting variability lives.

**Be honest about the limitation.** A real marine-heatwave definition needs the
exceedance to persist five days and needs a climatology built from years. Ours is
22 days of October. The exceedance base rate comes out near 50% instead of the
10% a true 90th-percentile climatology would give, which is exactly what you
would expect from a short, non-representative baseline. **Present this as a
demonstrated capability, not a validated product.** It becomes real once the
12-month collection lands.

---

## What none of this needs

No extra data. TCHP, D26, MLD and anomalies are all post-processing of profiles
we already predict; the robustness result is a training flag. Everything above
runs on the October 2024 cube already in the repo.

The one that improves most with the 12-month collection is anomaly detection,
because it is the only one that depends on having a real climatology.

## Reproducing

```
python scripts/07_products.py              # TCHP, D26, MLD, anomalies
python scripts/03d_sensor_robustness.py    # what each missing input costs
python scripts/03_train.py --channel-dropout 0    # the un-robust comparison
```
