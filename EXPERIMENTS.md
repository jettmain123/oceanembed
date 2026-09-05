# Experiment results (Person 2)

All on the SYNTHETIC cube. See the warning in `PERSON2_MODEL.md`: these numbers
validate the method and the code, they are not a claim about the real ocean.
Re-run everything once `harmonized.nc` from real data exists.

Hardware: RTX 4080 Laptop. Holdout: 7200 profiles over 18 days never trained on.

---

## Final configuration (now the default in `configs/config.yaml`)

CNN encoder, patch 9, 120 epochs, seed 42.

| | OceanEmbed | RF baseline (centre cell only) |
|---|---|---|
| mean correlation | **0.986** | 0.936 |
| mean RMSE | **0.218 degC** | 0.619 degC |
| mean bias | +0.003 | +0.063 |
| mixed layer 0-50 m RMSE | **0.182** | 0.271 |
| thermocline 75-300 m RMSE | **0.323** | 1.170 |
| deep 500-1000 m RMSE | **0.084** | 0.215 |

**64.7 percent lower mean RMSE than the baseline**, and 64-77 percent lower at
every depth from 50 m to 500 m.

The one place the baseline still edges ahead is the top 10 m (0.119 vs 0.164),
and the gap is 0.045 degC - small enough to be irrelevant in practice. Near the
surface, temperature is essentially the surface temperature, so a point model
recovers it almost exactly. From 20 m down we win everywhere.

---

## EXPERIMENT 1 and 2 - field of view (the important one)

**Question: does seeing the neighbouring ocean actually help?**

Method note that makes this trustworthy: rebuilding the dataset at each patch
size would ALSO change which cells get sampled, because a 9x9 patch needs 4
cells of margin from the coast and a 1x1 needs none. The variants would then be
scored on different holdout sets. So `03c_patch_ablation.py` builds the dataset
ONCE at patch 11 and centre-crops it. Identical samples, identical targets,
identical training days - the only thing that changes is how far the model sees.

25 epochs each.

| patch | mean RMSE | vs patch 1 | thermocline RMSE | vs patch 1 | corr | params |
|---|---|---|---|---|---|---|
| 1 (centre only) | 0.789 | - | 1.315 | - | 0.884 | 452k |
| 3 | 0.562 | 28.8% | 0.878 | 33.3% | 0.939 | 484k |
| 5 | 0.556 | 29.5% | 0.825 | 37.3% | 0.941 | 550k |
| 7 | 0.460 | 41.7% | 0.648 | 50.7% | 0.956 | 648k |
| **9** | **0.420** | **46.8%** | **0.553** | **57.9%** | **0.966** | 779k |
| 11 | 0.408 | 48.3% | 0.553 | 58.0% | 0.967 | 943k |

**The claim this supports:** same model, same data, same training, same samples.
The only difference is whether it can see the surrounding ocean. Seeing it cuts
error by 48 percent overall and 58 percent in the thermocline.

**It saturates at 9.** Going 9 to 11 improves the thermocline by 0.1 percent
while costing 21 percent more parameters, so 9 is the default.

**Honest caveat:** patch size and parameter count grow together, so this is not
a perfectly pure test of field of view alone. The saturation argues against a
capacity explanation though - if extra parameters were driving the gain, patch
11 should have kept improving, and it did not.

---

## EXPERIMENT 3 - training length

This was the single biggest win, bigger than any architecture change.

| epochs | mean RMSE | thermocline | mixed layer |
|---|---|---|---|
| 25 | 0.441 | 0.552 | 0.491 |
| 60 | 0.272 | 0.367 | 0.269 |
| **120** | **0.218** | **0.323** | **0.182** |

The original 15-25 epoch setting was badly undertrained. It also explains a
result we previously reported as a limitation: the baseline used to beat us
across the whole mixed layer. It no longer does. That was undertraining, not a
property of the architecture.

Validation loss is flat over the last 10 epochs at 120, so this is converged.
`configs/config.yaml` now defaults to 120.

---

## EXPERIMENT 4 - CNN vs ViT

The ViT looked better at first (0.227 vs 0.272 at 60 epochs), so we checked
whether that was real by running three seeds each.

| encoder | seed 42 | seed 43 | seed 44 | mean | spread |
|---|---|---|---|---|---|
| CNN @ 120 epochs | 0.218 | 0.228 | 0.228 | 0.225 | 0.010 |
| ViT @ 60 epochs | 0.228 | 0.237 | 0.206 | 0.224 | 0.031 |

**They are the same within noise.** The apparent ViT win was a single-seed
artefact. The ViT is three times more variable across seeds, and it overfits if
trained as long as the CNN (0.266 at 120 epochs, worse than its own 60-epoch
result).

**Decision: keep the CNN.** Equal accuracy, three times more stable, faster to
train, simpler to explain.

This is worth a slide. "We tested a transformer, measured it properly across
seeds, and found no real difference, so we kept the simpler model" is a stronger
statement than quietly picking whichever number looked best once.

---

## Reproducing any of this

```
python scripts/03c_patch_ablation.py --sizes 1,3,5,7,9,11 --epochs 25
python scripts/03_train.py --encoder cnn --epochs 120 --seed 42
python scripts/03_train.py --encoder vit --epochs 60 --seed 43
python scripts/04_evaluate.py
```

Training takes about 100 seconds for 120 epochs on a GPU, and roughly 45 minutes
on CPU. `03_train.py` prints which device it is using.

## What to do when the real data lands

Run all of it again. Expect worse numbers - real oceans are messier than a
formula - and expect the ordering of the experiments to possibly change. The
field-of-view ablation is the one to repeat first, because it is the experiment
the whole architecture argument rests on.
