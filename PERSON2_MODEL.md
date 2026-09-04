# PERSON 2 - THE MODEL

**Your job:** make the model better, and prove WHY it works.

**You are not waiting for anybody.** Start right now with the practice data.
When Person 1 sends the real file later, you run the exact same commands again.
Nothing you do gets thrown away.

Do `START_HERE.md` first if you have not.

---

## What the model does

It looks at a small square of ocean surface around one spot (9 cells by 9
cells), and predicts the temperature at 15 depths below that spot.

The square is the whole idea. A model that only looks at ONE cell cannot see
that there is a swirling eddy next door, and eddies push warm and cold water up
and down. Your main job is to PROVE that the square is what makes us better.

## First, train it once so you have a starting point

```
python scripts/03_train.py
```

Takes about 10 minutes on a normal laptop. Then:

```
python scripts/03b_baseline.py
```

```
python scripts/04_evaluate.py
```

Write down the "MEAN" line. That is your score to beat.

## IMPORTANT: save your results after every experiment

`04_evaluate.py` overwrites `outputs/scorecard.json` every time. So after each
experiment, copy it somewhere with a name:

```
cp outputs/scorecard.json outputs/result_patch9.json
```

Otherwise you will finish four experiments and only have the last one.

---

# EXPERIMENT 1 - the most important one

**Question: does looking at neighbouring cells actually help?**

Right now we compare our model against a RandomForest. But those two differ in
TWO ways at once: different model type, AND different input. So when we win, we
cannot honestly say which one caused it. A judge can attack that.

Fix: run the SAME model with a 1x1 square (only the centre cell) and compare to
the 9x9 square. Now only ONE thing changed, so the difference is proof.

### How to run it

Open `configs/config.yaml`, find:

```
patch:
  size: 9
```

Change `9` to `1`. Save. Then run all four steps, because the squares are built
into the data file and must be rebuilt:

```
python scripts/02_build_dataset.py
```

```
python scripts/03_train.py
```

```
python scripts/04_evaluate.py
```

```
cp outputs/scorecard.json outputs/result_patch1.json
```

Then set `size` back to `9` and rebuild:

```
python scripts/02_build_dataset.py
```

### What you are looking for

Compare `result_patch1.json` and `result_patch9.json`, especially the
`thermocline_75_300m` number. The 9x9 should be clearly better there. That is
the sentence for the slides:

"Same model, same data, same training. The only difference is whether it can see
the neighbouring ocean. Seeing it improves accuracy by X percent in the
thermocline."

---

# EXPERIMENT 2 - what size square is best?

Try 7 and 11 as well, the same way as above (edit config, rebuild dataset,
train, evaluate, save the result).

- Bigger square = more context, but slower and more blurry
- Smaller square = faster, but less context

Save each one as `result_patch7.json`, `result_patch11.json`. Whichever wins,
set that in the config and tell Person 3.

---

# EXPERIMENT 3 - just train longer (easy win)

The model was still improving when training stopped. Free accuracy:

```
python scripts/03_train.py --epochs 60
```

```
python scripts/04_evaluate.py
```

Watch the `val` number printed each epoch. When it stops dropping for about 10
epochs in a row, that is long enough. If it starts RISING, the model is
memorising instead of learning - go back to fewer epochs.

---

# EXPERIMENT 4 - try the other model type

There is a second model built in (a Vision Transformer) that has never actually
been tested. It might be better or worse - either answer is useful.

```
python scripts/03_train.py --encoder vit
```

```
python scripts/04_evaluate.py
```

It is slower. If it is not better after one honest try, say so and move on. "We
tried it, the simpler model won" is a perfectly good slide.

---

## Your deliverable

One table. Give it to Person 3:

| Experiment | mean RMSE | thermocline RMSE | mean correlation |
|---|---|---|---|
| patch 1 (centre only) | | | |
| patch 7 | | | |
| patch 9 (current) | 0.441 | 0.552 | 0.965 |
| patch 11 | | | |
| more epochs | | | |
| ViT | | | |
| RandomForest baseline | 0.619 | 1.170 | 0.936 |

Lower RMSE is better. Higher correlation is better.

Then commit the best settings:

```
git add -A && git commit -m "best model settings" && git push
```

---

## When the real data arrives

Put Person 1's `harmonized.nc` into `data/processed/`, then:

```
python scripts/02_build_dataset.py
```

```
python scripts/03_train.py
```

```
python scripts/04_evaluate.py
```

That is all. No code changes, because the real data has the same shape as the
practice data.

## If something goes wrong

**Training takes too long** - use `--epochs 5` while testing that a command
works, then run the full thing once.

**Out of memory** - in `configs/config.yaml`, lower
`max_samples_per_day` from 400 to 150, then rebuild the dataset.

**You changed patch size and got a shape error** - you forgot to re-run
`02_build_dataset.py`. The squares are baked into the data file.

**Using a computer with a GPU?** Training is currently locked to the CPU, so it
will not go faster on its own. Ask for the fix - it is a small change.
