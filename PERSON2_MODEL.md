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

Takes about 35 seconds on an NVIDIA GPU, or about 10 minutes on CPU. The script
prints which device it is using on the first few lines - check that. Then:

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

## READ THIS BEFORE YOU QUOTE ANY NUMBER

The practice data is FAKE. It was generated from a formula that a human wrote,
and then the model was trained to rediscover that formula. So of course it
scores well - the answer was planted there on purpose.

What good scores on fake data DO prove: the code runs, the shapes line up,
training converges, evaluation works, the demo displays. That is real and worth
having.

What they DO NOT prove: anything about the actual Indian Ocean.

**This affects Experiment 1 directly.** The fake ocean was deliberately built so
that deep temperature depends on the NEIGHBOURING cells. So when patch 9 beats
patch 1, part of that is just finding a signal that was planted. The experiment
is still worth running now - it proves your measurement method works and gets
your scripts ready - but the number that goes on the results slide must come
from the real GLORYS data, where nobody planted anything.

**Expect the real numbers to be worse.** Real oceans are messier than a formula.
That is normal and expected. Do not panic when the correlation drops.

Rule: run everything now on fake data, run it again on real data, report only
the real one.

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

**Training takes too long** - check the `device:` line it prints. If it says
`cpu` and you have an NVIDIA GPU, see the GPU note at the bottom; that is a
17x speed difference. Otherwise use `--epochs 5` while testing that a command
works, then run the full thing once.

**Out of memory** - in `configs/config.yaml`, lower
`max_samples_per_day` from 400 to 150, then rebuild the dataset.

**You changed patch size and got a shape error** - you forgot to re-run
`02_build_dataset.py`. The squares are baked into the data file.

**Using a computer with an NVIDIA GPU?** Training now uses it automatically -
but only if you installed the CUDA version of torch. Plain `pip install torch`
gives a CPU-only build on Windows, which cannot see your GPU at all no matter
what hardware you have.

Check which one you have:

```
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If it prints something ending in `+cpu`, or `False`, install the CUDA build:

```
pip install torch==2.14.0+cu130 --index-url https://download.pytorch.org/whl/cu130
```

That is about a 3 GB download. If it fails partway with a connection error,
just run it again - it picks up what it already downloaded.

`03_train.py` prints which device it chose on every run, so you never have to
guess. You can also force it with `--device cpu` or `--device cuda`.
