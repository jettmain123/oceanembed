# PERSON 3 - RESULTS, DEMO AND SLIDES

**Your job:** turn the numbers into something judges can see and understand.

**You are not waiting for anybody.** Start now with the practice data.

Do `START_HERE.md` first if you have not.

---

## Step 1 - Make the results and pictures

```
python scripts/03_train.py
```

```
python scripts/03b_baseline.py
```

```
python scripts/04_evaluate.py
```

The last one creates three pictures in `outputs/`:

- `skill_vs_depth.png` - how accurate we are at each depth, us vs the simple
  baseline. **This is your most important picture.**
- `example_profiles.png` - six real examples of predicted vs true temperature
- `spatial_map.png` - a map showing where the model is right and wrong

Re-run `04_evaluate.py` every time Person 2 finishes an experiment.

## Step 2 - Run the demo

```
streamlit run scripts/05_demo.py
```

Opens in your browser. You pick a date and a place on the map, and it shows the
predicted temperature going down into the ocean next to the true answer, plus a
colour map of the whole region.

### Take screenshots NOW

Do not wait until presentation day. Live demos break at the worst moment.
Screenshot these:

1. The profile chart with prediction and truth sitting on top of each other
2. The colour map of the region
3. The scorecard at the bottom of the page

If the live demo fails in front of judges, you show the screenshots and keep
talking. Nobody will mind.

## Step 3 - The slides (5 of them)

### Slide 1 - The problem
Satellites see only the ocean SURFACE. Ships and floating instruments measure
the deep ocean, but there are very few of them, and huge areas of sea have no
measurements at all. India needs to know deep ocean temperature for cyclone
forecasting, fisheries, and climate.

### Slide 2 - Our idea
Learn the connection between what the surface looks like and what is underneath.
The key point: we do not just look at one spot. We look at a square of ocean
around it, because swirling eddies next door push warm and cold water up and
down. Show the diagram from `README.md`.

### Slide 3 - Data and method
Seven surface measurements (temperature, saltiness, sea height, currents both
directions, wind both directions). Trained on GLORYS. Tested on ARGO, which is
real floating instruments. Predicts 15 depths from 0 to 1000 m.

Say clearly: **we tested on dates the model never saw during training.** Judges
care about this.

### Slide 4 - Results
Use `skill_vs_depth.png` and Person 2's table.

Headline: about 65 percent more accurate than a standard approach, and 64 to 77
percent more accurate at every depth from 50 m to 500 m - including the
thermocline, the tricky middle layer where temperature changes fastest.

The strongest single slide is the field-of-view test in `EXPERIMENTS.md`: with
everything else held identical and only the size of the square changed, going
from 1 cell to 9x9 cells cuts the error by 48 percent. That proves the idea the
whole project rests on.

**Include the honest part. It makes you more credible, not less:**

- In the top 10 m the simple baseline still edges us (0.119 vs 0.164 degC). That
  is expected - near the surface the temperature is nearly the same as the
  surface temperature, so anything can predict it. The gap is tiny and we win
  everywhere from 20 m down.
- Below 500 m our accuracy drops, because the surface simply stops telling you
  what is happening that deep. No method can fix that. It is physics.

The line to say out loud: **"Our accuracy follows the physics. It is highest
where the surface controls the ocean, and it drops where it does not. We measure
that instead of hiding it."**

Judges have heard a hundred teams claim 99 percent accuracy. Being the team that
explains its own weaknesses is memorable.

### Slide 5 - Demo and what is next
Live demo (or screenshots). Then honestly: what we did not finish, and what we
would do with more time - more years of data, more ocean variables, extending to
saltiness as well as temperature.

## Step 4 - The written report

Cover: where the data came from, how much was missing, how we split it into
training and testing, the accuracy at every depth, and honest limitations.

Most of this is already written down for you:

- `DATA_SOURCES.md` - Person 1 fills this in
- `outputs/scorecard.json` - every number, per depth
- `README.md` - the design decisions and why

---

## You are done when

- The three pictures are made from the FINAL model
- Demo works, and screenshots exist as backup
- Five slides done
- Report written
- You have rehearsed the demo twice

## Things that will save you

**Rehearse twice.** Once alone, once in front of the team. Time it.

**Know the answer to "how do you know it is not just guessing?"**
Answer: we tested on dates the model never saw, and against ARGO, which is real
instruments in real water, not a computer simulation.

**Know the answer to "why does accuracy drop deeper?"**
Answer: because the surface stops constraining the deep ocean. That is physics,
not a bug. Any honest method shows the same shape.

**Freeze early.** Stop adding things about 3 hours before the deadline. Use that
time to practise. A rehearsed simple demo beats a broken clever one.

## If something goes wrong

**The demo says no model found** - run `python scripts/03_train.py` first.

**The demo opens on a blank chart** - move the location sliders; you have landed
on land or right at the map edge.

**A picture looks wrong or empty** - re-run `python scripts/04_evaluate.py`.
