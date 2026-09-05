# The website -- what to build and how

The final deliverable is a working web frontend. This is the spec.

## The one architectural decision that matters

**Do NOT run the model live behind the website.** Precompute every prediction
offline, export it, and serve a static site.

Why: a live inference server is the single most likely thing to fail in front of
judges. It needs Python, torch, the checkpoint, a running process and a network
path that all work simultaneously on demo day, on conference wifi. A static site
is a folder of files. It cannot crash, it loads instantly, and it deploys free
to Vercel, Netlify or GitHub Pages.

Nothing is lost. The model is deterministic and the domain is fixed, so every
answer the site can ever give is already known.

```
model -> export script -> static data files -> web page
         (run once)       (a few MB)           (no backend)
```

## The data format (BUILT -- `scripts/06_export_web.py`)

One binary per day, and one index. Simpler than the PNG-per-depth idea this file
originally specified: a single cube per day serves BOTH the map and the profile,
so there is one fetch instead of hundreds of images.

**`day_NNN.bin`** -- uint16 `[2, 15, 101, 241]`, meaning `[prediction,
reference]`. About 1.46 MB per day. The page fetches only the selected day, draws
the chosen depth slice to a canvas, and reads the 15 values under a click
straight out of the array already in memory.

uint16 with a scale/offset rather than float16, because `Uint16Array` exists in
every browser and `Float16Array` does not. 0 is the missing-data sentinel, so
NaN cells become transparent and the basemap shows through.

**`index.json`** -- dates, depths, grid corners, the encoding constants, and the
headline metrics lifted from `outputs/scorecard.json`.

```
python scripts/06_export_web.py              # full export, ~45 MB for 31 days
python scripts/06_export_web.py --index-only # refresh metrics only
```

## What the page must show

Ranked. Build top to bottom and stop when time runs out -- each item is useful
on its own.

1. **A map of the basin** at a chosen depth and date, with a depth slider
   (15 stops) and a date slider. This is the "wow" and it must be first.
2. **Click anywhere -> the temperature profile** at that point, predicted vs
   GLORYS reference, plotted 0-1000 m with depth increasing downwards.
3. **The accuracy panel** -- per-depth RMSE and correlation from the scorecard,
   and the comparison against the baseline.
4. **The honesty panel.** Skill above climatology by depth: strongly positive
   above 200 m, negative below 300 m. Say plainly that below ~300 m the surface
   stops constraining temperature and we do not claim skill there.
5. **A method diagram** -- surface fields -> 9x9 patch -> embedding -> 15 depths.
6. **Data sources** with product names and DOIs, straight from `DATA_SOURCES.md`.

Item 4 is the one that separates you from teams showing only their best number.
Do not drop it for polish.

## Design guidance

- **Dark background.** Ocean data is colourful; a dark ground makes it read.
- **Use a perceptually uniform colormap** for temperature (viridis, or cmocean's
  thermal). Avoid rainbow/jet -- it invents features that are not in the data,
  and an oceanographer on the panel will notice.
- **Depth axis always downwards.** Surface at the top. A profile plotted upside
  down reads as wrong to anyone in the field.
- **Always show units** (degC, metres) and the date of what is displayed.
- **Label predictions as predictions.** Never let a judge wonder whether a panel
  is model output or observation.
- One page, scrolled. Not a multi-tab app -- there is no time and no need.
- Must work on a projector: large fonts, high contrast, nothing that depends on
  hover (you will be presenting with a laptop, not a mouse someone can see).

## What NOT to build

No login, no user accounts, no database, no file upload, no "select your own
region", no live model, no 3D globe. Every one of those is a demo-day failure
mode in exchange for nothing a judge is scoring.

## Stack (BUILT -- `web/index.html`)

MapLibre GL for the map, and nothing else. The profile chart is hand-built SVG
and the temperature field is drawn to a canvas, so there is no chart library and
no framework to break.

**Basemap: Esri, not CARTO.** CARTO now stamps "API KEY REQUIRED" across every
tile. Esri needs no key:

- Dark: `Canvas/World_Dark_Gray_Base` + `World_Dark_Gray_Reference`
- Bathymetry: `Ocean/World_Ocean_Base` + `World_Ocean_Reference`

The bathymetry option is worth keeping. It shows the seafloor, which is exactly
where GLORYS runs out of deep values and the model therefore has no profile to
give -- the gaps in our map line up with the shelf, and that is easier to show
than to explain.

The field is a MapLibre `image` source updated in place, with land left
transparent so coastlines and place names stay legible underneath. Labels are
drawn ON TOP of the data layer, otherwise the basin becomes unreadable.

## Fallback, and when to build it

**Before adding any polish**, screenshot the working site and paste the images
into the slide deck. If the site breaks on demo day you present the screenshots
and nobody minds. `05_demo.py` (Streamlit) already works and is a second
fallback -- keep it functional, do not delete it.

## Time budget

| | |
|---|---|
| export script + data files | 2 h |
| map + depth/date sliders | 3 h |
| click-to-profile | 2 h |
| metrics + honesty panels | 2 h |
| method diagram, sources, styling | 2 h |
| deploy + screenshots | 1 h |

About 12 hours for one person. If there is less time, items 1-3 alone are a
credible demo.
