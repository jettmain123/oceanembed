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

## The data format to export

Two artefacts per day, both small:

**1. Map layers -- one PNG per (day, depth).**
Colour-map the 101 x 241 prediction field to a PNG. Roughly 10-20 KB each, so
31 days x 15 depths is a few MB. The browser can display these directly as an
image overlay; no client-side maths.

**2. Profile cube -- one binary file per day.**
101 x 241 x 15 values as float16 = about 0.73 MB per day. Fetch only the day the
user has selected. When they click a point, read the 15 values straight out of
the array already in memory. Instant, no server round trip.

Ship a small `index.json` listing available dates, depths, the lat/lon grid, and
the headline metrics from `outputs/scorecard.json`.

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

## Suggested stack

Plain HTML + a chart library is entirely sufficient and the safest choice.
If the person building it is fluent in React, Next.js static export is fine.
Do not learn a new framework this week.

- Charts: Chart.js or Plotly (both fine from a CDN)
- Map: the PNG layers over a simple lat/lon canvas. A full mapping library
  (Leaflet/Mapbox) is optional polish, not a requirement.

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
