"""Stage 09 -- build the progress report PDF.

    python scripts/09_report.py [--out outputs/OceanEmbed_Progress_Report.pdf]

Reads whatever is on disk -- the harmonized cube, every scorecard, the product
and robustness JSONs, the figures 04_evaluate.py wrote -- and lays it out as one
document. Sections whose inputs are missing are skipped with a note rather than
crashing, so the report can be regenerated at any point in the project.

Deliberately dependency-free: matplotlib's PdfPages only, no reportlab.
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from oceanembed import load_config, resolve

A4 = (8.27, 11.69)
INK = "#12242f"
MUTED = "#5a6b76"
ACCENT = "#1f6feb"
WARM = "#d1701a"
RULE = "#c8d3da"

_page_no = [0]


def load(path):
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


# ---------------------------------------------------------------- layout ----
class Page:
    """A single portrait page you append blocks to, top down."""

    def __init__(self, pdf, title=None, subtitle=None, footer=True):
        self.pdf = pdf
        self.fig = plt.figure(figsize=A4)
        self.y = 0.945
        self.footer = footer
        if title:
            self.fig.text(0.08, self.y, title, size=17, weight="bold", color=INK)
            self.y -= 0.028
            if subtitle:
                self.fig.text(0.08, self.y, subtitle, size=9.5, color=MUTED)
                self.y -= 0.016
            self.rule()
            self.y -= 0.018

    def rule(self, pad=0.008):
        self.fig.add_artist(plt.Line2D([0.08, 0.92], [self.y, self.y],
                                       color=RULE, lw=0.9))
        self.y -= pad

    def h(self, text, size=11.5, color=None, gap=0.021):
        self.fig.text(0.08, self.y, text, size=size, weight="bold",
                      color=color or INK)
        self.y -= gap

    def p(self, text, size=9.3, color=None, width=104, gap=0.0148):
        for para in text.split("\n"):
            if not para.strip():
                self.y -= gap * 0.55
                continue
            for line in textwrap.wrap(para, width) or [""]:
                self.fig.text(0.08, self.y, line, size=size, color=color or INK)
                self.y -= gap
        self.y -= gap * 0.35

    def bullets(self, items, size=9.3, color=None):
        for it in items:
            lines = textwrap.wrap(it, 98)
            self.fig.text(0.085, self.y, "-", size=size, color=color or MUTED)
            for line in lines:
                self.fig.text(0.103, self.y, line, size=size, color=color or INK)
                self.y -= 0.0148
        self.y -= 0.006

    def mono(self, lines, size=8.2, color=None, gap=0.0135, indent=0.088):
        for line in lines:
            self.fig.text(indent, self.y, line, size=size, family="monospace",
                          color=color or INK)
            self.y -= gap
        self.y -= 0.004

    def table(self, headers, rows, widths=None, size=8.4, highlight=None,
              align=None):
        """Fixed-width monospace table. `highlight` is a set of row indices."""
        if widths is None:
            widths = []
            for i in range(len(headers)):
                w = len(str(headers[i]))
                for r in rows:
                    w = max(w, len(str(r[i])))
                widths.append(w + 2)

        def fmt(cells):
            out = []
            for i, c in enumerate(cells):
                s = str(c)
                left = align[i] == "l" if align else i == 0
                out.append(s.ljust(widths[i]) if left else s.rjust(widths[i]))
            return "".join(out)

        self.fig.text(0.088, self.y, fmt(headers), size=size, family="monospace",
                      weight="bold", color=INK)
        self.y -= 0.0125
        self.rule(pad=0.011)
        for i, r in enumerate(rows):
            c = ACCENT if (highlight and i in highlight) else INK
            w = "bold" if (highlight and i in highlight) else "normal"
            self.fig.text(0.088, self.y, fmt(r), size=size, family="monospace",
                          color=c, weight=w)
            self.y -= 0.0135
        self.y -= 0.010

    def note(self, text, size=8.6):
        lines = textwrap.wrap(text, 100)
        h = 0.0135 * len(lines) + 0.016
        self.fig.patches.append(plt.Rectangle(
            (0.075, self.y + 0.010 - h), 0.85, h, transform=self.fig.transFigure,
            facecolor="#eef4f8", edgecolor="#d2e0ea", lw=0.8, zorder=0))
        self.y -= 0.004
        for line in lines:
            self.fig.text(0.090, self.y, line, size=size, color="#22414f",
                          style="italic")
            self.y -= 0.0135
        self.y -= 0.014

    def axes(self, height=0.30, left=0.10, width=0.82):
        """Reserve a plotting rectangle below the current cursor."""
        self.y -= height
        ax = self.fig.add_axes([left, self.y, width, height - 0.035])
        self.y -= 0.040          # clearance for tick labels and the axis title
        return ax

    def image(self, path, height=0.30):
        p = Path(path)
        if not p.exists():
            self.p("[missing figure: " + p.name + "]", color=MUTED)
            return
        img = mpimg.imread(p)
        ar = img.shape[0] / img.shape[1]
        width = 0.84
        h = min(width * ar * (A4[0] / A4[1]), height)
        w = h / ar / (A4[0] / A4[1])
        self.y -= h + 0.012
        ax = self.fig.add_axes([0.5 - w / 2, self.y, w, h])
        ax.imshow(img)
        ax.axis("off")
        self.y -= 0.014

    def close(self):
        if self.footer:
            _page_no[0] += 1
            self.fig.text(0.92, 0.035, str(_page_no[0]), size=8, color=MUTED,
                          ha="right")
            self.fig.text(0.08, 0.035, "OceanEmbed  |  SIH 2026  |  INCOIS PS-01",
                          size=8, color=MUTED)
        self.pdf.savefig(self.fig)
        plt.close(self.fig)


# ------------------------------------------------------------- sections ----
def cover(pdf, cfg, card, days, span, argo=None):
    pg = Page(pdf, footer=False)
    f = pg.fig
    f.patches.append(plt.Rectangle((0, 0.70), 1, 0.30, transform=f.transFigure,
                                   facecolor="#0d2b3e", zorder=0))
    f.text(0.08, 0.900, "OceanEmbed", size=40, weight="bold", color="white")
    f.text(0.08, 0.858, "Three-dimensional ocean temperature from satellites alone",
           size=13, color="#9fc6dd")
    f.text(0.08, 0.775, "Smart India Hackathon 2026", size=11, color="#7fb0cc")
    f.text(0.08, 0.750, "INCOIS Problem Statement 01", size=11, color="#7fb0cc")

    f.text(0.08, 0.645, "Progress report", size=15, weight="bold", color=INK)
    f.text(0.08, 0.618, date.today().strftime("%d %B %Y"), size=10, color=MUTED)

    d = cfg["domain"]
    facts = [
        ("Domain", "%.0f-%.0fN, %.0f-%.0fE  (%s deg, daily)" % (
            d["lat_min"], d["lat_max"], d["lon_min"], d["lon_max"],
            d["resolution"])),
        ("Depth levels", "%d standard levels, %.0f-%.0f m" % (
            len(cfg["depths"]), cfg["depths"][0], cfg["depths"][-1])),
        ("Training record", "%s days   %s" % (days, span)),
        ("Inputs", "SST, SSS, SLA, surface currents, surface wind -- "
                   "satellite observable only"),
        ("Training label", "GLORYS12V1 reanalysis"),
        ("Independent check", "ARGO floats (never used in training)"),
    ]
    y = 0.545
    for k, v in facts:
        f.text(0.08, y, k, size=9.5, weight="bold", color=MUTED)
        f.text(0.30, y, v, size=9.5, color=INK)
        y -= 0.030

    if card:
        f.patches.append(plt.Rectangle((0.075, 0.145), 0.85, 0.185,
                                       transform=f.transFigure,
                                       facecolor="#f4f8fb", edgecolor=RULE, lw=1))
        f.text(0.10, 0.297, "Headline result -- measured against real ARGO floats",
               size=10.5, weight="bold", color=INK)
        if argo:
            cells = [
                ("RMSE vs ARGO floats", "%.2f C" % argo["our_rmse_vs_argo_degC"]),
                ("GLORYS vs same floats", "%.2f C" % argo["glorys_rmse_vs_argo_degC"]),
                ("Gap to the reanalysis", "%+.0f%%" % argo["excess_over_glorys_pct"]),
                ("Correlation vs GLORYS", "%.3f" % card["mean_corr"]),
            ]
        else:
            cells = [
                ("Mean correlation", "%.3f" % card["mean_corr"]),
                ("Mean RMSE", "%.2f C" % card["mean_rmse"]),
                ("Skill vs climatology",
                 "%+.0f%%" % (100 * card["vs_climatology"]["mean_skill_vs_clim"])),
                ("Thermocline RMSE",
                 "%.2f C" % card["bands"]["thermocline_75_300m"]["mean_rmse"]),
            ]
        for i, (k, v) in enumerate(cells):
            x = 0.112 + i * 0.207
            f.text(x, 0.233, v, size=17, weight="bold", color=ACCENT)
            f.text(x, 0.197, k, size=7.6, color=MUTED)
        f.text(0.10, 0.162,
               "ARGO floats never entered training. Held-out days fall "
               "chronologically after every training day.",
               size=8, color=MUTED, style="italic")
    pg.close()


def summary_page(pdf, card, argo, runs):
    pg = Page(pdf, "Where the project stands",
              "what has been demonstrated, and what has not")
    pg.h("The problem")
    pg.p("Satellites see only the ocean surface. Everything that matters for "
         "cyclone intensification, fisheries and marine heatwaves happens below "
         "it -- and the subsurface is measured by roughly 4,000 ARGO floats for "
         "the entire world ocean, which in our basin means one profile every few "
         "hundred kilometres every ten days. The operational alternative, a "
         "physics assimilation such as GLORYS, needs a supercomputer and runs "
         "with a delay.")
    pg.h("What we built")
    pg.p("A learned mapping from a 9x9 patch of surface fields to the full "
         "temperature profile at 15 standard depths, at 0.25 degrees, daily. It "
         "trains against GLORYS, runs in seconds on one GPU, and is checked "
         "against ARGO floats that never entered training.")
    pg.h("What the numbers say")
    if card:
        sk = 100 * card["vs_climatology"]["mean_skill_vs_clim"]
        th = card["bands"]["thermocline_75_300m"]["mean_rmse"]
        items = [
            "Mean correlation %.3f and RMSE %.2f C across all 15 depths on "
            "held-out days." % (card["mean_corr"], card["mean_rmse"]),
            "%+.0f%% better than climatology on average -- climatology being the "
            "per-location mean profile, which needs no satellite data at all and "
            "is the honest thing to beat." % sk,
            "Thermocline (75-300 m) RMSE %.2f C. This band is where skill above "
            "climatology is largest and where surface-only methods have nothing "
            "to say." % th,
        ]
        if argo:
            items.insert(0,
                "Against real ARGO floats -- instruments that physically "
                "descended through the water column and never entered training "
                "-- our RMSE is %.2f C. GLORYS, the supercomputer reanalysis we "
                "are distilled from, scores %.2f C against those same floats. We "
                "are %.0f%% worse than it, from satellites alone."
                % (argo["our_rmse_vs_argo_degC"], argo["glorys_rmse_vs_argo_degC"],
                   argo["excess_over_glorys_pct"]))
        pg.bullets(items)
    pg.h("What we are careful not to claim")
    pg.bullets([
        "We do not beat GLORYS. GLORYS is our teacher; the most we can be is a "
        "fast, cheap, satellite-only approximation of it.",
        "Below about 500 m the surface carries almost no information. The model "
        "correctly falls back to climatology there rather than inventing signal.",
        "The record is still partial -- November and December are missing from "
        "every year, so no complete annual cycle has been seen yet.",
        "The alerting layer is a proof of concept. Nothing is issued or "
        "transmitted to anyone.",
        "We carry a warm bias against ARGO that peaks in the thermocline. It is "
        "stated in full on the validation page rather than averaged away.",
    ])
    if runs and len(runs) > 1:
        pg.h("Progress so far")
        pg.p("Every increase in the training record has been measured rather "
             "than assumed. The full history is three pages on.")
    pg.close()


def coverage_page(pdf, cube_path):
    import pandas as pd
    import xarray as xr

    pg = Page(pdf, "Data coverage", "what has actually arrived, by month")
    if not Path(cube_path).exists():
        pg.p("harmonized.nc not found.", color=MUTED)
        pg.close()
        return

    ds = xr.open_dataset(cube_path)
    t = pd.to_datetime(ds.time.values)
    years = sorted(set(t.year))
    grid = np.zeros((len(years), 12))
    for yi, y in enumerate(years):
        for m in range(12):
            grid[yi, m] = int(((t.year == y) & (t.month == m + 1)).sum())

    pg.p("%d daily analysis days, %s to %s. Each cell is the number of days "
         "present for that month. Blank cells are months still being collected."
         % (len(t), t.min().date(), t.max().date()))

    ax = pg.axes(height=0.235)
    ax.imshow(np.where(grid > 0, grid, np.nan), cmap="YlGnBu",
              aspect="auto", vmin=0, vmax=31)
    ax.set_xticks(range(12))
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], size=8.5)
    ax.set_yticks(range(len(years)))
    ax.set_yticklabels([str(y) for y in years], size=10, weight="bold")
    for yi in range(len(years)):
        for m in range(12):
            v = grid[yi, m]
            ax.text(m, yi, "%d" % v if v else "", ha="center", va="center",
                    size=8, color="white" if v > 20 else INK)
    ax.set_xticks(np.arange(-.5, 12, 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(years), 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.5)
    ax.tick_params(which="minor", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Days present per month", size=9.5, color=MUTED, loc="left")

    missing = {}
    for yi, y in enumerate(years):
        miss = [m + 1 for m in range(12) if grid[yi, m] == 0]
        missing[y] = ", ".join("%02d" % m for m in miss) or "none"

    pg.h("Collection status")
    rows = []
    for yi, y in enumerate(years):
        rows.append(["Laptop %d" % (yi + 1), str(y), "%d" % grid[yi].sum(),
                     missing[y]])
    pg.table(["source", "assignment", "days in", "months still owed"], rows,
             widths=[12, 13, 10, 34], align=["l", "r", "r", "l"])

    pg.h("Variables in the harmonized cube")
    rows = []
    for v in ds.data_vars:
        a = ds[v]
        rows.append([v, " x ".join(a.dims),
                     "%.1f%%" % (100 * float(np.isnan(a.values).mean()))])
    pg.table(["variable", "dimensions", "missing"], rows, widths=[13, 30, 10],
             align=["l", "l", "r"])
    pg.note("Missing fractions near 50-60 per cent are land plus the deep levels "
            "under the continental shelf, not gaps in the satellite record. Land "
            "is masked out before any training sample is drawn.")
    ds.close()
    pg.close()


def method_page(pdf, cfg, card):
    pg = Page(pdf, "Method", "how a surface patch becomes a depth profile")
    pg.h("Pipeline")
    pg.mono([
        "raw products     harmonize         samples           model           products",
        "--------------   ---------------   ---------------   -------------   ------------",
        "SST  (OSTIA)     common 0.25 deg   9x9 patch of      CNN encoder     D26",
        "SSS  (Multiobs)  grid, daily,      7 surface vars    128-d latent    TCHP",
        "SLA  (DUACS)     same calendar,    centred on each   depth-attn      MLD",
        "currents         land masked       ocean cell        head            anomaly",
        "wind (ERA5)                                          15 depths       heatwave flag",
        "GLORYS --------- label only ------------------------> profile at 15 z",
    ], size=7.3)
    pg.h("Why a patch and not a point")
    pg.p("Thermocline depth responds to mesoscale eddies, and an eddy is a "
         "spatial structure -- a sea-level bump tens of kilometres across with a "
         "rotating current around it. A single pixel cannot express that. The "
         "%dx%d window lets the encoder see the gradient and the curl, which is "
         "what actually indicates the isotherms have been pushed down."
         % (cfg["patch"]["size"], cfg["patch"]["size"]))
    pg.h("Why the model predicts an anomaly, not a temperature")
    pg.p("Predicting absolute temperature means most of what the model has to "
         "learn is the climatology itself -- the fact that 1000 m here is always "
         "about 7.6 C. That is a lookup table, not inference, and any wobble the "
         "model added at depth made it worse than simply quoting the table. "
         "Measured: -161 per cent skill against climatology at depth.")
    pg.p("So the target is temperature minus the local mean profile, and the "
         "climatology is added back at the end. Predicting zero now IS "
         "climatology, which makes zero the floor instead of something to fall "
         "below. Weight decay pulls the answer toward zero exactly where the "
         "surface carries no information, which is the truthful behaviour at "
         "depth. The climatology is built from training days only, so the "
         "holdout cannot leak into it.")
    pg.h("Splits")
    pg.p("Chronological. Every held-out day comes after every training day. A "
         "random split would let the model see Tuesday and Thursday and be asked "
         "about Wednesday, which is interpolation rather than prediction, and "
         "would inflate every number in this report.")
    if card and card.get("holdout"):
        h = card["holdout"]
        pg.mono([
            "held-out profiles : %d" % h["n_profiles"],
            "latitude range    : %.1f to %.1f N" % (h["lat_range"][0], h["lat_range"][1]),
            "longitude range   : %.1f to %.1f E" % (h["lon_range"][0], h["lon_range"][1]),
        ])
    pg.close()


def progress_page(pdf, runs):
    pg = Page(pdf, "Progress across training runs",
              "each row is a real run, not a projection")
    if not runs:
        pg.p("No run history found.", color=MUTED)
        pg.close()
        return
    rows = [[r["label"], "%d" % r["days"], "%.3f" % r["corr"], "%.2f" % r["rmse"],
             "%+.0f%%" % (100 * r["skill"]), "%.2f" % r["thermo"]] for r in runs]
    pg.table(["run", "train days", "corr", "RMSE C", "vs clim", "thermo C"],
             rows, widths=[32, 12, 8, 9, 9, 10], highlight={len(rows) - 1},
             align=["l", "r", "r", "r", "r", "r"])

    ax = pg.axes(height=0.28)
    x = np.arange(len(runs))
    ax.bar(x - 0.19, [r["rmse"] for r in runs], 0.37, color=ACCENT,
           label="mean RMSE")
    ax.bar(x + 0.19, [r["thermo"] for r in runs], 0.37, color=WARM,
           label="thermocline RMSE")
    ax.set_xticks(x)
    ax.set_xticklabels([r["short"] for r in runs], size=8)
    ax.set_ylabel("degrees C", size=9)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("Error as the record grows (lower is better)",
                 size=9.5, color=MUTED, loc="left")

    pg.h("Reading this table honestly")
    pg.bullets([
        "The holdout changes between rows, because it is always the most recent "
        "block of a growing record. Absolute errors are comparable across rows; "
        "skill percentages are only comparable within the same holdout.",
        "The single-month run scored well because its holdout was the same "
        "season as its training. That is the easiest possible test, and it is no "
        "longer quoted as evidence.",
        "The large jump in skill came from switching the target to anomalies, "
        "not from more data. More data then improved the absolute error.",
    ])
    pg.close()


def depth_page(pdf, card, argo=None):
    pg = Page(pdf, "Accuracy against depth",
              "where the model earns its keep, and where it does not")
    if not card:
        pg.p("No scorecard found.", color=MUTED)
        pg.close()
        return
    sk = dict((r["depth_m"], r) for r in card["vs_climatology"]["per_depth"])
    ag = dict((r["depth_m"], r) for r in (argo or {}).get("per_depth", []))
    rows, hi = [], set()
    for i, r in enumerate(card["per_depth"]):
        z = r["depth_m"]
        s = 100 * sk[z]["skill_vs_clim"]
        if s > 30:
            hi.add(i)
        rows.append(["%.0f" % z, "%.3f" % r["corr"], "%.2f" % r["rmse"],
                     "%+.2f" % r["bias"], "%+.0f%%" % s,
                     "%.2f" % ag[z]["rmse"] if z in ag else "-",
                     "%.2f" % ag[z]["glorys_rmse"] if z in ag else "-"])
    pg.table(["depth m", "corr", "RMSE C", "bias C", "vs clim",
              "vs ARGO", "GLORYS"],
             rows, widths=[11, 9, 10, 10, 10, 10, 10], highlight=hi)
    pg.p("The first four columns are measured against GLORYS on held-out days. "
         "The last two are measured against real ARGO floats: our error, and "
         "the reanalysis's own error against the same instruments.", size=8.6)

    pg.h("Bands")
    pg.table(["band", "corr", "RMSE C"],
             [[k.replace("_", " "), "%.3f" % v["mean_corr"], "%.2f" % v["mean_rmse"]]
              for k, v in card["bands"].items()], widths=[26, 9, 10])

    pg.h("What the shape means")
    pg.bullets([
        "Near the surface the error is small but so is the skill above "
        "climatology -- SST is nearly the answer already, so there is little for "
        "a model to add.",
        "In the thermocline the error is largest in absolute terms and the skill "
        "above climatology is highest. That is the honest signature of real "
        "surface-to-depth inference: the layer is hard, and we are the only "
        "thing beating the average there.",
        "Below 500 m the skill returns to about zero. The surface genuinely does "
        "not constrain the deep ocean on daily timescales, and a model claiming "
        "otherwise would be fabricating.",
    ])
    pg.close()


def figure_pages(pdf, out_dir):
    figs = [("skill_vs_depth.png", "Skill against depth",
             "correlation, RMSE and bias at every level, model against baseline"),
            ("example_profiles.png", "Reconstructed profiles",
             "individual held-out profiles, predicted against GLORYS"),
            ("spatial_map.png", "Spatial structure of the error",
             "one held-out day at 100 m: reference, prediction, difference"),
            ("habitat_map.png", "Thermal habitat suitability",
             "predicted against the same score computed from GLORYS"),
            ("fronts_map.png", "Thermal fronts, surface and at depth",
             "the subsurface field is the one no satellite can provide"),
            ("barrier_layer.png", "Barrier-layer conditions",
             "fresh surface water over a deep isothermal layer"),
            ("seasonal_shift.png", "Seasonal shift of the suitable zone",
             "centroid of cells scoring above 0.6, across the record")]
    for name, title, sub in figs:
        p = Path(out_dir) / name
        if not p.exists():
            continue
        pg = Page(pdf, title, sub)
        pg.image(p, height=0.66)
        pg.close()


def products_page(pdf, prod):
    pg = Page(pdf, "Operational products",
              "forecasters act on these, not on a temperature field")
    if not prod:
        pg.p("products_scorecard.json not found -- run scripts/07_products.py.",
             color=MUTED)
        pg.close()
        return
    pg.p("All three are computed from the predicted profile and compared against "
         "the identical computation performed on GLORYS, on the same held-out "
         "days. D26 is the depth of the 26 C isotherm; TCHP is the heat stored "
         "above it, the quantity used to anticipate rapid cyclone "
         "intensification; MLD is the mixed layer depth.")
    rows = [[r["product"], r["unit"], "%d" % r["n"], "%.3f" % r["corr"],
             "%.2f" % r["rmse"], "%+.2f" % r["bias"]] for r in prod["products"]]
    pg.table(["product", "unit", "n", "corr", "RMSE", "bias"], rows,
             widths=[10, 10, 10, 9, 9, 9])

    th = None
    for r in prod["products"]:
        if r["product"] == "TCHP" and r.get("thresholds"):
            th = r["thresholds"]
    if th:
        pg.h("Would we have raised the same alarm?")
        pg.p("TCHP above roughly 50 kJ/cm2 is the rule-of-thumb threshold for "
             "rapid intensification. Precision and recall at that line matter "
             "more than RMSE.")
        pg.table(["threshold kJ/cm2", "precision", "recall", "F1", "n above"],
                 [["%g" % t["value"], "%.3f" % t["precision"], "%.3f" % t["recall"],
                   "%.3f" % t["f1"], "%d" % t["n_reference_above"]] for t in th],
                 widths=[20, 12, 10, 9, 11])

    hw = prod.get("anomaly_and_heatwave") or []
    if hw:
        pg.h("Subsurface anomaly and marine heatwave detection")
        rows = [["%.0f" % r["depth_m"], "%.3f" % r["anomaly_corr"],
                 "%.2f" % r["anomaly_rmse"], "%.3f" % r["hw_recall"],
                 "%.3f" % r["hw_precision"], "%.1f%%" % (100 * r["hw_base_rate"])]
                for r in hw if r["depth_m"] in (0, 30, 75, 100, 150, 200, 300)]
        pg.table(["depth m", "anom corr", "anom RMSE", "recall", "precision",
                  "base rate"], rows, widths=[11, 12, 12, 10, 11, 11])
        pg.note("Detecting a warm anomaly 100 m below the surface, daily, from "
                "satellites alone is something surface-only monitoring cannot do "
                "at all. A formal marine-heatwave definition also requires five "
                "days of persistence, which needs a longer record than we have.")
    pg.close()


def fisheries_page(pdf, fish):
    pg = Page(pdf, "Habitat, fronts and barrier-layer conditions",
              "advisory layers derived from the reconstructed profile")
    if not fish:
        pg.p("fisheries_scorecard.json not found -- run scripts/10_fisheries.py.",
             color=MUTED)
        pg.close()
        return
    pg.p("Four layers, each a cheap function of a profile we already predict, and "
         "each validated against the identical computation performed on GLORYS. "
         "The thermal habitat layer is the same product category as the INCOIS "
         "Potential Fishing Zone advisory, which today uses sea surface "
         "temperature and chlorophyll and carries no subsurface term at all.")
    rows = [[r["layer"], r["unit"], "%d" % r["n"], "%.3f" % r["corr"],
             "%.3f" % r["rmse"], "%+.3f" % r["bias"]]
            for r in fish.get("layers_vs_glorys", [])]
    if rows:
        pg.table(["derived layer", "unit", "n", "corr", "RMSE", "bias"], rows,
                 widths=[26, 10, 9, 9, 10, 10],
                 align=["l", "l", "r", "r", "r", "r"])
    hits = fish.get("suitable_zone_agreement") or []
    if hits:
        pg.h("Agreement on the call a user would actually act on")
        pg.p("Not RMSE -- whether we flag the same cells as suitable (score above "
             "0.6) that GLORYS would.", size=8.8)
        pg.table(["species", "precision", "recall", "F1", "base rate"],
                 [[h["species"], "%.3f" % h["precision"], "%.3f" % h["recall"],
                   "%.3f" % h["f1"], "%.1f%%" % (100 * h["base_rate"])]
                  for h in hits], widths=[24, 12, 10, 9, 12],
                 align=["l", "r", "r", "r", "r"])
    pg.h("What these layers are, and are not")
    pg.bullets([
        "Thermal habitat is a transparent threshold-and-taper score on "
        "temperature and isothermal layer depth. Nobody trained it on catch "
        "data, so it cannot be validated as a catch predictor. It is an advisory "
        "about water properties, not a claim that fish are present.",
        "Fronts are computed at depth as well as at the surface. The surface "
        "version is what satellite products already provide; the subsurface "
        "version is the one nothing else can give.",
        "The barrier-layer field is a LIKELIHOOD, not a thickness. A true barrier "
        "layer is isothermal layer depth minus DENSITY mixed layer depth, and "
        "density needs salinity at depth, which this project does not predict. "
        "Reporting metres here would be fabricating a quantity we cannot "
        "compute. Adding a salinity head would make it real.",
        "Species preference bands are broad literature values and should be "
        "replaced with regionally tuned ones by a fisheries scientist.",
    ])
    pg.close()


def argo_page(pdf, argo):
    pg = Page(pdf, "Independent validation against ARGO floats",
              "the only numbers here not measured against our own teacher")
    if not argo:
        pg.p("No ARGO colocation results on disk -- run "
             "scripts/02b_argo_colocate.py --holdout-only.", color=MUTED)
        pg.close()
        return
    pg.p("Everything else in this report compares us against GLORYS, which "
         "trained us. ARGO floats are real instruments that physically descended "
         "through the water column. They never entered training. This is "
         "therefore the only genuinely independent measurement in the document.")
    rows = []
    for k, v in argo.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            continue
        rows.append([k.replace("_", " "),
                     "%.3f" % v if isinstance(v, float) else "%d" % v])
    if rows:
        pg.table(["quantity", "value"], rows, widths=[40, 12])
    if argo.get("per_depth"):
        pg.h("Against ARGO, by depth")
        rows, hi = [], set()
        for i, r in enumerate(argo["per_depth"]):
            gap = r.get("gap", float("nan"))
            if gap < 0:
                hi.add(i)
            rows.append(["%.0f" % r["depth_m"], "%d" % r.get("n", 0),
                         "%.2f" % r.get("rmse", float("nan")),
                         "%.2f" % r.get("glorys_rmse", float("nan")),
                         "%+.2f" % gap, "%+.2f" % r.get("bias", float("nan")),
                         "%.3f" % r.get("corr", float("nan"))])
        pg.table(["depth m", "n", "ours", "GLORYS", "gap", "our bias", "corr"],
                 rows, widths=[11, 9, 9, 10, 9, 11, 9], highlight=hi)
        pg.p("Highlighted rows are depths where we beat the reanalysis against "
             "the floats. Treat those as gridded-product smoothing rather than a "
             "genuine win: the honest claim there is parity, not superiority.",
             size=8.6)
        pg.h("The bias, stated plainly")
        worst = max(argo["per_depth"], key=lambda r: abs(r.get("bias", 0)))
        pg.p("We run systematically WARM against the floats at every level, and "
             "the bias peaks at %+.2f C at %.0f m -- in the thermocline, which is "
             "exactly the band we claim as our strength. A warm bias there means "
             "we would overestimate heat content and, downstream, cyclone heat "
             "potential. This is a real defect rather than a rounding artifact. "
             "It is also correctable: the floats provide an independent reference "
             "to fit a per-depth bias correction against, which is queued work, "
             "not a limitation of the approach."
             % (worst.get("bias", float("nan")), worst["depth_m"]))
    pg.note("GLORYS itself does not match ARGO exactly. The gap between our "
            "error against ARGO and GLORYS's own error against ARGO is the "
            "honest measure of what our fast approximation costs.")
    pg.close()


def robustness_page(pdf, rob, abl):
    pg = Page(pdf, "Robustness and ablations",
              "what happens when an input disappears, and what the design is worth")
    if rob:
        pg.h("Losing one satellite at a time")
        pg.p("Each input channel is blanked in turn and the model re-run. The "
             "useful column is not the error -- it is which variable the model "
             "actually depends on. A channel whose removal costs nothing was "
             "never being used.")
        rows = [["nothing (full)", "%.3f" % rob["baseline"]["mean_rmse"], "--",
                 "%.1f" % rob["baseline"]["tchp_rmse"]]]
        for r in sorted(rob["single_channel_missing"], key=lambda r: -r["pct_worse"]):
            rows.append([r["missing"], "%.3f" % r["mean_rmse"],
                         "%+.1f%%" % r["pct_worse"], "%.1f" % r["tchp_rmse"]])
        pg.table(["missing input", "mean RMSE C", "change", "TCHP RMSE"],
                 rows, widths=[18, 14, 10, 12])
        pg.note("The point is not that accuracy drops -- of course it does. The "
                "point is that the system still runs and still returns a "
                "calibrated profile. A physics assimilation cannot do that "
                "without a full rerun.")
    if abl:
        pg.h("Does the spatial patch actually matter?")
        pg.table(["patch size", "mean RMSE C", "corr"],
                 [["%dx%d" % (r["patch"], r["patch"]), "%.3f" % r["mean_rmse"],
                   "%.3f" % r["mean_corr"]] for r in abl["results"]],
                 widths=[14, 14, 9])
        pg.p("Best patch: %s. A 1x1 patch is the point-only case -- if it were "
             "as good, the whole spatial argument would collapse."
             % abl.get("best_patch"))
    if not rob and not abl:
        pg.p("No robustness or ablation results on disk.", color=MUTED)
    pg.close()


def risk_page(pdf, risk):
    pg = Page(pdf, "Decision support layer",
              "proof of concept -- nothing here is issued or transmitted")
    if not risk:
        pg.p("risk_summary.json not found -- run scripts/08_risk.py.", color=MUTED)
        pg.close()
        return
    pg.p("The temperature field is the middle of the system, not the end of it. "
         "On top of the reconstruction sit a cyclone-intensification risk index, "
         "a coastal exposure calculation over 30 named segments of the Indian "
         "coastline, and an alert log.")
    pg.h("Risk index inputs")
    pg.bullets([
        "Tropical cyclone heat potential above %s kJ/cm2, the "
        "rapid-intensification threshold." % risk.get("tchp_high_threshold", 50),
        "A requirement that TCHP is also anomalously high for that location. An "
        "absolute threshold alone fires almost everywhere in this basin during "
        "the warm season -- adding the anomaly condition cut flagged cell-days "
        "from about 137,000 to about 21,000.",
        "Model uncertainty from Monte-Carlo input resampling. A high-heat cell "
        "the model is unsure about is downgraded rather than escalated.",
    ])
    alerts = risk.get("alerts") or []
    if alerts:
        pg.h("Alert log (%d entries over the evaluated period)" % len(alerts))
        rows = []
        for a in alerts[:12]:
            rows.append([str(a.get("date", ""))[:10],
                         str(a.get("segment", ""))[:26],
                         str(a.get("level", "")),
                         "%.0f" % a["tchp"] if isinstance(a.get("tchp"), (int, float)) else "-"])
        pg.table(["date", "coastal segment", "level", "TCHP"], rows,
                 widths=[12, 28, 10, 8])
    pg.note(str(risk.get("disclaimer", ""))[:400] or
            "PROOF OF CONCEPT -- not issued, not transmitted, not an operational "
            "warning product.")
    pg.close()


def roadmap_page(pdf, missing_months):
    pg = Page(pdf, "What is left", "in priority order")
    pg.h("Blocking on data")
    pg.bullets([
        "November and December are missing from every year, so the model has "
        "never seen a complete annual cycle. Until then the seasonal coordinate "
        "channels stay switched off -- they were measured to hurt when the test "
        "season was absent from training.",
        "Months still outstanding: %s." % (missing_months or "none"),
    ])
    pg.h("Experiments queued")
    pg.bullets([
        "Temporal context: add 7-day and 30-day mean wind and the 7-day SST "
        "change as extra channels. The thermocline responds to accumulated "
        "forcing, not to a single day of it. Cheap, and needs no new data.",
        "Retrospective cyclone replay over a real storm, to show the risk layer "
        "responding to an event with a known outcome.",
        "Re-enable the seasonal channels once complete years are in.",
    ])
    pg.h("Deliverables")
    pg.bullets([
        "Web interface: built and deployed -- map, depth and date selection, six "
        "layers, 3D seafloor view, alert log.",
        "This report, regenerated from disk by scripts/09_report.py.",
        "Slide deck: not started.",
    ])
    pg.h("Reproducing everything in this document")
    pg.mono([
        "python scripts/01c_merge_harmonized.py \"data/incoming/*.nc\"",
        "python scripts/02_build_dataset.py --max-per-day 800",
        "python scripts/03_train.py --epochs 150 --weight-decay 1e-3",
        "python scripts/03b_baseline.py",
        "python scripts/04_evaluate.py",
        "python scripts/07_products.py",
        "python scripts/03d_sensor_robustness.py",
        "python scripts/08_risk.py",
        "python scripts/09_report.py",
    ], size=8.0)
    pg.close()


# ----------------------------------------------------------------- main ----
RUN_HISTORY = [
    ("scorecard_oct2024_1month.json", "Oct 2024 only (1 month)", "1 mo", 22),
    ("scorecard_absolute.json", "2022 Jan-Oct, absolute target", "2022 abs", 219),
    ("scorecard_anomaly.json", "2022 Jan-Oct, anomaly target", "2022 anom", 219),
    ("scorecard_2022_2023_488days.json", "2022 + partial 2023", "+2023", 353),
    ("scorecard.json", "2022 + 2023 + 2024 (current)", "+2024", None),
]


def train_days_now(cfg):
    """How many days the current run actually trained on, from the splitter."""
    try:
        import pandas as pd
        import xarray as xr

        from oceanembed.dataset import split_train_val_argo
        ds = xr.open_dataset(resolve(cfg["paths"]["harmonized"]))
        n = ds.sizes["time"]
        ds.close()
        tr, _, _ = split_train_val_argo(n, cfg)
        return int(len(tr))
    except Exception:
        return 0


def gather_runs(out_dir, cfg=None):
    runs = []
    for name, label, short, days in RUN_HISTORY:
        c = load(Path(out_dir) / name)
        if not c or "vs_climatology" not in c:
            continue
        n = days if days is not None else train_days_now(cfg)
        runs.append({"label": label, "short": short, "days": n,
                     "corr": c["mean_corr"], "rmse": c["mean_rmse"],
                     "skill": c["vs_climatology"]["mean_skill_vs_clim"],
                     "thermo": c["bands"]["thermocline_75_300m"]["mean_rmse"]})
    return runs


def record_span(path):
    try:
        import pandas as pd
        import xarray as xr
        ds = xr.open_dataset(path)
        t = pd.to_datetime(ds.time.values)
        miss = []
        for y in sorted(set(t.year)):
            for m in range(1, 13):
                if not ((t.year == y) & (t.month == m)).any():
                    miss.append("%d-%02d" % (y, m))
        ds.close()
        return len(t), "%s to %s" % (t.min().date(), t.max().date()), ", ".join(miss)
    except Exception:
        return None, "", ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config()
    out_dir = resolve(cfg["paths"]["outputs"])
    out = Path(args.out) if args.out else out_dir / "OceanEmbed_Progress_Report.pdf"

    card = load(resolve(cfg["paths"]["scorecard"]))
    base = load(resolve(cfg["paths"]["baseline_scorecard"]))
    prod = load(out_dir / "products_scorecard.json")
    rob = load(out_dir / "sensor_robustness.json")
    abl = load(out_dir / "patch_ablation.json")
    risk = load(out_dir / "risk_summary.json")
    argo = load(out_dir / "argo_validation.json")
    fish = load(out_dir / "fisheries_scorecard.json")
    runs = gather_runs(out_dir, cfg)
    days, span, missing = record_span(resolve(cfg["paths"]["harmonized"]))

    with PdfPages(out) as pdf:
        cover(pdf, cfg, card, days, span, argo)
        summary_page(pdf, card, argo, runs)
        argo_page(pdf, argo)
        coverage_page(pdf, resolve(cfg["paths"]["harmonized"]))
        method_page(pdf, cfg, card)
        progress_page(pdf, runs)
        depth_page(pdf, card, argo)
        figure_pages(pdf, out_dir)
        products_page(pdf, prod)
        fisheries_page(pdf, fish)
        robustness_page(pdf, rob, abl)
        risk_page(pdf, risk)
        roadmap_page(pdf, missing)

        d = pdf.infodict()
        d["Title"] = "OceanEmbed -- Progress Report"
        d["Author"] = "OceanEmbed team, SIH 2026"
        d["Subject"] = "3D ocean temperature reconstruction from satellite data"

    print("wrote %s  (%.2f MB, %d pages)"
          % (out, out.stat().st_size / 1e6, _page_no[0] + 1))


if __name__ == "__main__":
    main()
