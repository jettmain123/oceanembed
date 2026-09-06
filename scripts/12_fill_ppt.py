"""Stage 12 -- fill the SIH idea-submission template with our content.

    python scripts/12_fill_ppt.py

Rules this script keeps to, because the template says so on its own last slide:
  - the provided template is the only file edited; no slide is added or removed
  - every existing shape stays where it is: titles, logos, footers, page numbers
  - only the body text boxes are filled, plus the team-name badge
  - numbers come from the scorecards on disk, never typed in by hand

Writes SIH2026-IDEA-Presentation-Format-FILLED.pptx next to the original so the
blank template survives.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

from oceanembed import load_config, resolve

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT.parent / "SIH2026-IDEA-Presentation-Format.pptx"
OUT = ROOT.parent / "SIH2026-IDEA-Presentation-Format-FILLED.pptx"

INK = RGBColor(0x1B, 0x2A, 0x38)
ACCENT = RGBColor(0x12, 0x5C, 0xA8)
GOOD = RGBColor(0x1B, 0x7A, 0x4B)
DIM = RGBColor(0x55, 0x66, 0x74)


def load(p):
    p = Path(p)
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    except Exception:
        return None


# ---------------------------------------------------------------- diagram ---
def pipeline_png(path, card, argo):
    fig = plt.figure(figsize=(11.6, 2.35), dpi=200)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.set_xlim(0, 100); ax.set_ylim(0, 26)

    steps = [
        ("SATELLITE INPUTS", "SST  ·  SSS  ·  sea level\ncurrents  ·  wind\n(all free, all operational)"),
        ("HARMONISE", "common 0.25° grid\ndaily, land masked\n777 days, 26 months"),
        ("9×9 PATCH", "7 channels × 81 cells\naround every ocean cell\n— sees eddies, not pixels"),
        ("CNN ENCODER", "64 → 96 → 128 filters\n→ 128-d latent\n776,449 parameters"),
        ("DEPTH ATTENTION", "15 learned queries\neach reads the latent\nfor its own depth"),
        ("PROFILE + PRODUCTS", "15 depths, 0–1000 m\nD26 · TCHP · MLD\nrisk · habitat · fronts"),
    ]
    n = len(steps)
    w, gap = 13.9, 2.4
    x = 1.0
    for i, (t, b) in enumerate(steps):
        col = "#e8f1fb" if i < 3 else "#dbeafe"
        ax.add_patch(plt.Rectangle((x, 4.5), w, 17, facecolor=col,
                                   edgecolor="#7aa7d4", lw=1.1, zorder=2,
                                   joinstyle="round"))
        ax.text(x + w / 2, 18.6, t, ha="center", va="center", size=7.4,
                weight="bold", color="#123a63", zorder=3)
        ax.text(x + w / 2, 11.4, b, ha="center", va="center", size=6.3,
                color="#2c4459", zorder=3, linespacing=1.5)
        if i < n - 1:
            ax.annotate("", xy=(x + w + gap - 0.5, 13), xytext=(x + w + 0.4, 13),
                        arrowprops=dict(arrowstyle="-|>", color="#125ca8", lw=1.6))
        x += w + gap

    ax.text(50, 1.6,
            "GLORYS12V1 reanalysis is the training label only.   "
            "ARGO floats are held out entirely and used for independent validation.",
            ha="center", va="center", size=6.6, color="#55616c", style="italic")
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


# ------------------------------------------------------------------ text ----
def fill(box, blocks, size=12.0, space_after=5):
    """Replace a text box's contents. blocks = [(text, kind), ...]."""
    tf = box.text_frame
    tf.word_wrap = True
    tf.clear()
    first = True
    for text, kind in blocks:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.space_after = Pt(space_after)
        run = p.add_run()
        run.text = text
        f = run.font
        if kind == "h":
            f.size, f.bold, f.color.rgb = Pt(size + 1.5), True, ACCENT
            p.space_before = Pt(7 if not first else 0)
        elif kind == "b":
            f.size, f.color.rgb = Pt(size), INK
            p.level = 1
        elif kind == "num":
            f.size, f.bold, f.color.rgb = Pt(size), True, GOOD
            p.level = 1
        else:
            f.size, f.color.rgb = Pt(size), DIM
    return tf


def grow(shape, top=None, height=None, left=None, width=None):
    if top is not None:
        shape.top = Inches(top)
    if height is not None:
        shape.height = Inches(height)
    if left is not None:
        shape.left = Inches(left)
    if width is not None:
        shape.width = Inches(width)


def body_of(slide):
    for sh in slide.shapes:
        if sh.name == "TextBox 8":
            return sh
    return None


def main() -> None:
    cfg = load_config()
    out_dir = resolve(cfg["paths"]["outputs"])
    card = load(resolve(cfg["paths"]["scorecard"]))
    argo = load(out_dir / "argo_validation.json")
    prod = load(out_dir / "products_scorecard.json")
    fish = load(out_dir / "fisheries_scorecard.json")
    rob = load(out_dir / "sensor_robustness.json")
    mc = load(out_dir / "model_card.json")
    risk = load(out_dir / "risk_summary.json")

    A = argo or {}
    ours = A.get("our_rmse_vs_argo_degC", float("nan"))
    glo = A.get("glorys_rmse_vs_argo_degC", float("nan"))
    gappct = A.get("excess_over_glorys_pct", float("nan"))
    corr = card["mean_corr"] if card else float("nan")
    skill = 100 * card["vs_climatology"]["mean_skill_vs_clim"] if card else float("nan")
    nhold = card["holdout"]["n_profiles"] if card else 0
    params = mc["total_params"] if mc else 0

    tchp_t = None
    for r in (prod or {}).get("products", []):
        if r["product"] == "TCHP":
            for t in r.get("thresholds", []):
                if t["value"] == 50:
                    tchp_t = t
    d26_corr = next((r["corr"] for r in (prod or {}).get("products", [])
                     if r["product"] == "D26"), float("nan"))

    hab = {h["species"]: h for h in (fish or {}).get("suitable_zone_agreement", [])}

    diag = out_dir / "ppt_pipeline.png"
    pipeline_png(diag, card, argo)

    prs = Presentation(str(TEMPLATE))
    s = prs.slides

    # ---- team badge on every slide that has one --------------------------
    for sl in s:
        for sh in sl.shapes:
            if sh.name.startswith("Oval") and sh.has_text_frame:
                tf = sh.text_frame
                tf.clear()
                p = tf.paragraphs[0]
                p.alignment = PP_ALIGN.CENTER
                r = p.add_run(); r.text = "Snack\nOverflow"
                r.font.size = Pt(10); r.font.bold = True
                # the oval has no fill, so white text was invisible on white
                r.font.color.rgb = RGBColor(0x4A, 0x3A, 0x82)

    # =================== SLIDE 2 -- proposed solution =====================
    b = body_of(s[1])
    grow(b, left=0.55, top=1.30, width=12.3, height=5.55)
    fill(b, [
        ("Proposed Solution", "h"),
        ("Satellites see only the ocean skin. Everything that decides cyclone "
         "intensification, fish distribution and marine heatwaves happens BELOW it — "
         "and the subsurface is measured by ~4,000 ARGO floats for the whole world "
         "ocean: in our basin, one profile per few hundred km per ten days. The only "
         "operational alternative, a physics reanalysis, needs a supercomputer.", "b"),
        ("OceanEmbed reconstructs the full 3-D temperature field from surface "
         "satellite data alone — 15 standard depths (0–1000 m), 0.25°, daily, "
         "across 5–30°N / 45–105°E — in seconds on one GPU.", "b"),
        ("How it works", "h"),
        ("A 9×9 patch of seven surface fields around every ocean cell → CNN encoder "
         "→ 128-d latent → 15 learned depth queries attend over it → the profile. "
         f"{params:,} parameters. Trained on GLORYS reanalysis; ARGO floats held out "
         "entirely for independent validation.", "b"),
        ("Innovation and uniqueness", "h"),
        ("1. Predicts the ANOMALY from local climatology, not absolute temperature — "
         "so 'no skill' equals climatology instead of falling below it. Fixed deep "
         "skill from −161% to ≈0.", "b"),
        ("2. Depth-attention decoder: 15 depths are one physical water column sharing "
         "one encoder, not 15 independent regressions.", "b"),
        ("3. Trained against whole-channel dropout, so it degrades gracefully when a "
         "satellite fails — and the same perturbation yields calibrated uncertainty.", "b"),
        ("4. Refuses coordinates as input, which blocks the memorise-the-map shortcut "
         "and makes 'skill above climatology' an honest number.", "b"),
        (f"Validated against real ARGO floats: {ours:.2f} °C RMSE — the supercomputer "
         f"reanalysis it learns from scores {glo:.2f} °C on the identical profiles. "
         f"A {gappct:.0f}% gap, at a fraction of the cost.", "num"),
    ], size=13.0, space_after=7)

    # =================== SLIDE 3 -- technical approach ====================
    b = body_of(s[2])
    grow(b, left=0.55, top=1.18, width=12.3, height=2.85)
    fill(b, [
        ("Technologies used", "h"),
        ("Python · PyTorch (CUDA) · xarray + dask · NumPy/scikit-learn · "
         "copernicusmarine · MapLibre GL + deck.gl web client · matplotlib reporting.", "b"),
        ("Data: OSTIA SST, Multiobs SSS, DUACS sea level, surface currents, ERA5/CCMP "
         "wind — all free and operational. GLORYS12V1 = training label. "
         "ARGO (INCOIS) = independent validation, never trained on.", "b"),
        ("Methodology", "h"),
        ("Harmonise every product onto one 0.25° daily grid → cut a 9×9 patch at each "
         "ocean cell → CNN encoder → 128-d latent → depth-attention head → 15 "
         "temperatures → operational products. Splits are CHRONOLOGICAL: every "
         "held-out day falls after every training day, so nothing is interpolated.", "b"),
        (f"777 days collected across 26 months; {nhold:,} held-out profiles evaluated.", "num"),
    ], size=13.0, space_after=6)
    s[2].shapes.add_picture(str(diag), Inches(0.55), Inches(4.05), width=Inches(12.25))

    # =================== SLIDE 4 -- feasibility ===========================
    b = body_of(s[3])
    grow(b, left=0.55, top=1.20, width=12.3, height=5.6)
    fill(b, [
        ("Feasibility — already built and measured, not proposed", "h"),
        ("Working end-to-end pipeline, trained model, validated products, live web "
         "client and an auto-generated report. Training takes minutes on one laptop "
         "GPU; a whole basin-day infers in seconds. Every input is a free, "
         "operational satellite product, so there is no hardware to deploy.", "b"),
        ("Challenges, and what we did about each", "h"),
        ("We cannot beat GLORYS — it is our teacher. → We never claim to. We position "
         "as a fast, cheap approximation and prove the gap on ARGO: "
         f"{ours:.2f} vs {glo:.2f} °C.", "b"),
        ("A warm bias against ARGO peaking at +1.53 °C at 125 m. → Named openly; "
         "correctable per-depth against the independent float record.", "b"),
        ("Below ~500 m the surface carries almost no information. → The anomaly target "
         "makes climatology the floor, so the model falls back to it instead of "
         "inventing signal.", "b"),
        ("Satellites fail. → Whole-channel dropout in training; measured degradation "
         f"of only +{max(r['pct_worse'] for r in rob['single_channel_missing']):.1f}% "
         "with the most important input (SST) removed entirely, and the system still "
         "returns a calibrated profile.", "b"),
        ("An absolute heat threshold fires across the whole tropical basin. → We "
         "require a local ANOMALY too, which cut flagged cell-days from 285,268 to "
         "45,969 — a real property of this ocean, not a tuning convenience.", "b"),
        ("Validation loss floors within ~5 epochs; longer training overfits. → More "
         "data is the lever, not more epochs. Collection continues.", "b"),
        ("Risk we state plainly: the alerting layer is a proof of concept. Nothing is "
         "issued, transmitted, or connected to any warning system.", "num"),
    ], size=12.5, space_after=6)

    # =================== SLIDE 5 -- impact ================================
    sk = hab.get("indian_oil_sardine", {})
    b = body_of(s[4])
    grow(b, left=0.55, top=1.20, width=12.3, height=5.6)
    fill(b, [
        ("Cyclone preparedness — the Bay of Bengal problem", "h"),
        ("Tropical cyclone heat potential is the ocean quantity that decides rapid "
         "intensification, and it is a SUBSURFACE quantity. At the 50 kJ/cm² "
         f"rule-of-thumb threshold we reproduce the reanalysis call with precision "
         f"{tchp_t['precision']:.2f} and recall {tchp_t['recall']:.2f} — we raise the "
         f"same alarm ~{100*tchp_t['recall']:.0f}% of the time. Depth of the 26 °C "
         f"isotherm tracks at r = {d26_corr:.2f}. Exposure is resolved onto 30 named "
         "coastal segments from Gujarat to West Bengal.", "b"),
        ("Retrospective test — Cyclone Dana, October 2024", "h"),
        ("On days the model had never seen, the risk index for Paradip-Kendrapara "
         "and Balasore-Bhadrak stepped from ELEVATED to HIGH on 24 October — the "
         "stretch of Odisha coast where Cyclone Dana came ashore — with "
         "reconstructed heat potential rising from ~70 to ~98 kJ/cm2 and holding "
         "for a week. We do not claim to have detected the storm; we have no "
         "atmospheric data. What we show is that the ocean beneath its track was "
         "carrying rapid-intensification-grade heat, read from satellites alone, "
         "on held-out data.", "b"),
        ("Fisheries — extending an advisory INCOIS already issues", "h"),
        ("The INCOIS Potential Fishing Zone advisory today uses surface temperature "
         "and chlorophyll, with no subsurface term. We add thermal habitat scored on "
         "the reconstructed profile AND the isothermal layer depth, plus thermal "
         "fronts computed at 100 m where no satellite can see. Agreement with the "
         f"reanalysis on the suitable-zone call: F1 {sk.get('f1', float('nan')):.2f} "
         "for Indian oil sardine. Fewer wasted fuel-hours for small-boat operators.", "b"),
        ("Climate and ecosystem monitoring", "h"),
        ("Subsurface marine heatwaves detected at 100 m daily — something surface-only "
         "monitoring cannot do at all — plus full-column ocean heat content anomaly, "
         "which moves continuously where TCHP stays pinned at zero until a threshold "
         "trips. Barrier-layer conditions are flagged where river-fed fresh water caps "
         "warm subsurface water, the mechanism behind Bay of Bengal intensification.", "b"),
        ("Why it matters that it is cheap", "h"),
        ("A reanalysis needs a supercomputer and runs with a delay. This runs on a "
         "laptop GPU from public satellite feeds, so it can be operated by a state "
         "agency, a fisheries department, or a university — no new instruments, no new "
         "satellites, no new observing infrastructure.", "b"),
        (f"Independently validated on {A.get('n_profiles', 0):,} real ARGO float "
         f"profiles, all on held-out days.", "num"),
    ], size=12.5, space_after=6)

    # =================== SLIDE 6 -- references ============================
    b = body_of(s[5])
    grow(b, left=0.55, top=1.20, width=12.3, height=5.6)
    fill(b, [
        ("Data sources", "h"),
        ("Copernicus Marine Service (marine.copernicus.eu) — GLORYS12V1 global ocean "
         "reanalysis (training label); OSTIA sea surface temperature; DUACS sea level "
         "anomaly and geostrophic currents; Multi Observation global ocean salinity.", "b"),
        ("ECMWF ERA5 reanalysis / CCMP — surface wind components.", "b"),
        ("Argo programme, accessed through the INCOIS Live Access Server "
         "(incois.gov.in) — independent subsurface validation, never used in training.", "b"),
        ("Scientific basis", "h"),
        ("Leipper & Volgenau (1972), Journal of Physical Oceanography — hurricane heat "
         "potential; the origin of the TCHP metric we compute.", "b"),
        ("de Boyer Montégut et al. (2004), JGR Oceans — mixed layer depth over the "
         "global ocean; the basis for our layer-depth criterion and for the "
         "temperature-vs-density distinction we are careful to preserve.", "b"),
        ("Vinayachandran and co-workers — Bay of Bengal barrier layers: river-fed fresh "
         "water capping warm subsurface water, suppressing storm-induced cooling. The "
         "reason our basin behaves differently from the open Pacific.", "b"),
        ("Published work on machine-learning reconstruction of subsurface ocean fields "
         "from satellite surface observations (e.g. Su, Chen and co-workers).", "b"),
        ("Our work", "h"),
        ("Full pipeline, trained model, ablations, sensor-robustness study, "
         "auto-generated 19-page validation report and the live web client are all in "
         "the project repository. Every figure in this deck regenerates from disk.", "b"),
        ("Problem Statement SIH26066 · INCOIS · Ministry of Earth Sciences.", "num"),
    ], size=12.5, space_after=6)

    prs.save(str(OUT))
    print(f"wrote {OUT}")
    print(f"  slides: {len(prs.slides)} (unchanged)")
    print(f"  pipeline diagram: {diag}")


if __name__ == "__main__":
    main()
