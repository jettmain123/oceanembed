"""Stage 05 -- Streamlit proof of concept.

    streamlit run scripts/05_demo.py

Pick a date and a point: the app reconstructs the temperature profile from
surface data alone and plots it against the reference profile, and draws a basin
heatmap at any of the 15 standard depths.

Degrades gracefully at every step -- if the checkpoint, the cube or torch is
missing it says exactly what to run instead of crashing.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import streamlit as st

from oceanembed import load_config, resolve
from oceanembed.evaluate import format_table, load_card

st.set_page_config(page_title="OceanEmbed", layout="wide", initial_sidebar_state="expanded")


# --------------------------------------------------------------------------
# loading -- everything cached and guarded
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_cfg():
    return load_config()


@st.cache_resource(show_spinner="loading harmonized cube...")
def get_cube(path: str):
    import xarray as xr

    return xr.open_dataset(path)


@st.cache_resource(show_spinner="loading model...")
def get_predictor():
    from oceanembed.inference import Predictor

    return Predictor.load()


def patch_at(ds, cfg, it: int, iy: int, ix: int):
    """Cut the model input patch for one cell on one day. Returns None if invalid."""
    P = int(cfg["patch"]["size"])
    half = P // 2
    svars = list(cfg["surface_vars"])
    ny, nx = ds.sizes["lat"], ds.sizes["lon"]
    if not (half <= iy < ny - half and half <= ix < nx - half):
        return None
    sl = (slice(iy - half, iy + half + 1), slice(ix - half, ix + half + 1))
    arr = np.stack([np.asarray(ds[v].isel(time=it).values, np.float32)[sl] for v in svars])
    if not np.isfinite(arr).all():
        return None
    if cfg["patch"].get("add_coords", True):
        dom = cfg["domain"]
        lat = float(ds["lat"].values[iy])
        lon = float(ds["lon"].values[ix])
        t = np.datetime64(ds["time"].values[it], "D").astype(object)
        doy = t.timetuple().tm_yday
        extra = np.empty((4, P, P), np.float32)
        extra[0] = (lat - dom["lat_min"]) / (dom["lat_max"] - dom["lat_min"]) * 2 - 1
        extra[1] = (lon - dom["lon_min"]) / (dom["lon_max"] - dom["lon_min"]) * 2 - 1
        extra[2] = np.sin(2 * np.pi * doy / 365.25)
        extra[3] = np.cos(2 * np.pi * doy / 365.25)
        arr = np.concatenate([arr, extra])
    return arr[None]


@st.cache_data(show_spinner=False)
def default_point(land: np.ndarray, lat_v, lon_v):
    """Open on the point furthest from land.

    The domain centre is central India, and a coastal cell has a patch that
    touches land -- either way the demo would open on an empty panel.
    """
    try:
        from scipy.ndimage import distance_transform_edt

        # pad with land so the domain edge counts as a boundary too -- otherwise
        # the furthest-from-land cell is on the border and its patch runs off grid
        padded = np.pad(~land, 1, mode="constant", constant_values=False)
        dist = distance_transform_edt(padded)[1:-1, 1:-1]
    except Exception:
        dist = (~land).astype(float)
        dist[0, :] = dist[-1, :] = dist[:, 0] = dist[:, -1] = 0.0
    iy, ix = np.unravel_index(int(np.argmax(dist)), dist.shape)
    return float(lat_v[iy]), float(lon_v[ix])


# --------------------------------------------------------------------------
st.title("OceanEmbed -- subsurface temperature from surface satellites")
st.caption(
    "Reconstructs temperature at 15 standard depths (0-1000 m) over the North Indian Ocean "
    "using only surface fields: SST, SSS, sea level anomaly, currents and winds."
)

cfg = get_cfg()
hpath = resolve(cfg["paths"]["harmonized"])

if not hpath.exists():
    st.error(f"No harmonized cube at `{hpath}`.")
    st.code("python scripts/00_make_synthetic.py\npython scripts/01_harmonize.py", language="bash")
    st.stop()

ds = get_cube(str(hpath))

try:
    pred_ = get_predictor()
    model_ready = True
except Exception as exc:
    pred_ = None
    model_ready = False
    st.warning(f"No usable model yet ({type(exc).__name__}: {exc}). Showing reference data only.")
    st.code("python scripts/03_train.py", language="bash")

lat = np.asarray(ds["lat"].values, np.float32)
lon = np.asarray(ds["lon"].values, np.float32)
depths = np.asarray(ds["depth"].values, np.float32)
times = np.asarray(ds["time"].values)
land = np.asarray(ds["land_mask"].values) > 0.5

with st.sidebar:
    st.header("Select")
    di = st.select_slider(
        "date", options=list(range(times.size)),
        format_func=lambda i: str(np.datetime64(times[i], "D")),
        value=times.size - 1,
    )
    dlat, dlon = default_point(land, lat, lon)
    plat = st.slider("latitude (N)", float(lat[0]), float(lat[-1]), dlat, 0.25)
    plon = st.slider("longitude (E)", float(lon[0]), float(lon[-1]), dlon, 0.25)
    zsel = st.selectbox("map depth (m)", [int(z) for z in depths], index=7)
    st.divider()
    if model_ready:
        st.success(f"backend `{pred_.backend}` / encoder `{pred_.meta.get('encoder')}`")
    st.caption(f"cube: {times.size} days, {lat.size} x {lon.size} cells, {depths.size} depths")

iy = int(np.argmin(np.abs(lat - plat)))
ix = int(np.argmin(np.abs(lon - plon)))

left, right = st.columns([1, 1.15])

# --------------------------------------------------------------------------
with left:
    st.subheader(f"Profile at {lat[iy]:.2f} N, {lon[ix]:.2f} E")
    if land[iy, ix]:
        st.info("That point is on land. Move the sliders over the ocean.")
    else:
        ref = np.asarray(ds["temp"].isel(time=di, lat=iy, lon=ix).values, np.float32)
        x = patch_at(ds, cfg, di, iy, ix)
        pred = pred_.predict(x)[0] if (model_ready and x is not None) else None

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(4.6, 5.6))
        ax.plot(ref, depths, "k-o", ms=4, label="reference")
        if pred is not None:
            ax.plot(pred, depths, "-s", ms=4, color="#1f6feb", label="OceanEmbed")
        ax.invert_yaxis()
        ax.set_yscale("symlog", linthresh=50)
        ax.set_ylim(1100, -1)
        ax.set_xlabel("temperature (degC)")
        ax.set_ylabel("depth (m)")
        ax.grid(alpha=0.3)
        ax.legend()
        st.pyplot(fig, use_container_width=True)

        if pred is not None:
            rmse = float(np.sqrt(np.mean((pred - ref) ** 2)))
            c1, c2, c3 = st.columns(3)
            c1.metric("profile RMSE (degC)", f"{rmse:.2f}")
            c2.metric("surface error (degC)", f"{pred[0] - ref[0]:+.2f}")
            k = int(np.argmin(np.abs(depths - 100)))
            c3.metric("error at 100 m (degC)", f"{pred[k] - ref[k]:+.2f}")
        elif x is None:
            st.info("The patch around this point touches land or missing data.")

# --------------------------------------------------------------------------
with right:
    st.subheader(f"Basin temperature at {zsel} m -- {np.datetime64(times[di], 'D')}")
    k = int(np.argmin(np.abs(depths - zsel)))
    field = np.asarray(ds["temp"].isel(time=di, depth=k).values, np.float32)
    field = np.where(land, np.nan, field)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    im = ax.pcolormesh(lon, lat, field, cmap="turbo", shading="auto")
    ax.plot([lon[ix]], [lat[iy]], "w*", ms=16, mec="k", mew=1.0)
    ax.set_xlabel("longitude (E)")
    ax.set_ylabel("latitude (N)")
    fig.colorbar(im, ax=ax, label="degC")
    st.pyplot(fig, use_container_width=True)

    st.caption(
        "Reference field from the harmonized cube. The star marks the selected point. "
        "Swap in the real GLORYS/ARGO cube and this panel is unchanged -- same schema."
    )

# --------------------------------------------------------------------------
st.divider()
card = load_card(resolve(cfg["paths"]["scorecard"]))
base = load_card(resolve(cfg["paths"]["baseline_scorecard"]))
if card:
    st.subheader("Validation on the independent holdout")
    a, b, c, d = st.columns(4)
    a.metric("mean correlation", f"{card['mean_corr']:.3f}")
    b.metric("mean RMSE (degC)", f"{card['mean_rmse']:.3f}")
    ml = card["bands"]["mixed_layer_0_50m"]
    tc = card["bands"]["thermocline_75_300m"]
    c.metric("mixed layer RMSE (degC)", f"{ml['mean_rmse']:.3f}" if ml else "-")
    if base:
        imp = 100 * (base["mean_rmse"] - card["mean_rmse"]) / base["mean_rmse"]
        d.metric("vs RF baseline", f"{imp:+.1f}% RMSE")
    else:
        d.metric("thermocline RMSE (degC)", f"{tc['mean_rmse']:.3f}" if tc else "-")
    with st.expander("per-depth scorecard"):
        st.code(format_table(card))
    st.caption(
        "Skill is highest in the mixed layer and decays below the thermocline, where the "
        "surface no longer constrains temperature. We report that rather than hide it."
    )
else:
    st.info("No scorecard yet.")
    st.code("python scripts/03b_baseline.py\npython scripts/04_evaluate.py", language="bash")
