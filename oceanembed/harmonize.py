"""Harmonization -- put any collection of ocean fields on the common grid.

Output is Contract B: data/processed/harmonized.nc with

    coords    time (daily), depth (15 standard levels), lat, lon (0.25 deg)
    data_vars sst,sss,sla,ucur,vcur,uwind,vwind  (time,lat,lon)
              temp                                (time,depth,lat,lon)
              land_mask                           (lat,lon)

Everything downstream reads only that file. Both the synthetic path
(01_harmonize.py) and the real path (01_harmonize_real.py) end here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from . import load_config

__all__ = [
    "target_grid",
    "normalize_coords",
    "to_datetime64",
    "kelvin_to_celsius",
    "regrid_to_target",
    "interp_to_depths",
    "daily_resample",
    "harmonize",
    "harmonize_from_cube",
    "missing_report",
]


def target_grid(cfg: dict | None = None):
    """The canonical (lat, lon) axes of the harmonized cube."""
    cfg = cfg or load_config()
    d = cfg["domain"]
    res = d["resolution"]
    lat = np.arange(d["lat_min"], d["lat_max"] + 1e-6, res, dtype=np.float32)
    lon = np.arange(d["lon_min"], d["lon_max"] + 1e-6, res, dtype=np.float32)
    return lat, lon


def to_datetime64(values) -> np.ndarray:
    """Coerce any time axis to numpy datetime64.

    Some products (OSCAR) declare a non-standard calendar, so xarray decodes
    their time axis into cftime objects rather than datetime64. Those cannot be
    aligned or compared against the datetime64 axes of every other product, so
    the merge silently produces an empty intersection.

    We read the calendar fields (year, month, day...) straight off each object.
    The declared calendar is taken at face value for the DATE it displays, which
    is what the daily filenames mean -- reinterpreting a Julian label as a real
    Julian calendar would shift these dates by about 13 days.
    """
    vals = np.asarray(values)
    if np.issubdtype(vals.dtype, np.datetime64):
        return vals
    out = []
    for t in vals.ravel():
        if hasattr(t, "year") and hasattr(t, "month") and hasattr(t, "day"):
            out.append(np.datetime64(
                f"{t.year:04d}-{t.month:02d}-{t.day:02d}"
                f"T{getattr(t, 'hour', 0):02d}:{getattr(t, 'minute', 0):02d}"
                f":{getattr(t, 'second', 0):02d}"
            ))
        else:
            out.append(np.datetime64(pd.Timestamp(t)))
    return np.asarray(out, dtype="datetime64[ns]").reshape(vals.shape)


def normalize_coords(obj: xr.Dataset | xr.DataArray) -> xr.Dataset | xr.DataArray:
    """Rename common coordinate aliases to time/depth/lat/lon and sort ascending.

    Real products use latitude/longitude/lev/deptht/valid_time and sometimes a
    descending latitude axis or 0-360 longitudes. This normalises all of that.
    """
    ren = {}
    for name in list(obj.coords) + list(getattr(obj, "dims", [])):
        low = str(name).lower()
        if low in ("latitude", "nav_lat", "y", "lat_bnds_dim"):
            ren[name] = "lat"
        elif low in ("longitude", "nav_lon", "x"):
            ren[name] = "lon"
        elif low in ("valid_time", "t", "date"):
            ren[name] = "time"
        elif low in ("lev", "level", "deptht", "z", "depth_levels"):
            ren[name] = "depth"
    if ren:
        obj = obj.rename(ren)

    # cftime -> datetime64, so products with odd calendars still align
    if "time" in obj.coords and obj["time"].dtype == object:
        try:
            obj = obj.assign_coords(time=to_datetime64(obj["time"].values))
        except Exception:
            pass

    if "lon" in obj.coords:
        lon = obj["lon"].values
        if np.nanmax(lon) > 180.0:
            obj = obj.assign_coords(lon=(((obj["lon"] + 180) % 360) - 180))
        obj = obj.sortby("lon")
    for c in ("lat", "time", "depth"):
        if c in obj.coords and obj[c].ndim == 1 and obj[c].size > 1:
            if np.asarray(obj[c][0]) > np.asarray(obj[c][-1]):
                obj = obj.sortby(c)
    return obj


def _fill_edges(da: xr.DataArray, dim: str) -> xr.DataArray:
    """Forward- then backward-fill NaNs along `dim` using pure NumPy.

    xarray's own ffill/bfill require bottleneck/numbagg; this keeps the pipeline
    dependency-light so it runs on every teammate's machine.
    """
    if dim not in da.dims:
        return da
    axis = da.get_axis_num(dim)
    arr = np.asarray(da.values)
    n = arr.shape[axis]
    if n < 2:
        return da
    a = np.moveaxis(arr, axis, 0)
    valid = np.isfinite(a)
    idx = np.where(valid, np.arange(n).reshape((n,) + (1,) * (a.ndim - 1)), 0)
    # forward fill
    fwd = np.maximum.accumulate(idx, axis=0)
    a = np.take_along_axis(a, fwd, axis=0)
    # backward fill for any leading NaNs
    idx_b = np.where(valid, np.arange(n).reshape((n,) + (1,) * (a.ndim - 1)), n - 1)
    bwd = np.minimum.accumulate(idx_b[::-1], axis=0)[::-1]
    filled = np.take_along_axis(a, bwd, axis=0)
    a = np.where(np.isfinite(a), a, filled)
    out = da.copy(data=np.moveaxis(a, 0, axis))
    return out


def kelvin_to_celsius(da: xr.DataArray) -> xr.DataArray:
    """Convert only if the values actually look like Kelvin (OSTIA SST does)."""
    finite = np.isfinite(da.values) if da.size < 5_000_000 else np.isfinite(da.values[:1])
    sample = da.values[np.isfinite(da.values)] if da.size < 5_000_000 else da.values[0]
    med = float(np.nanmedian(sample)) if np.size(sample) else np.nan
    if np.isfinite(med) and med > 100.0:
        out = da - 273.15
        out.attrs = dict(da.attrs)
        out.attrs["units"] = "degC"
        out.attrs["converted_from"] = "kelvin"
        return out
    return da


def regrid_to_target(obj, cfg: dict | None = None, method: str = "linear"):
    """Bilinear-interpolate onto the target 0.25 deg grid and subset the domain."""
    cfg = cfg or load_config()
    lat, lon = target_grid(cfg)
    obj = normalize_coords(obj)
    # subset first (with a halo) so interpolation stays cheap
    halo = 2.0
    obj = obj.sel(
        lat=slice(float(lat[0]) - halo, float(lat[-1]) + halo),
        lon=slice(float(lon[0]) - halo, float(lon[-1]) + halo),
    )
    out = obj.interp(lat=lat, lon=lon, method=method, kwargs={"fill_value": None})
    return out


def interp_to_depths(da: xr.DataArray, cfg: dict | None = None) -> xr.DataArray:
    """Interpolate a (…, depth, …) array onto the 15 standard levels.

    GLORYS levels never match the standard levels, so this is always needed on
    the real path. Values shallower/deeper than the source range are held at the
    nearest available level rather than left NaN.
    """
    cfg = cfg or load_config()
    depths = np.asarray(cfg["depths"], dtype=np.float32)
    da = normalize_coords(da)
    if "depth" not in da.dims:
        raise ValueError("interp_to_depths needs a 'depth' dimension")
    src = np.asarray(da["depth"].values, dtype=np.float32)
    if src.size == depths.size and np.allclose(src, depths, atol=1e-3):
        return da.assign_coords(depth=depths)
    out = da.interp(depth=depths, method="linear", kwargs={"fill_value": None})
    # hold the end members instead of producing NaN outside the source range
    out = _fill_edges(out, "depth")
    return out.assign_coords(depth=depths)


def daily_resample(obj, cfg: dict | None = None):
    """Force a daily time axis (mean within each day, then gap-fill by interp)."""
    cfg = cfg or load_config()
    if cfg["domain"].get("temporal", "daily") != "daily" or "time" not in getattr(obj, "dims", {}):
        return obj
    obj = obj.resample(time="1D").mean()
    obj = obj.interpolate_na(dim="time", method="linear", use_coordinate=False)
    if isinstance(obj, xr.Dataset):
        return obj.map(lambda d: _fill_edges(d, "time"))
    return _fill_edges(obj, "time")


def _build_land_mask(surface: dict, temp: xr.DataArray, cfg: dict) -> xr.DataArray:
    """Land = cells that are NaN on every day in the SST field (and in temp)."""
    lat, lon = target_grid(cfg)
    ref = surface.get("sst", next(iter(surface.values())))
    always_nan = (~np.isfinite(ref)).all(dim="time")
    if "time" in temp.dims and "depth" in temp.dims:
        always_nan = always_nan | (~np.isfinite(temp.isel(depth=0))).all(dim="time")
    mask = always_nan.astype("float32").rename("land_mask")
    mask.attrs = {"description": "1=land, 0=ocean", "derived_from": "all-time NaN in sst/temp"}
    return mask.assign_coords(lat=lat, lon=lon)


def harmonize(
    surface: dict[str, xr.DataArray],
    temp: xr.DataArray,
    cfg: dict | None = None,
    land_mask: xr.DataArray | None = None,
    already_on_grid: bool = False,
    verbose: bool = True,
) -> xr.Dataset:
    """Assemble Contract B from per-variable DataArrays.

    surface : {canonical_name: DataArray(time,lat,lon)} for the 7 surface vars
    temp    : DataArray(time,depth,lat,lon) target temperature
    """
    cfg = cfg or load_config()
    want = list(cfg["surface_vars"])
    missing = [v for v in want if v not in surface]
    if missing:
        raise KeyError(f"missing canonical surface variables: {missing}")

    out = {}
    for name in want:
        da = normalize_coords(surface[name])
        if name == "sst":
            da = kelvin_to_celsius(da)
        if not already_on_grid:
            da = regrid_to_target(da, cfg)
        da = daily_resample(da, cfg)
        out[name] = da.astype("float32").rename(name)
        if verbose:
            print(f"[harmonize] {name:6s} -> {dict(out[name].sizes)}")

    t = normalize_coords(temp)
    t = kelvin_to_celsius(t)
    if not already_on_grid:
        t = regrid_to_target(t, cfg)
    t = interp_to_depths(t, cfg)
    t = daily_resample(t, cfg)
    t = t.transpose("time", "depth", "lat", "lon").astype("float32").rename("temp")
    if verbose:
        print(f"[harmonize] temp   -> {dict(t.sizes)}")

    # align every field on the common time axis (intersection of all sources)
    ds = xr.Dataset(out)
    ds["temp"] = t
    ds = ds.dropna(dim="time", how="all", subset=["sst"]) if "sst" in ds else ds

    if land_mask is None:
        land_mask = _build_land_mask({k: ds[k] for k in want}, ds["temp"], cfg)
    else:
        land_mask = normalize_coords(land_mask)
        if not already_on_grid:
            land_mask = regrid_to_target(land_mask, cfg, method="nearest")
        land_mask = (land_mask > 0.5).astype("float32")
    ds["land_mask"] = land_mask.rename("land_mask")

    ds = ds.transpose("time", "depth", "lat", "lon", missing_dims="ignore")
    ds.attrs.update(
        {
            "title": "OceanEmbed harmonized cube (Contract B)",
            "resolution_deg": cfg["domain"]["resolution"],
            "domain": f"{cfg['domain']['lat_min']}-{cfg['domain']['lat_max']}N, "
                      f"{cfg['domain']['lon_min']}-{cfg['domain']['lon_max']}E",
            "depths_m": ",".join(str(d) for d in cfg["depths"]),
        }
    )
    return ds


def harmonize_from_cube(cube: xr.Dataset, cfg: dict | None = None, verbose: bool = True) -> xr.Dataset:
    """Harmonize a cube that already carries canonical names (the synthetic path)."""
    cfg = cfg or load_config()
    surface = {v: cube[v] for v in cfg["surface_vars"]}
    lm = cube["land_mask"] if "land_mask" in cube else None
    return harmonize(surface, cube["temp"], cfg=cfg, land_mask=lm, verbose=verbose)


def missing_report(ds: xr.Dataset, cfg: dict | None = None) -> dict[str, float]:
    """Percent of OCEAN cells that are NaN, per variable. Goes into DATA_SOURCES.md."""
    cfg = cfg or load_config()
    ocean = ds["land_mask"] < 0.5
    rep = {}
    for name in list(cfg["surface_vars"]) + ["temp"]:
        if name not in ds:
            continue
        da = ds[name]
        valid_total = int(ocean.sum()) * int(da.sizes.get("time", 1)) * int(da.sizes.get("depth", 1))
        nan_ocean = int((~np.isfinite(da) & ocean).sum())
        rep[name] = 100.0 * nan_ocean / max(valid_total, 1)
    return rep
