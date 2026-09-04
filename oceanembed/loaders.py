"""Real-product loaders -- open a downloaded NetCDF and hand back a canonical DataArray.

Every loader does the same three things:
  1. open the file(s) (globs are fine -- they are opened with open_mfdataset),
  2. pick the variable, trying the known names for that product first,
  3. rename coords to time/lat/lon(/depth) and return the DataArray.

Nothing here regrids or converts units -- harmonize.py owns that, so the same
code path serves the synthetic and the real cube.

The product-specific variable names live in PRODUCT_VARS below. If your download
uses a different name, add it there rather than editing the functions.
"""
from __future__ import annotations

import glob as _glob
from pathlib import Path

import xarray as xr

from .harmonize import normalize_coords

__all__ = [
    "PRODUCT_VARS",
    "open_any",
    "pick_var",
    "load_variable",
    "load_surface_stack",
    "load_glorys_temp",
    "load_argo_gridded",
]

# Known variable names per canonical field, most likely first.
# Add to these lists; do not rename the canonical keys.
PRODUCT_VARS: dict[str, list[str]] = {
    # SST -- OSTIA (Copernicus SST_GLO_SST_L4_NRT_OBSERVATIONS_010_001)
    "sst": ["analysed_sst", "sst", "sea_surface_temperature", "thetao", "SST"],
    # SSS -- SMAP / SMOS / Copernicus multiobs
    "sss": ["sos", "sss", "so", "smap_sss", "sea_surface_salinity", "salinity"],
    # SLA -- DUACS (SEALEVEL_GLO_PHY_L4)
    "sla": ["sla", "adt", "zos", "sea_level_anomaly"],
    # Currents -- OSCAR (PODAAC) or Copernicus geostrophic velocities
    "ucur": ["u", "ugos", "uo", "eastward_sea_water_velocity", "ugosa"],
    "vcur": ["v", "vgos", "vo", "northward_sea_water_velocity", "vgosa"],
    # Winds -- CCMP (PODAAC) or ERA5
    "uwind": ["uwnd", "u10", "eastward_wind", "u_wind", "U"],
    "vwind": ["vwnd", "v10", "northward_wind", "v_wind", "V"],
    # Target -- GLORYS12V1 reanalysis potential temperature
    "temp": ["thetao", "to", "temperature", "temp", "sea_water_potential_temperature"],
}


def open_any(path_or_glob: str | Path, **kw) -> xr.Dataset:
    """Open one file or a glob of files as a single Dataset."""
    s = str(path_or_glob)
    files = sorted(_glob.glob(s)) if any(c in s for c in "*?[") else [s]
    if not files:
        raise FileNotFoundError(f"no files match {s!r}")
    if len(files) == 1:
        return xr.open_dataset(files[0], **kw)
    return xr.open_mfdataset(files, combine="by_coords", **kw)


def pick_var(ds: xr.Dataset, canonical: str, override: str | None = None) -> xr.DataArray:
    """Find the DataArray for a canonical field inside an opened Dataset."""
    if override:
        if override not in ds:
            raise KeyError(f"{override!r} not in file (have: {list(ds.data_vars)})")
        return ds[override]
    for name in PRODUCT_VARS.get(canonical, []):
        if name in ds:
            return ds[name]
    raise KeyError(
        f"could not find a variable for {canonical!r}. Tried {PRODUCT_VARS.get(canonical)}. "
        f"File has: {list(ds.data_vars)}. Pass an explicit name, or add yours to PRODUCT_VARS."
    )


def load_variable(path_or_glob, canonical: str, var: str | None = None,
                  squeeze_dims: tuple[str, ...] = ("depth", "lev", "altitude", "nv")) -> xr.DataArray:
    """Open a product and return one canonical surface DataArray (time,lat,lon).

    Surface products sometimes carry a length-1 depth or altitude axis -- it is
    squeezed out so the result is strictly (time,lat,lon).
    """
    ds = open_any(path_or_glob)
    da = pick_var(ds, canonical, var)
    for dim in squeeze_dims:
        if dim in da.dims and da.sizes[dim] == 1:
            da = da.isel({dim: 0}, drop=True)
    da = normalize_coords(da)
    return da.rename(canonical)


def load_surface_stack(sources: dict[str, tuple[str, str | None]]) -> dict[str, xr.DataArray]:
    """Load all 7 surface fields.

    sources maps canonical name -> (path_or_glob, variable_name_or_None), e.g.

        {"sst": ("data/raw/ostia/*.nc", "analysed_sst"),
         "sla": ("data/raw/duacs/*.nc", None), ...}
    """
    out = {}
    for canonical, (path, var) in sources.items():
        out[canonical] = load_variable(path, canonical, var)
        print(f"[loaders] {canonical:6s} <- {path}  {dict(out[canonical].sizes)}")
    return out


def load_glorys_temp(path_or_glob, var: str | None = None) -> xr.DataArray:
    """GLORYS potential temperature -> (time,depth,lat,lon), coords normalised.

    Depth levels are NOT interpolated here -- harmonize.interp_to_depths puts
    them on the 15 standard levels.
    """
    ds = open_any(path_or_glob)
    da = normalize_coords(pick_var(ds, "temp", var))
    if "depth" not in da.dims:
        raise ValueError(f"expected a depth dimension, got dims {da.dims}")
    return da.transpose("time", "depth", "lat", "lon").rename("temp")


def load_argo_gridded(path_or_glob, var: str | None = None) -> xr.DataArray:
    """Gridded ARGO temperature from the INCOIS Live Access Server.

    Same shape contract as GLORYS: (time,depth,lat,lon). Monthly LAS products are
    fine -- 02b_argo_colocate.py matches each grid cell to the nearest model day.
    """
    ds = open_any(path_or_glob)
    da = normalize_coords(pick_var(ds, "temp", var))
    if "depth" not in da.dims:
        raise ValueError(f"expected a depth dimension, got dims {da.dims}")
    dims = [d for d in ("time", "depth", "lat", "lon") if d in da.dims]
    return da.transpose(*dims).rename("temp")
