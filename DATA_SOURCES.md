# Data sources

Products actually used, checked against the datasets named in the problem
statement. Resolutions were verified from the downloaded files themselves (see
`scripts/00_check_real_data.py`), not taken on trust from the filenames.

## Products used

| Canonical | Product | File signature | Resolution (verified) | Variable | PS match |
|---|---|---|---|---|---|
| sst | OSTIA L4 GHRSST | `*-UKMO-L4_GHRSST-SSTfnd-OSTIA-GLOB-v02.0-fv02.0.nc` | 0.05 deg daily (500x1200 over 25x60 deg) | `analysed_sst` (Kelvin) | yes -- doi:10.48670/moi-00168 |
| sss | Copernicus multi-obs SSS (SMAP/SMOS) | `dataset-sss-ssd-rep-daily_*.nc` | 0.125 deg daily (200x480) | `sos` | yes -- doi:10.48670/moi-00051 |
| sla | DUACS L4 all-sat | `nrt_global_allsat_phy_l4_*.nc` | 0.25 deg daily (100x240) | `sla` | yes -- doi:10.48670/moi-00145 (NRT stream; confirm whether the DOI is NRT or reprocessed) |
| ucur, vcur | OSCAR L4 OC FINAL V2.0 | `oscar_currents_final_*.nc` | 0.25 deg daily (101x241) | `u`, `v` | yes -- PODAAC OSCAR_L4_OC_FINAL_V2.0 |
| uwind, vwind | CCMP v3.1 L4 | `CCMP_Wind_Analysis_*_V03.1_L4.nc` | 0.25 deg, 6-hourly (100x240) | `uwnd`, `vwnd` | yes -- PODAAC CCMP_WINDS_10M6HR_L4_V3.1 |
| temp (target) | GLORYS12V1 reanalysis | `mercatorglorys12v1_gl12_mean_*.nc` | 1/12 deg daily, 36 levels to 1062 m | `thetao` | CHECK against the PS target row |
| temp (validation) | Gridded ARGO, INCOIS LAS | `argo_temp_2024-10.nc` | gridded, 19 levels, 25x60 cells, 3 time steps | `temperature` | independent validation |

The PS lists ASCAT-L2-Coastal as an alternative wind source. CCMP was chosen
because it is already a gridded L4 analysis; ASCAT L2 is along-track swath data
and would need gridding first.

## Window and domain

- Domain: 5-30 N, 45-105 E, regridded to a common 0.25 deg grid
- Dates used: **2024-10-01 to 2024-10-31** (31 days, the overlap of all products)
- Split: chronological -- train 2024-10-01..22 (22 d), val 10-23..26 (4 d),
  holdout 10-27..31 (5 d). The holdout is never used for training or model
  selection.

## Unit and grid conversions applied

- OSTIA SST is in **Kelvin**; converted to degC automatically
  (`harmonize.kelvin_to_celsius`, triggered only when the median exceeds 100).
- CCMP is **6-hourly** (124 steps for 31 days); averaged to daily means.
- SSS carries a length-1 depth dimension; squeezed out.
- OSCAR declares a **non-standard (Julian) calendar**, so xarray decodes its
  time axis to cftime rather than datetime64. Coerced to datetime64 keeping the
  displayed date, otherwise it cannot align with the other products.
- GLORYS native 36 levels linearly interpolated onto the 15 standard depths.
  Interpolation fills **upward only**: 0 m may take the 0.5 m value, but levels
  below the seafloor stay NaN rather than inheriting the last valid value.
- All fields bilinearly regridded to 0.25 deg.

## Missing data (percent of OCEAN cells that are NaN)

```
  sst      0.00%
  sss      1.07%
  sla      0.47%
  ucur     4.07%
  vcur     4.07%
  uwind    0.00%
  vwind    0.00%
  temp    13.33%
```

`temp` at 13.33% is the continental shelf: GLORYS has no data below the seafloor,
so cells shallower than 1000 m are incomplete at the deep levels. Any sample with
a NaN anywhere in its patch or its 15-level target is dropped by
`02_build_dataset.py`, leaving 44,000 train / 8,000 val / 10,000 holdout profiles.

## Sanity checks performed

- Basin-mean profile falls monotonically, 29.19 degC at the surface to 7.74 degC
  at 1000 m, matching raw GLORYS (7.3 degC at its native 1062 m level).
- Every product independently confirmed to cover the full domain and all 31 days.

## ARGO co-location

- File: `argo_temp_2024-10.nc`, gridded, dims (time 3, depth 19, lat 25, lon 60)
- Co-located profiles: `____`
- Median gap between ARGO observation and matched model day: `____ days`
- **Open issue:** ARGO has only 3 time steps in October. If they do not fall
  within the 10-27..31 holdout window, the surface patches used to score them
  were seen during training. The ARGO measurements stay independent, but the
  comparison would be optimistic. Run
  `python scripts/02b_argo_colocate.py --dry-run` to see which days they land on.
