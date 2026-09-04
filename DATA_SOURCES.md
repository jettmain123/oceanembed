# Data sources

Fill this in as real products land. Judges and the technical report both want it,
and it takes two minutes if you do it at download time rather than at hour 19.

## Products used

| Canonical | Product | Version / dataset-id | Resolution | Access date | DOI / link |
|---|---|---|---|---|---|
| sst   | OSTIA L4 SST | METOFFICE-GLO-SST-L4-NRT-OBS-SST-V2 | 0.05 deg daily | | |
| sss   | | | | | |
| sla   | DUACS L4 | cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.25deg_P1D | 0.25 deg daily | | |
| ucur  | OSCAR | | 0.25 deg | | |
| vcur  | OSCAR | | 0.25 deg | | |
| uwind | CCMP v3.1 | | 0.25 deg 6-hourly | | |
| vwind | CCMP v3.1 | | 0.25 deg 6-hourly | | |
| temp (target) | GLORYS12V1 | cmems_mod_glo_phy_my_0.083deg_P1D-m | 1/12 deg daily, 50 levels | | |
| temp (validation) | Gridded ARGO | INCOIS LAS | | | |

## Window and domain

- Domain: 5-30 N, 45-105 E, regridded to 0.25 deg
- Date range actually used: `____ to ____` (must be the overlap of ALL products)
- Train / val / test split: chronological -- earliest days train, later days val,
  ARGO is the independent test

## Unit conversions applied

- OSTIA SST is in **Kelvin**; `harmonize.kelvin_to_celsius` converts it
  automatically (only when the median value exceeds 100).
- GLORYS native depth levels are linearly interpolated onto the 15 standard
  levels by `harmonize.interp_to_depths`.
- Sub-daily products (CCMP) are averaged to daily by `harmonize.daily_resample`.

## Missing data

Paste the table `01_harmonize_real.py` prints at the end (percent of OCEAN cells
that are NaN, per variable):

```
  sst     __%
  sss     __%
  sla     __%
  ucur    __%
  vcur    __%
  uwind   __%
  vwind   __%
  temp    __%
```

Samples whose patch or target contains any NaN are dropped in `02_build_dataset.py`.

## ARGO co-location

- Profiles co-located: `____`
- Region / dates covered: `____`
- Median time gap between an ARGO observation and the matched model day: `____ days`

Reported by `02b_argo_colocate.py`.
