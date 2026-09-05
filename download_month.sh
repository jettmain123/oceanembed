#!/bin/bash
# Usage: bash download_month.sh 2022-01 31
# Downloads one month of all six products, subset to the project domain.
set -e

M="$1"       # e.g. 2022-01
LAST="$2"    # last day of that month, e.g. 31

if [ -z "$M" ] || [ -z "$LAST" ]; then
  echo "Usage: bash download_month.sh YYYY-MM last_day_of_month"
  echo "Example: bash download_month.sh 2022-01 31"
  exit 1
fi

BOX="--minimum-longitude 45 --maximum-longitude 105 --minimum-latitude 5 --maximum-latitude 30"
DATES="--start-datetime ${M}-01 --end-datetime ${M}-${LAST}"

echo "=== SST (OSTIA REP) ==="
copernicusmarine subset -i METOFFICE-GLO-SST-L4-REP-OBS-SST \
  --variable analysed_sst $BOX $DATES \
  -o data/raw/ostia --output-filename ostia_$M.nc

echo "=== Sea level anomaly (DUACS MY, 0.125deg) ==="
copernicusmarine subset -i cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D \
  --variable sla $BOX $DATES \
  -o data/raw/duacs --output-filename duacs_$M.nc

echo "=== Salinity ==="
copernicusmarine subset -i cmems_obs-mob_glo_phy-sss_my_multi_P1D \
  --variable sos $BOX $DATES \
  -o data/raw/sss --output-filename sss_$M.nc

echo "=== GLORYS (0.083deg, thetao only, depth capped at 1100m) ==="
copernicusmarine subset -i cmems_mod_glo_phy_my_0.083deg_P1D-m \
  --variable thetao $BOX $DATES \
  --minimum-depth 0 --maximum-depth 1100 \
  -o data/raw/glorys --output-filename glorys_$M.nc

echo "=== Currents (OSCAR) ==="
podaac-data-downloader -c OSCAR_L4_OC_FINAL_V2.0 -d data/raw/oscar \
  -sd ${M}-01T00:00:00Z -ed ${M}-${LAST}T23:59:59Z -b="45,5,105,30"

echo "=== Winds (CCMP) ==="
podaac-data-downloader -c CCMP_WINDS_10M6HR_L4_V3.1 -d data/raw/ccmp \
  -sd ${M}-01T00:00:00Z -ed ${M}-${LAST}T23:59:59Z -b="45,5,105,30"

echo ""
echo "=== Done. Downloaded $M into data/raw/ ==="
