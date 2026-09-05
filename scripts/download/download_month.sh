#!/bin/bash
# Download one month of all six products, subset to the project domain.
#
#   bash scripts/download/download_month.sh 2022-01
#
# The last day of the month is worked out for you. ARGO is NOT here -- it comes
# from the INCOIS web interface, see PERSON1_DATA.md.
#
# Dataset ids are the REPROCESSED / multi-year streams. That is deliberate: the
# near-real-time streams do not reach back to 2022 (OSTIA NRT starts 2024-01-17,
# DUACS NRT starts 2022-10-04), and mixing streams across months puts artificial
# jumps into the training data. Coverage verified against the catalogue:
#
#   METOFFICE-GLO-SST-L4-REP-OBS-SST                          1981-10 .. 2026-03
#   cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D  1993-01 .. 2026-01
#   cmems_obs-mob_glo_phy-sss_my_multi_P1D                    1993-01 .. 2024-12
#   cmems_mod_glo_phy_my_0.083deg_P1D-m                       1993-01 .. 2026-06
#
# All four cover Jan 2022 through Oct 2024, so the whole collection is
# internally consistent. There is no 0.25 degree GLORYS in the my stream --
# 1/12 degree is the only choice, about 0.5 GB per month for thetao alone.

# anchor to the repo root, so this works from anywhere
cd "$(dirname "$0")/../.." || exit 1

PY=python
command -v python >/dev/null 2>&1 || PY=python3

M="$1"
if [ -z "$M" ]; then
  echo "Usage: bash scripts/download/download_month.sh YYYY-MM"
  echo "Example: bash scripts/download/download_month.sh 2022-01"
  exit 1
fi

# Last day of the month, so nobody has to remember February.
LAST=$($PY -c "
import calendar,sys
y,m=sys.argv[1].split('-'); print(calendar.monthrange(int(y),int(m))[1])" "$M")

# NOTE the T23:59:59. A bare date means midnight, and the daily products are
# stamped at 12:00, so '--end-datetime 2022-01-31' silently DROPS the 31st.
BOX="--minimum-longitude 45 --maximum-longitude 105 --minimum-latitude 5 --maximum-latitude 30"
DATES="--start-datetime ${M}-01T00:00:00 --end-datetime ${M}-${LAST}T23:59:59"

mkdir -p data/raw/{ostia,duacs,sss,oscar,ccmp,glorys}

# Deliberately NOT `set -e`: if one product fails we still want the other five,
# and a summary at the end telling us exactly which to retry.
FAILED=""
try() { name="$1"; shift; echo "=== $name ==="; if ! "$@"; then FAILED="$FAILED $name"; fi; }

try SST copernicusmarine subset -i METOFFICE-GLO-SST-L4-REP-OBS-SST \
  --variable analysed_sst $BOX $DATES \
  -o data/raw/ostia --output-filename ostia_$M.nc

try SLA copernicusmarine subset -i cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D \
  --variable sla $BOX $DATES \
  -o data/raw/duacs --output-filename duacs_$M.nc

try SSS copernicusmarine subset -i cmems_obs-mob_glo_phy-sss_my_multi_P1D \
  --variable sos $BOX $DATES \
  -o data/raw/sss --output-filename sss_$M.nc

# thetao only, capped at 1100 m -- the file carries ~10 variables and 50 levels
# otherwise, and we use exactly one variable down to 1000 m.
try GLORYS copernicusmarine subset -i cmems_mod_glo_phy_my_0.083deg_P1D-m \
  --variable thetao $BOX $DATES \
  --minimum-depth 0 --maximum-depth 1100 \
  -o data/raw/glorys --output-filename glorys_$M.nc

try CURRENTS podaac-data-downloader -c OSCAR_L4_OC_FINAL_V2.0 -d data/raw/oscar \
  -sd ${M}-01T00:00:00Z -ed ${M}-${LAST}T23:59:59Z -b="45,5,105,30"

try WINDS podaac-data-downloader -c CCMP_WINDS_10M6HR_L4_V3.1 -d data/raw/ccmp \
  -sd ${M}-01T00:00:00Z -ed ${M}-${LAST}T23:59:59Z -b="45,5,105,30"

echo
echo "=============================================="
for d in ostia duacs sss glorys oscar ccmp; do
  n=$(ls data/raw/$d/*.nc 2>/dev/null | wc -l)
  sz=$(du -sh data/raw/$d 2>/dev/null | cut -f1)
  printf "  %-8s %3s files  %6s\n" "$d" "$n" "$sz"
done

if [ -n "$FAILED" ]; then
  echo
  echo "  FAILED:$FAILED  -- rerun this script, completed products are skipped or overwritten cleanly"
  exit 1
fi

echo
echo "  $M downloaded ($LAST days). Next:"
echo "    python scripts/00_check_real_data.py"
echo "    python scripts/01_harmonize_real.py"
echo "    mv data/processed/harmonized.nc harmonized_$M.nc     # upload this to Drive"
echo "    rm -f data/raw/*/*.nc                                 # clear before the next month"
