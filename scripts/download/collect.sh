#!/bin/bash
# Collect a list of months end to end: download, check, harmonize, stage for upload.
#
#   bash scripts/download/collect.sh 2022-01 2022-04 2022-07 2022-10
#
# Normally you do not call this directly -- run your laptop's script instead:
#   bash scripts/download/laptop1.sh
#
# For each month it will:
#   1. download all six products, subset to the domain
#   2. run the data check; skip the month if it fails
#   3. harmonize into one cube
#   4. verify the profile actually decreases with depth
#   5. move the cube to to_upload/ and clear the raw folders
#
# Safe to stop and rerun. Months already in to_upload/ are skipped, so a crash
# or a closed laptop costs you one month, not the whole run.

cd "$(dirname "$0")/../.." || exit 1
ROOT="$(pwd)"
MONTHS=("$@")

if [ ${#MONTHS[@]} -eq 0 ]; then
  echo "Usage: bash scripts/download/collect.sh YYYY-MM [YYYY-MM ...]"
  exit 1
fi

PY=python
command -v python >/dev/null 2>&1 || PY=python3

mkdir -p to_upload logs

echo "=========================================================="
echo " OceanEmbed collection"
echo " months : ${MONTHS[*]}"
echo " output : $ROOT/to_upload"
echo " logs   : $ROOT/logs"
echo "=========================================================="
echo

DONE=(); SKIPPED=(); FAILED=()

for M in "${MONTHS[@]}"; do
  OUT="to_upload/harmonized_$M.nc"
  LOG="logs/$M.log"

  if [ -f "$OUT" ]; then
    echo "[$M] already done ($(du -h "$OUT" | cut -f1)) -- skipping"
    SKIPPED+=("$M"); continue
  fi

  echo "----------------------------------------------------------"
  echo "[$M] starting  ($(date '+%H:%M'))"
  echo "----------------------------------------------------------"

  # start each month from a clean slate, so a half-finished previous month
  # cannot leak into this one's harmonized cube
  rm -f data/raw/*/*.nc

  echo "[$M] downloading..."
  if ! bash scripts/download/download_month.sh "$M" >>"$LOG" 2>&1; then
    echo "[$M] DOWNLOAD FAILED -- see $LOG"
    tail -5 "$LOG" | sed 's/^/       /'
    FAILED+=("$M(download)"); continue
  fi

  echo "[$M] checking the files..."
  if ! $PY scripts/00_check_real_data.py >>"$LOG" 2>&1; then
    echo "[$M] CHECK FAILED -- these lines say why:"
    grep -E "\[FAIL\]" "$LOG" | tail -6 | sed 's/^/       /'
    FAILED+=("$M(check)"); continue
  fi

  echo "[$M] harmonizing..."
  if ! $PY scripts/01_harmonize_real.py >>"$LOG" 2>&1; then
    echo "[$M] HARMONIZE FAILED -- see $LOG"
    tail -8 "$LOG" | sed 's/^/       /'
    FAILED+=("$M(harmonize)"); continue
  fi

  # The harmonize script prints a warning rather than failing, so check for it
  # here. A profile that warms with depth means the month is unusable and must
  # not be uploaded.
  if grep -q "WARNING: temperature INCREASES" "$LOG"; then
    echo "[$M] BAD PROFILE -- temperature increases with depth. NOT uploading."
    FAILED+=("$M(profile)"); continue
  fi

  if [ ! -f data/processed/harmonized.nc ]; then
    echo "[$M] no cube produced -- see $LOG"
    FAILED+=("$M(no output)"); continue
  fi

  mv data/processed/harmonized.nc "$OUT"
  rm -f data/raw/*/*.nc

  echo "[$M] done -> $OUT ($(du -h "$OUT" | cut -f1))"
  grep -E "^  (sst|temp) " "$LOG" | tail -2 | sed 's/^/       missing: /'
  grep -E "^  OK: falls from" "$LOG" | tail -1 | sed 's/^/       /'
  DONE+=("$M")
  echo
done

echo
echo "=========================================================="
echo " SUMMARY"
echo "=========================================================="
[ ${#DONE[@]}    -gt 0 ] && echo "  collected : ${DONE[*]}"
[ ${#SKIPPED[@]} -gt 0 ] && echo "  already had: ${SKIPPED[*]}"
[ ${#FAILED[@]}  -gt 0 ] && echo "  FAILED    : ${FAILED[*]}"
echo
echo "  files ready to upload:"
ls -lh to_upload/*.nc 2>/dev/null | awk '{printf "    %-34s %s\n", $NF, $5}'
echo
echo "  Upload everything in to_upload/ to the shared Drive folder"
echo "  oceanembed/harmonized/  then post in the group which months you did."

if [ ${#FAILED[@]} -gt 0 ]; then
  echo
  echo "  Rerun this same command to retry only the failed months --"
  echo "  finished ones are skipped automatically."
  exit 1
fi
