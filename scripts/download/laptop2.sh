#!/bin/bash
# LAPTOP 2 -- collect 2023 and stage it for upload.
#
#   bash scripts/download/laptop2.sh
#
# Downloads, checks, harmonizes and stages four months. Expect a few hours,
# mostly GLORYS. Safe to stop and rerun -- finished months are skipped.

bash "$(dirname "$0")/collect.sh" 2023-01 2023-04 2023-07 2023-10
