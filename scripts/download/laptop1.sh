#!/bin/bash
# LAPTOP 1 -- collect 2022 and stage it for upload.
#
#   bash scripts/download/laptop1.sh
#
# Downloads, checks, harmonizes and stages four months. Expect a few hours,
# mostly GLORYS. Safe to stop and rerun -- finished months are skipped.

bash "$(dirname "$0")/collect.sh" 2022-01 2022-04 2022-07 2022-10
