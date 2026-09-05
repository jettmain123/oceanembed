#!/bin/bash
# LAPTOP 3 -- collect 2024 and stage it for upload.
#
#   bash scripts/download/laptop3.sh
#
# Downloads, checks, harmonizes and stages four months. Expect a few hours,
# mostly GLORYS. Safe to stop and rerun -- finished months are skipped.
# October 2024 is included on purpose. The first pull of it used the
# near-real-time products; everything else uses the reprocessed ones. Doing it
# again on the same stream removes the only inconsistency in the dataset.

bash "$(dirname "$0")/collect.sh" 2024-01 2024-04 2024-07 2024-10
