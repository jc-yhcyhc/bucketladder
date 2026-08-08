#!/usr/bin/env bash
# =============================================================================
# find_zone.sh — list every zone that can actually run our TPU config.
# =============================================================================
# Spot capacity, not quota, is the binding risk: quota is 16 v5e chips almost
# everywhere, but capacity is per-zone and transient. v5e machine types exist in
# ~25 zones across four continents, so a capacity failure in one zone is a
# retry elsewhere, not a blocked session.
#
# Cross-references three things per zone:
#   1. does the accelerator type exist there
#   2. does the region hold quota for it
#   3. (informational) does the region hold v6e spot quota as a fallback
#
# Usage:
#   ./infra/find_zone.sh              # zones for the configured TPU_TYPE
#   ./infra/find_zone.sh --family v6e # v6e alternatives (SPOT ONLY — see below)
# =============================================================================

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./config.env
source "$HERE/config.env"

FAMILY="v5e"
for arg in "$@"; do
  case "$arg" in
    --family) shift; FAMILY="${1:-v5e}" ;;
    v5e|v6e) FAMILY="$arg" ;;
    -h|--help) sed -n '2,20p' "${BASH_SOURCE[0]}"; exit 0 ;;
  esac
  shift || true
done

case "$FAMILY" in
  v5e) PREFIX="ct5lp"; METRIC="TPU_LITE_PODSLICE_V5" ;;
  v6e) PREFIX="ct6e";  METRIC="PREEMPTIBLE_TPU_V6E" ;;
  *) echo "unknown family: $FAMILY (v5e|v6e)" >&2; exit 2 ;;
esac

echo "Zones offering $FAMILY ($PREFIX*), with regional quota for $METRIC:"
echo

gcloud compute machine-types list --filter="name~^${PREFIX}" \
  --format='value(zone)' 2>/dev/null | sort -u \
| while read -r z; do
    r="${z%-*}"
    q=$(gcloud compute regions describe "$r" --format="value(quotas)" 2>/dev/null \
        | tr ';' '\n' | grep -F "'$METRIC'" \
        | sed -E "s/.*'limit': ([0-9.]+).*/\1/" | head -1)
    printf "  %-26s region=%-22s %s=%s\n" "$z" "$r" "$METRIC" "${q:-<not in regions describe; see alpha quota API>}"
  done

cat <<'NOTE'

NOTE on v6e: we hold PREEMPTIBLE (spot) v6e quota only — there is no
non-preemptible tpu_v6e metric on this project. That rules v6e out as the
primary platform, because plan_v4.md requires on-demand for W0b bring-up
(stability while debugging) and W7-8 holdout (a preempted holdout is a
corrupted holdout). v5e is the only family where we hold both on-demand and
spot. Keep v6e as a fallback for the re-runnable primitives block only, and
only if v5e capacity fails repeatedly — switching platforms mid-study costs a
hardware confound and a second bring-up.
NOTE
