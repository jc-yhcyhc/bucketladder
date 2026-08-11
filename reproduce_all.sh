#!/usr/bin/env bash
# =============================================================================
# reproduce_all.sh — regenerate every number in the paper from captured runs.
# =============================================================================
# No hardware, no manual steps, no network. Everything the paper claims is
# recomputed from `captured/`, which holds the raw output of all twelve sessions.
#
# This is the project's own verification bar, stated in plan_v3 and unbuilt until
# now. It exists because the two checks that preceded it each found real defects
# on their first run -- paper_numbers.py found a transcription error and a table
# presented as more precise than its data, and the invariance guardrail flagged
# five claims already believed correct. A script that regenerates everything is
# the only way to know that stays true after an edit.
#
# Exit non-zero if anything fails to reproduce. Silence is not success.
#
# Usage:
#   ./reproduce_all.sh            # verify
#   ./reproduce_all.sh --figures  # also regenerate figures
# =============================================================================

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
PY="${PY:-$HERE/.venv/bin/python}"
[[ -x "$PY" ]] || PY=python3

FIGURES=false
[[ "${1:-}" == "--figures" ]] && FIGURES=true

FAILED=0
step() {
  local name="$1"; shift
  printf '\n\033[1m== %s ==\033[0m\n' "$name"
  if "$@"; then
    printf '   \033[32mOK\033[0m  %s\n' "$name"
  else
    printf '   \033[31mFAILED\033[0m  %s\n' "$name"
    FAILED=$((FAILED + 1))
  fi
}

# 1. The harness itself. Every experiment's analysis path is covered here, and
#    several tests pin errors that cost hardware time to discover -- a test that
#    encodes an assumption is only as good as the assumption, so these are the
#    first thing to break when one is wrong.
step "unit tests (184)" "$PY" -m pytest tests/ -q

# 2. Every claim in the paper, recomputed from captured/ and diffed against the
#    stated value, plus the invariance guardrail over derived claims.
step "paper claims + invariance guardrail" "$PY" scripts/paper_numbers.py

# 3. The cost model refit and both holdouts -- seeds the fit never saw, and
#    rates the fit never saw. Regenerates sim/measured_cost_curve.json.
step "cost model refit + 2 holdouts" "$PY" scripts/refit_cost_model.py

# 4. The padding headroom, recomputed from per-step histograms rather than read
#    from a stored table.
step "padding headroom (H1)" "$PY" scripts/h1_headroom.py \
    --results-root "$HERE/results/repro"

# 5. Offline analyses that depend on the refitted curve. These are simulation,
#    not measurement, and are included so a curve change cannot silently
#    invalidate a downstream conclusion.
step "offline optimum (e21)" "$PY" scripts/e21_offline_optimum.py \
    --config configs/e21_offline_optimum.json --results-root "$HERE/results/repro"
step "policy sweep (e30)" "$PY" scripts/e30_policy_sweep.py \
    --config configs/e30_policy_sweep.json --results-root "$HERE/results/repro"

if $FIGURES; then
  step "figures" "$PY" scripts/make_figures.py --out "$HERE/figures"
fi

printf '\n'
if [[ $FAILED -eq 0 ]]; then
  printf '\033[32mALL REPRODUCED\033[0m — every paper number regenerated from captured/\n'
  exit 0
fi
printf '\033[31m%d STEP(S) FAILED\033[0m — the paper does not currently reproduce\n' "$FAILED"
exit 1
