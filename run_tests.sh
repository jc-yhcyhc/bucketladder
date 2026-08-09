#!/usr/bin/env bash
# Everything that can be verified without a TPU. Run before every commit.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
PY=${PY:-.venv/bin/python}

echo "=== shell syntax ==="
for f in infra/*.sh run_tests.sh; do bash -n "$f" && echo "  OK  $f"; done

echo; echo "=== unit tests ==="
"$PY" -m pytest tests/ -q

echo; echo "=== infra dry-runs (must not touch gcloud) ==="
for s in create_tpu teardown_tpu setup_gcs deploy capture; do
  ./infra/$s.sh --dry-run >/dev/null 2>&1 && echo "  OK  $s.sh --dry-run"
done
bash infra/vm_setup.sh --dry-run >/dev/null && echo "  OK  vm_setup.sh --dry-run"

echo; echo "=== e00 end-to-end, mock mode ==="
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
"$PY" scripts/e00_smoke_test.py --config configs/e00_default_ladder.json --mock --results-root "$TMP" >/dev/null
"$PY" scripts/e00_smoke_test.py --config configs/e00_gap512_ladder.json  --mock --results-root "$TMP" >/dev/null
echo "  OK  2 runs, $(wc -l < "$TMP/MANIFEST.jsonl") manifest entries"

echo; echo "=== e00 against REAL hardware logs (v5litepod-4, 2026-08-09) ==="
"$PY" scripts/e00_smoke_test.py --config configs/e00_default_ladder.json \
      --warmup-log tests/fixtures/real_v5e4_default.log --results-root "$TMP/real-default"
"$PY" scripts/e00_smoke_test.py --config configs/e00_gap512_ladder.json \
      --warmup-log tests/fixtures/real_v5e4_gap512.log --results-root "$TMP/real-gap512"

echo; echo "=== session-2 experiments, mock mode, BOTH hypotheses ==="
"$PY" scripts/e03_noise_floor.py  --config configs/e03_noise_floor.json  --mock --results-root "$TMP/e03" | tail -2
"$PY" scripts/e01_oracle_gap.py   --config configs/e01_marginal_cost.json --mock                --results-root "$TMP/e01s" | tail -1
"$PY" scripts/e01_oracle_gap.py   --config configs/e01_marginal_cost.json --mock --mock-linear  --results-root "$TMP/e01l" | tail -1
"$PY" scripts/e02_stock_baseline.py --config configs/e02_stock_baseline.json --mock --mock-policy promote --results-root "$TMP/e02p" | grep VERDICT
"$PY" scripts/e02_stock_baseline.py --config configs/e02_stock_baseline.json --mock --mock-policy queue   --results-root "$TMP/e02q" | grep VERDICT

echo; echo "=== controlled-variable contract aborts on the bad config ==="
if "$PY" scripts/e00_smoke_test.py --config configs/e00_BAD_apc_unrecorded.json --mock --results-root "$TMP" 2>/dev/null; then
  echo "  FAIL: bad config was accepted"; exit 1
else
  echo "  OK  aborted with exit $?"
fi

echo; echo "ALL CHECKS PASSED"
