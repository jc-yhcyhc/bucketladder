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

echo; echo "=== e00 against a realistic vLLM log fixture (closest to a real run) ==="
"$PY" scripts/e00_smoke_test.py --config configs/e00_default_ladder.json \
      --warmup-log tests/fixtures/vllm_tpu_warmup.log --results-root "$TMP/fx"

echo; echo "=== controlled-variable contract aborts on the bad config ==="
if "$PY" scripts/e00_smoke_test.py --config configs/e00_BAD_apc_unrecorded.json --mock --results-root "$TMP" 2>/dev/null; then
  echo "  FAIL: bad config was accepted"; exit 1
else
  echo "  OK  aborted with exit $?"
fi

echo; echo "ALL CHECKS PASSED"
