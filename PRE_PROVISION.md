# Pre-provision checklist

Everything that must be true before `./infra/create_tpu.sh` is run for the first
time. Derived from an audit, not from memory — four gaps were found and three
were closed; the remainder are yours.

## Blocking — you

- [ ] **Gated `meta-llama` access granted** on HuggingFace for
      `meta-llama/Llama-3.1-8B-Instruct`. Commonly a multi-hour stall, so start
      it first. https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
- [ ] **`export HF_TOKEN=hf_...`** (or `~/.cache/huggingface/token` present).
      `create_tpu.sh` treats a missing token as a preflight failure and refuses
      to provision — the model download would fail several minutes in, after
      billing had started.
- [ ] **`./infra/setup_gcs.sh`** — creates the results bucket. One-off, costs
      approximately nothing, and without it the session-end sync has nowhere to
      write. **Currently missing.**

## Blocking — verified already, no action

- [x] `v5litepod-4` offered in `us-central1-a` — checked live
- [x] Image family `ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e` resolves — checked live
- [x] Machine type `ct5lp-hightpu-4t` available in us-central1-a — checked live
- [x] Quota `TPU_LITE_PODSLICE_V5` = 16 chips, preemptible likewise (need 4) —
      checked live in us-central1 and us-east5. `TPU_LITE_DEVICE_V5` is 0, which
      is irrelevant: a `v5litepod-4` is a podslice, not a device.
- [x] **v5e is the right choice, and available in ~25 zones on four continents.**
      Quota is a global default of 16 chips (on-demand and spot), not a special
      grant. Exceptions: us-east1 has 0/0, us-west1 has 0 on-demand / 16 spot.
      `./infra/find_zone.sh` prints the live list.
- [x] **v6e spot quota also exists** (16 chips in 10 regions) but there is **no
      on-demand v6e**. Since bring-up and holdout both require on-demand, v5e is
      the only family that covers the whole study. v6e is a recorded fallback for
      the re-runnable primitives block only. Note `regions describe` does not show
      v6e quota at all — use `gcloud alpha services quota list`.
- [x] **GPUs do not help.** P100/K80/P4 cannot run vLLM (compute capability
      below the required 7.0). T4/L4 could, but a GPU is the paper's contrast
      case, not its platform.
- [x] `tpu-inference` 0.26.0 on PyPI alongside vLLM 0.26.0
- [x] `gcloud ... tpu-vm create` accepts `--version --accelerator-type --zone --spot`
- [x] Harness green: `./run_tests.sh` — 80 tests, all dry-runs, mock and
      fixture end-to-end, deliberate abort

## Not blocking session 1 — needed before session 2

- [ ] `scripts/e01_oracle_gap.py` — marginal-cost design (hold one bucket, vary
      occupancy over ~5 shapes). Does not exist.
- [ ] `scripts/e02_stock_baseline.py`. Does not exist.
- [ ] Noise-floor measurement — three repeats of one config; sets the threshold
      units for both gates. Does not exist.

Session 1 needs none of these: its only job is to capture a real warmup log and
tear down. Writing them before then would be guessing at a log format we have
not seen.

## Session 1, once the boxes above are ticked

This project uses the **GCE-native TPU path** ("Lightweight Exploration" in the
console). A TPU here is an ordinary Compute Engine instance with a TPU machine
type, so everything uses `gcloud compute ssh/scp` — **not** `gcloud compute tpus
tpu-vm ssh/scp`, which addresses Cloud TPU API nodes that this project cannot
create.

```bash
./infra/setup_gcs.sh                  # one-off
./infra/create_tpu.sh --check         # live machine-type/image/quota, no spend
./infra/create_tpu.sh                 # BILLING STARTS
./infra/deploy.sh
gcloud compute ssh bucketladder-v5e4 --zone=us-central1-a \
    --command='bash ~/bucketladder/infra/vm_setup.sh'
./infra/capture.sh --tag default      # THE DELIVERABLE
./infra/teardown_tpu.sh               # BILLING STOPS
./infra/teardown_tpu.sh --status      # must report nothing billing
```

Optionally repeat the middle with `VLLM_TPU_BUCKET_PADDING_GAP=512` before
teardown, for a second ladder, then `./infra/capture.sh --tag gap512`.

Record billed VM-hours in `DECISIONS.md`. Then, at $0:

```bash
python scripts/e00_smoke_test.py --config configs/e00_default_ladder.json \
    --warmup-log captured/default/vllm_warmup.log
```

Iterate on `parse_warmup_log` / `parse_server_config` until that passes. **Do not
start session 2 until it does.**

## What is most likely to break, in order

1. **Warmup-log format** — `Compiling graph for num_tokens=N` is inferred, never
   observed. This is why session 1 captures and leaves.
2. **`--no-enable-prefix-caching`** — flag name unverified against vLLM 0.26.
3. **`tpu-inference` install on `tpu-ubuntu2204-base`** — the image workarounds
   are inherited from a MaxText-era script and have never been run against this
   package.
4. **`vllm serve` as the entrypoint** for the tpu-inference backend — assumed.
5. **Spot capacity** — quota is confirmed, availability is not. Only a real
   provision attempt answers it.

All five are the interface to vLLM, which is deliberate: one surface to debug,
and `--warmup-log` mode lets it be debugged with the VM switched off.
