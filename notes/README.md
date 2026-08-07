# notes/

Research records: prior-art research, kill-condition verdicts, design reviews, and
the planning documents the project is being built against. Prose only — no code, no
results. Following the `gapcache` convention, where the prior-art verdict was written
before the build and ended up being the most valuable file in the repo.

## Here now

| File | What it is |
|---|---|
| `plan_v3.md` | The execution plan. Start here. Standalone — you can execute from it without reading v1 or v2. |
| `design_review_v2.md` | Disk-verified review of v2. Ten findings, four consequential. Explains why v3 differs from v2 rather than just asserting it. |
| `plan_v2_assessment.md` | Superseded. The cost reframe that named admission control as the spine and killed the determinism framing. |
| `plan_v1_determinism_review.md` | Retired direction. Kept because L82–96 holds the controlled-variables contract, which v3 inlines and which matters more under a cost framing, not less. |

## Not written yet — both are W0, both block code

| File | What it must contain |
|---|---|
| `prior_art.md` | The search, in full. Forward citations of RPA and BucketServe; compiled-shape bucketing outside LLMs (TensorRT optimization profiles, ONNX Runtime dynamic shapes, TF Serving batch buckets, `tf.data.bucket_by_sequence_length`); **JAX/XLA shape polymorphism** (`jax.export` polymorphic dims — if prefill compiles polymorphically the ladder is optional and the premise weakens badly); varlen serving (ByteTransformer, Effective Transformer); admission and batch composition (Sarathi-Serve, QLM, Andes, Llumnix) — specifically, has anyone published promote-and-pad vs queue-and-wait?; and **Vidur**, which a reviewer will ask about. |
| `kill_condition.md` | The verdict, in `gapcache/notes/kill_condition.md`'s format: does the gate fire, what exactly is already claimed by whom, what is genuinely still open, and a recommendation that does not quietly decide the project's fate unilaterally. |

`gapcache`'s prior-art pass surfaced five systems its reading list never anticipated
and ended that project in a week. That is the outcome this directory exists to make
possible cheaply.
