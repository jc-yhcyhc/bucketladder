# Related work the draft is missing

Review, minor items: *"Pope et al., Efficiently Scaling Transformer Inference
(MLSys 2023), is the canonical treatment of batch-size-dependent inference cost
on TPU and derives the memory-bound/compute-bound transition that §4.3
rediscovers empirically; its absence is conspicuous. Also expected:
PagedAttention/vLLM, Orca, Sarathi-Serve, and DistServe or Splitwise."*

All correct. Written up here rather than dropped straight into the draft,
because two of these change more than the citation list.

---

## The one that is not a citation fix

**Pope et al., *Efficiently Scaling Transformer Inference*, MLSys 2023.**

This is the paper whose absence is actually a problem, and the review is being
polite about why. It derives the memory-bound/compute-bound transition on TPU
analytically, in terms of arithmetic intensity and the ratio of weight bytes to
per-token work — which is exactly what `m9_roofline.py` computed from our own
measurements this session, three sessions after the curve was first measured and
left unexplained.

**Honest statement of the relationship:** our §4.5 decode curve is a measured
instance of a transition Pope et al. characterised analytically. That is not a
contribution over them, and the draft must not imply it is. What is ours is the
*consequence* — that because the step reads the whole weight set regardless of
batch size, padding on the request dimension falls inside a floor the batch size
does not move, so a compiled-shape ladder's request dimension is nearly free.
Pope et al. do not discuss compiled shape ladders, padding, or bucketing.

This also strengthens the paper: the mechanism is not a post-hoc story we
invented to explain our own curve, it is a known regime our measurement lands in.

**Where it goes:** §2 (background), and cited again at §4.5 and in the roofline
discussion. It should be one of the first three citations in the paper.

## The one that reframes §4.6

**DistServe (OSDI 2024) / Splitwise (ISCA 2024) — prefill/decode
disaggregation.**

§4.6 finds that the run-to-run variance is a prefill phenomenon and that decode
is well-behaved. Disaggregation is the architectural response to exactly that
asymmetry, so these papers are the answer to the finding, not merely adjacent to
it. Citing them turns §4.6 from an observation into a result with a known
consequence — and pre-empts "so what?"

Note it also bounds our own advice: on a disaggregated deployment the padding
question splits into two independent questions, and this paper measures the
co-located case only. That belongs in the limitations section, not buried.

## Straightforward citation fixes

| Work | Why it is expected | Where |
|---|---|---|
| **PagedAttention / vLLM** (SOSP 2023) | We measure vLLM and never cite it. | §2, first mention of the stack |
| **Orca** (OSDI 2022) | Iteration-level scheduling is the mechanism producing the per-step batches this whole paper measures. | §2 |
| **Sarathi-Serve** (OSDI 2024) | Chunked prefill, which §4.4 explicitly controls for and disables. Currently the draft describes the mechanism without naming its source. | §2 and §4.4 |
| **RPA** (Ragged Paged Attention) | Cited by name throughout with no full citation at first mention. | §1 |
| **LENS** (arXiv 2606.18042) | Same — it is the subject of §4.2 and has no formal citation. | §1 |

## Acronyms unexpanded at first use

RPA, LENS, MAPE, TTFT, ITL, MFU. Fix on the next draft pass.

## Terms used before they are defined

The review flagged three, and all three are used as if defined:

- **"flatness 0.97"** — the ratio of measured cost growth to the growth
  proportional to length. 1.0 means cost rises exactly with real tokens; below
  1.0 means sublinear. Currently never stated.
- **"share of nominal padding paid"** — now defined once in
  `paper_numbers.py::m1_share` as `(measured − real) / (padded − real)`, where 0
  is padding-free and 1 is fully paid. The draft uses it three times and defines
  it nowhere.
- **"rejected by 44–618%"** — the per-cell percentage by which the
  batch-padding model's *prediction* exceeds the measurement. Needs its
  denominator stated, because the project has already had one spurious mismatch
  caused by two definitions of a percentage error with different denominators.

All three belong in §2, not inline at first use.
