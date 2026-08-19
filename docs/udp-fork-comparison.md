# Sibling fork comparison: UDPSendToFailed/ninfer-4090

Date: 2026-08-17. Their branch: `feat/rtx-4090-sm89-native` at v0.9.0 (`8426c45a`).
Ours: `rtx4090-port` at `d78df936`.

Both forks descend from `Don-Chad/ninfer-3090` (merge base `c55c23bb`) and target
Qwen3.8-27B on a single RTX 4090. The work is largely complementary:

- **Ours**: attention-prefill retune for sm_89 (fp16-acc PV tiles, +30% kernel
  roofline), serve hardening (vision caps, 413 masking, swscale overflow,
  retained-slot `/slots`, `/metrics`, `context_window`), depth-sweep benchmarks.
- **Theirs**: KV-cache quantization (Hadamard-rotated 4/8-bit, E8 lattice 2/4-bit
  key codebooks), prompt-lookup draft speculation, L2 persistence for MTP weights,
  context envelope lift to 1M, Windows build support.

## Serve fix exchange

Their PR #1 describes the OpenAI tool-message content-part array bug (tool role
hard-required string content; agent clients such as Qwen Code send part arrays and
got 400). The same bug existed here; fixed in `d78df936` and deployed. Cross-link
posted on their PR.

## KV quantization evaluation (their branch, our 4090)

Method: their branch built unmodified for sm_89 (`-DCMAKE_CUDA_ARCHITECTURES=89`),
run against the official 16.96 GiB artifact on the same card that serves our
production config. NIAH: filler haystack, passphrase needle at a depth fraction,
exact-match answer, `temperature 0`, `enable_thinking false`; prefill rate
approximated as `prompt_tokens / wall` with `max_tokens 24`. Decode: greedy
512-token Go-code generation from a 45-token prompt (shallow), their
`--spec mtp --draft-tokens 4 --lm-head-draft`.

| Config | Max ctx loaded | KV runtime | NIAH (tokens@depth) | Prefill t/s | Decode t/s |
|---|---:|---:|---|---:|---:|
| their `int8` (control) | 96k | - | 59k@0.35 PASS | 1802 @59k | 139.2 |
| their `rk4v4-e8` | 200k | 4.92 GiB | 59k@0.35, 118k@0.8, 189k@0.35 all PASS | 1799 / 1499 / 1251 | 133.3 |
| their `rk2v4-e8` | 330k | 5.83 GiB | 118k@0.8, 314k@0.35 all PASS | 1404 @118k / 867 @314k | 130.3 |
| ours `int8` (deployed, reference) | 172k | - | exact 64k/128k | 1724 @88k / 1561 @128k | 148.6 (MTP3) / 152.6 (MTP4) |

Observations:

- **314k tokens on 24 GB with exact single-needle retrieval.** The 2-bit E8
  cylinder key codebook survives the easy quality bar at extreme depth.
- E8 modes cost ~4-6% decode vs int8 on their branch (139.2 -> 133.3 -> 130.3).
- At 59k depth, prefill is identical across int8 and rk4v4-e8 (1802 vs 1799) on
  their branch, and within ~5% of our retuned int8 at ~118k (1499 vs our 1561 at
  128k) despite their branch lacking the fp16-acc PV retune - the halved K
  traffic in attention roughly offsets it. Combining their KV modes with our
  retune should therefore beat our current depth prefill.
- Their decode is ~7-9% below ours at matched int8 (139.2 vs 148.6-152.6),
  consistent with their README ranges; their prompt-lookup only fires when MTP
  returns zero drafts, so it does not move MTP-on numbers.
- Runtime-per-token from load logs: rk4v4-e8 ~26.4 KB/token, rk2v4-e8
  ~19.0 KB/token, vs our measured int8 slope ~35.9 KB/token. Projected ceilings
  on our deployment: ~234k (rk4v4-e8) and ~325k (rk2v4-e8; needs their
  `kGqaAttentionMaximumVisibleKeys` lift `1da9ed5a` above 262k).

## Port assessment

Worth porting (in order):

1. `rk4v4-e8` (and the rk4v4/rk8v4 plumbing it builds on): commits `0a39efe3`,
   `0701b973`, `46116bf2`, `e4616151`, `9e774969`. The codec is isolated in
   `e8_lattice.cuh` + `e8_root_codec.cuh`; the integration touches
   `gqa_attention_prefill_i8.cuh` / `gqa_attention_decode_i8.cuh`, which conflict
   with our sm_89 retune (`ce50e995`) - the merge must re-apply the retune's
   maxnreg/producer-warp/PRMT/fp16-acc structure around their dequant hooks.
2. Context envelope lift `1da9ed5a` (only needed above 262k; small, clean).
3. Skip: prompt-lookup stack (`aee631c1`, `fc65ff96`, `bdf4a90b`), L2 policy
   (`3482bc46`), W8 double-buffer (`7c5970b2`) - measured decode with all of them
   is below our current numbers; revisit only for MTP-off serving.

Open questions before switching production KV dtype:

- Single-needle NIAH is a weak bar. Run multi-needle and code-repair-at-depth
  probes before trusting 2-bit keys for agent traffic; their README cosine
  similarities (98.7% rk4v4-e8, 96.2% rk2v4-e8) imply measurable degradation.
- MTP acceptance under E8 KV at depth is unmeasured (shallow decode only).
- Their bench matrix was Windows/CUDA 13.3; our numbers here are the Linux
  container stack.

## Port results (2026-08-17)

All six KV commits are merged onto `rtx4090-port` (`7c2ce91e..ec56f922` plus the
`2619adf7` request-log name fix). Merge notes:

- Their mode machinery never overlaps the `ce50e995` retune except at kernel
  signatures: fill kernels, staging lambda, and Q-quant loop hooks all landed
  automatically; every hook site was diffed against their tip. The ops layer is
  byte-identical to their branch except `gqa_attention_prefill_i8.cuh` (retune +
  modes interleaved) and the restored `int8-group64` request-log name.
- Their 1 GiB CUDA-graph allowance bump (`layouts_impl.h`) was NOT taken: with our
  allowances the INT8 172032 profile still loads (136 MiB slack); with theirs it
  cannot. No graph-allowance errors appeared in any E8 run.
- Upstream bug found: their `kv_cache_name` rename to "int8" breaks
  `test_request_log`, invisible on their side because the Windows flow never runs
  ctest. `ninfer_test_e8_codec` builds but is not registered with ctest.
- `ninfer_state_store_test` failed once under `ctest -j8` (GPU contention with the
  new codec test); serial and isolated runs pass 84/84 consistently.

Measured on the merged branch (same probes as the pre-port eval, same card):

| Mode | Ctx loaded | KV runtime / slack | NIAH | 5-needle @118k | Code detail @168k | Prefill t/s | Decode | MTP acc @111k |
|---|---:|---|---|---|---|---|---:|---:|
| `int8` | 172032 | 6.31 / 0.13 GiB | 59k, 118k pass | - | - | 1869 @59k / 1604 @118k | 134.2 | 78.4% |
| `rk4v4-e8` | 262144 | 5.08 / 1.37 GiB | 59k-260k all pass | 5/5 | 3/3 | 1846 / 1575 / 1347 @189k / 1172 @260k | 126.6 | 78.8% |
| `rk2v4-e8` | 262144 | 4.01 / 2.43 GiB | 260k pass | 5/5 | 3/3 | 1059 @260k | 120.5 | - |

The retune carries into E8: at matched depth the merged branch beats their branch by
+2.6% @59k growing to +7.7% @189k on rk4v4-e8 prefill. E8 keys cost nothing in MTP
acceptance at depth. **Deployed config since 2026-08-17: `rk4v4-e8` at the full
native 262,144 context** - the 24 GB card now serves the same context window as the
32 GB 5090. `rk2v4-e8` adds slack, not context (262,144 is the model's own limit);
it stays available for a future vision-plus-long-context profile.

## Second wave (their 2026-08-18 evening push, assessed 2026-08-18)

Sixteen commits (`0a925796..6d3fd165`), all with claimed bit-exact parity and a
green 84/84 suite on their side. Disposition per group:

- **`--vision-max-tokens` (6d3fd165): ported.** The vision scratchpad drops from a
  hardcoded 32768 tokens to a configurable default of 8192 and frees about 1.5 GiB.
  Cherry-picked clean. Their commit leaves the processor budget
  (`max_vision_tokens`, still 32768) out of sync with the shrunken workspace: a
  request with 8K-32K image tokens passes the budget check and reaches the
  undersized encoder. This fork wires the budget to the same limit
  (`fix(frontend)` follow-up commit), so the failure is a clean
  `media_budget_exceeded`. With the port, `rk4v4-e8` serves the full native
  262,144 context with `--vision` at 780 MiB slack - the 208K practical line and
  the vision-against-context tradeoff are gone.
- **CUDA-graph allowance tightening (c85db47a): skipped.** They replace their old
  1 GiB SM86/SM89 per-lane padding with flat 64 MiB (ordinary) / 256-320 MiB (MTP)
  allowances. This branch already carries the per-topology-class accounting, which
  measures 8 MiB / 86 MiB for the same profiles - tighter than their new flat
  values. Their commit is a catch-up, not a win.
- **Causal-tile partitioned prefill attention (c5f70526): ported 2026-08-19
  (commit 694e01f0), re-implemented inside the retuned kernel.** Our baseline
  already skipped masking on interior tiles (`full_score_tile`), so their
  headline could not transfer whole; what did transfer is the structural split:
  a FullTile-tagged instantiation of the key-block body with unconditional
  staging, no masking, and no softmax zero-selects, dispatched per block from a
  precomputed `n_full_blocks`. Their bf16 kernel and common-header changes were
  taken verbatim (identical base). Measured on the INT8 `d256-h24-kv4` append
  shape at W=1024: 144 to 165 TFLOP/s at 32K-224K context (-12 to -13%
  latency), 200 to 171 us shallow, registers stay at 128 with no spills,
  84/84 tests bit-exact. End-to-end serve prefill: +1.1% at 51K on INT8 KV
  (attention wall share of the hybrid-GDN model is only ~10% there and the
  share grows with depth), within noise on the deployed `rk4v4-e8` mode - E8
  staging time is lattice-decode compute, not the removed guards. The biggest
  serve-level beneficiary is the INT8-KV 5090 deployment; porting this to
  ninfer-5090 is queued.
- **Q4/Q5/Q6/W8 dequantization micro-optimizations (73f3d7be, 8f298555, b8ddda48,
  d9d701bc): rejected on measurement (2026-08-19).** All four cherry-pick clean
  and pass the 84-test suite, but on this Linux CUDA 13.1.2 `sm_89` build they
  regress the Q4/Q5 MMA rowsplit path hard. Measured medians at T=1024
  (`ninfer_q5_linear_add_bench --k 17408`, `ninfer_q4_linear_swiglu_bench`,
  suite shapes):

  | State | Q5 down k=17408 | Fused Q4 swiglu |
  |---|---:|---:|
  | pre-wave baseline | 1494.9 us (122.1 TFLOP/s) | 3166.2 us (115.3 TFLOP/s) |
  | + 73f3d7be (Q4/Q5 hoist) | 1631.2 us (-9.1%) | 3436.5 us (-8.5%) |
  | + 8f298555, b8ddda48 | 1618.9 us (flat) | 3394.6 us (flat) |
  | + d9d701bc (shuffle) | 2281.5 us (**-52.6%**) | 3123.1 us (+1.4% net) |

  The suite confirms the pattern on production shapes: `gdn_output_gate` Q5
  -38 to -49%, `draft_head` Q4 -5 to -9%, vision Q4/Q5 projections -13 to -54%,
  Q6/W8 and every small-T SIMT path neutral. The only net winner is the fused
  swiglu at +1.4% (~0.5% end-to-end), and it is inseparable from the losses
  without splitting the shared decode atoms per kernel. Suspected cause of the
  mismatch with their results: their WDDM and MSVC commits indicate a Windows
  toolchain, and the shuffle and `bfe.s32` patterns compile to different SASS
  there. The picks were dropped from this branch after the bisect.
- **GDN / conv1d / 2D-memcpy decode-tail work (52fc4aec, fa629318, b860bd5f,
  fa767237, d520f7bf, cf586d09, 4d79043e): low priority.** Decode on this card
  measures bandwidth-saturated end to end; expected gain is 1-3%.
- **Windows WDDM/D3D12 residency and MSVC flags (a35acf6a, aa8a1c98): not
  applicable.** All `_WIN32`-guarded.
