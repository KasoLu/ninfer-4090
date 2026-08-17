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
