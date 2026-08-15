# NInfer-4090

NInfer-4090 runs **Qwen3.8-27B** on one 24 GB NVIDIA GeForce RTX 4090. It is an `sm_89` port of
[NInfer-3090](https://github.com/Don-Chad/ninfer-3090), which derives from
[Neroued/ninfer](https://github.com/Neroued/ninfer), a specialized C++20/CUDA inference engine.
The engine loads the official groupwise `.ninfer` artifact, serves OpenAI- and
Anthropic-compatible APIs, and supports paged KV, compatible-prefix reuse, CUDA Graphs, MTP
speculative decoding, reasoning-effort control, and ReplaySSM state transactions.

This fork targets `sm_89` and Linux. Blackwell-only NVFP4/W4A4 execution is unavailable; the
engine uses the same groupwise-int path as the 3090 base. The Windows path and the
Qwen3.6-35B-A3B target are inherited but untested on the RTX 4090.

## Measured results on the RTX 4090

Conditions: single request, greedy decoding, CUDA Graphs on, INT8 KV, `--prefill-chunk 1024`,
official 16.96 GiB Qwen3.8-27B artifact. The code-generation decode row and the prefill rows
are measured from the `ninfer-serve` `/metrics` counters (computed prefill only); the other
decode rows use the `ninfer` CLI.

| Test | Result |
|---|---|
| Decode, code generation, MTP3 | **148.6 tok/s** at 81.0% draft acceptance |
| Decode, bench corpus, MTP3 | 106.5 tok/s at 48.7% acceptance |
| Decode, no speculation | 50.5 tok/s |
| Decode at 128K depth, no speculation | 39.6 tok/s |
| 64K needle-in-a-haystack | exact answer, 1,849 tok/s prefill |
| 128K needle-in-a-haystack | exact answer, 1,561 tok/s prefill |
| Vision, chart reading | 3 of 3 oracle facts, 22 ms vision tower |
| Ops test suite | 78 of 78 runnable tests pass on `sm_89` |

MTP acceptance, and with it the decoded rate, tracks how predictable the output is: structured
code accepts about 81% of draft tokens, the mixed bench corpus about 49%.

For scale: llama.cpp on the same card decodes the Qwen3.8-27B `UD-Q4_K_XL` GGUF at about
46 tok/s in a 144K-context configuration where the MTP buffers do not fit. The upstream engine
on an RTX 5090 measures 172 tok/s on the same code-generation prompts with a 400 W power cap
(the upstream README quotes about 200), so this card lands within 14% of it under MTP.

### Depth sweep against llama.cpp

Both engines were measured on the same card on 2026-08-15. llama.cpp build 10358 ran
`llama bench` on the `UD-Q4_K_XL` GGUF (16.68 GiB) with q8_0 KV cache, flash attention, and
`-ub 1024 -b 4096`, which matches its deployed configuration. NInfer ran the deployed 168K
serve configuration and was measured through the `/metrics` counters. Two caveats: the
artifacts differ by about 2% in size, and `llama bench` is a bare kernel loop while the
NInfer numbers include the full server path.

Marginal rates at depth:

| Depth | llama.cpp pp2048 | llama.cpp tg32 | NInfer decode, no speculation |
|---:|---:|---:|---:|
| 0 | 3,024 tok/s | 45.9 tok/s | 50.5 tok/s |
| 32K | 2,327 | 42.0 | - |
| 64K | 1,866 | 38.6 | - |
| 128K | 1,336 | 33.1 | 39.6 |

Wall time to prefill one full prompt (llama.cpp integrated from the marginal rates, NInfer
measured):

| Prompt | llama.cpp | NInfer |
|---:|---:|---:|
| 32K | 12.5 s (2,630 tok/s) | 15.5 s (2,012 tok/s) |
| 64K | 28.3 s (2,317 tok/s) | 34.4 s (1,849 tok/s) |
| 128K | 70.4 s (1,862 tok/s) | 82.0 s (1,561 tok/s) |

The llama.cpp prefill lead narrows with depth. Server-measured, it prefills a 64K prompt in
28.7 s against 34.4 s (a 20% lead) and a 128K prompt in 71.6 s against 82.0 s (15%); the
server path costs llama.cpp 2-4% over the bare-loop estimates above. Decode inverts this.
NInfer leads by 10% shallow and by 20% at 128K without speculation, and the MTP3 gap grows
with depth:

| Workload | llama.cpp `draft-mtp` | NInfer MTP3 |
|---|---:|---:|
| Code, shallow | 118.8 tok/s at 85.9% acceptance | 148.6 tok/s at 81.0% |
| Prose, 64K depth | 55.5 tok/s at 45.3% | 85.9 tok/s at 44.6% |
| Prose, 128K depth | 42.3 tok/s at 45.4% | 77.1 tok/s at 45.6% |

The llama.cpp MTP rows required a reduced 131,584-token context; the draft buffers push VRAM
to 23.8 of 24 GiB, and the deployed 144K llama.cpp configuration cannot fit them at all.
NInfer serves 172,032 tokens with MTP in the same VRAM. Acceptance matches per content type,
so the decode gap is engine time, not draft quality.

## Quick start (Linux)

Requirements: an RTX 4090, a recent NVIDIA driver, Docker with the NVIDIA Container Toolkit.

Build the image and download the model once:

```bash
docker build --tag ninfer-4090:sm89 .
NINFER_MODEL_DIR="$PWD/models" bash scripts/download-qwen38.sh
```

Then start one of the two profiles. The API is available at `http://127.0.0.1:8080/v1`.

### Text-only, 168K context

```bash
docker run --rm --gpus all --publish 8080:8080 \
  --volume "$PWD/models:/workspace/models:ro" \
  ninfer-4090:sm89 \
  ninfer-serve models/qwen3_8_27b.ninfer \
  --host 0.0.0.0 --port 8080 \
  --max-context 172032 --kv-capacity 172032 \
  --max-concurrency 1 --max-pending-requests 16 \
  --prefill-chunk 1024 --kv-dtype int8 \
  --spec mtp --draft-tokens 3 --lm-head-draft \
  --preserve-thinking
```

### With vision, 96K context

```bash
docker run --rm --gpus all --publish 8080:8080 \
  --volume "$PWD/models:/workspace/models:ro" \
  ninfer-4090:sm89 \
  ninfer-serve models/qwen3_8_27b.ninfer \
  --host 0.0.0.0 --port 8080 \
  --max-context 98304 --kv-capacity 98304 \
  --max-concurrency 1 --max-pending-requests 16 \
  --prefill-chunk 1024 --kv-dtype int8 \
  --spec mtp --draft-tokens 3 --lm-head-draft \
  --vision --preserve-thinking
```

### The tradeoff

Vision and maximum context trade against each other on a 24 GB card:

| Profile | Context | Startup VRAM |
|---|---:|---:|
| Text-only, MTP3 | 172032 (168K) | 23.9 GiB |
| With `--vision`, MTP3 | 98304 (96K) | 23.5 GiB |

Dropping vision buys about 72K more tokens of context at INT8 KV. The text-only ceiling is near
176K: 172032 starts, and 196608 is rejected at startup with a byte-exact deficit. The server
validates memory before it listens, so an oversized context fails fast instead of at request
time.

For a native build, follow the [Linux build guide](docs/rtx-3090-linux.md) with
`CMAKE_CUDA_ARCHITECTURES=89` (the default in this fork). The build requires CUDA 12.8 or newer,
GCC 13, and CMake 3.28 or newer; the Docker image builds with CUDA 13.1.

## What this fork changes

- **`sm_89` retarget.** The CMake architecture pin, the runtime compute-capability check, and the
  NVFP4 stub gate now select `sm_89`. Most SM86 kernel schedules run unmodified on Ada; the
  INT8 attention prefill schedule is retuned (below).
- **Ada-retuned INT8 attention prefill.** The SM120 schedule spills registers on Ada and pays the
  consumer half-rate penalty for f32-accumulate HMMA. Arch-gated for `sm_89`: the full
  128-register budget, eight paired producer warps over `Bc` column halves with one named-barrier
  exchange per key tile, byte-permute V dequantization (bit-identical), and fp16-accumulated PV
  tiles folded into the fp32 running accumulator each tile. The kernel gains 30% at 64K depth
  (109 to 143 TFLOP/s on the `d256-h24-kv4` INT8 append shape); serve prefill gains 5-7% at
  88K-128K. Needle-in-a-haystack retrieval stays exact at both depths and all 84 suite tests
  pass, which bounds the fp16-accumulation numerics change.
- **`/v1/models` reports `context_window`.** Clients without access to a llama.cpp `/props` or a
  vLLM `max_model_len` can size prompts from the models payload.
- **`GET /metrics`.** Prometheus counters under llama.cpp-compatible names
  (`llamacpp:prompt_tokens_total`, `llamacpp:prompt_seconds_total`,
  `llamacpp:tokens_predicted_total`, `llamacpp:tokens_predicted_seconds_total`,
  `llamacpp:requests_processing`, `llamacpp:requests_deferred`), so existing scrapers read this
  server without changes. Prompt tokens count only computed prefill; prefix-cache hits are
  excluded, as in llama.cpp. Additional `ninfer:` series report request totals, prefix-cache
  hits, and MTP draft/acceptance totals.
- **`GET /slots`.** A llama.cpp-shaped slot table built from in-flight requests, for dashboards
  that poll slot state. Entries are HTTP-layer FIFO positions; per-slot cache detail is unknown
  mid-flight and reported as zero.
- **NVFP4-A4 test gating.** The A4 activation tests skip on hardware without FP4 tensor cores
  instead of aborting. The full remaining suite passes on the RTX 4090.

## Known limits on the RTX 4090

- Prefill trails llama.cpp by 16-24% on full 32K-128K prompts under matched conditions (see
  the depth sweep above). The rate is flat across `--prefill-chunk` 1024 to 2688, so the
  chunk size is not the lever. With the attention schedule retuned, the remaining gap sits in
  the custom quantized GEMMs, which run about 10% below cuBLAS on Ada. Decode is where this
  engine leads.
- Keep `--prefill-chunk` at 2688 or below. This fork carries measured `sm_89` cooperative
  residency tables (the former hard abort above chunk 1024 is fixed), and chunks through 2688
  stay on split-K. Larger chunks route to the unsplit schedule, which is marginally less
  accurate at its onset (about 1e-5 relative).
- Concurrency above one request is untested in this fork. The published cohort results in the
  [3090 base](https://github.com/Don-Chad/ninfer-3090) do not transfer directly.
- The limits of the base engine apply: one process, one GPU, one model, bounded FIFO admission,
  no multi-GPU execution, no weight offload.

## Artifact

| Model | Artifact | Size |
|---|---|---:|
| Qwen3.8-27B | [official NInfer groupwise artifact](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) | 16.96 GiB |

The artifact is architecture-independent; the model card's RTX 5090 requirement describes the
upstream engine, not the file. Verify the download against the SHA-256 published on the card.

## Reasoning effort

Qwen3.8-27B has three trained reasoning depths plus an off switch. OpenAI Chat Completions
accepts a top-level `reasoning_effort` field (`low`, `medium`, `xhigh`) and a top-level
`enable_thinking` boolean; hidden reasoning returns separately as `message.reasoning_content`.
The `chat_template_kwargs` request field of llama.cpp is not supported and is rejected. For the
CLI, pass `--reasoning-effort` or `--no-thinking`. Sampling defaults come from the model card and
switch with the thinking mode.

## Serving APIs

OpenAI Chat Completions, OpenAI Responses with streaming and local continuation state, Anthropic
Messages, prompt-rendered function tools with parsed tool calls, compatible-prefix reuse, and
JSONL request logs. See [HTTP serving](docs/serving.md) and [CLI usage](docs/cli.md).

## Upstream and credits

- [Neroued/ninfer](https://github.com/Neroued/ninfer) - the engine, developed for the RTX 5090
  (`sm_120a`).
- [Don-Chad/ninfer-3090](https://github.com/Don-Chad/ninfer-3090) - the SM86 compatibility layer,
  ReplaySSM integration, and Qwen3.8 runtime support this fork builds on. Its
  [v0.6.1 release notes](RELEASE_NOTES_0.6.1.md) describe the inherited state.
- [jram4/ninfer-4090](https://github.com/jram4/ninfer-4090) - an earlier RTX 4090 port of a July
  2026 snapshot. Its Ada dispatch tuning targets a kernel organization that upstream has since
  replaced, so this fork starts from the current 3090 base instead.

## License

Apache License 2.0. See [LICENSE](LICENSE).
