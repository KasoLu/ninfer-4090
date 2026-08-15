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
`ninfer` CLI with the official 16.96 GiB Qwen3.8-27B artifact.

| Test | Result |
|---|---|
| Decode, code prompt, MTP3 | **106.5 tok/s** at 48.7% acceptance |
| Decode, no speculation | 50.5 tok/s |
| Decode at 128K depth, no speculation | 39.6 tok/s |
| 64K needle-in-a-haystack | exact answer, 1,794 tok/s prefill |
| 128K needle-in-a-haystack | exact answer, 1,483 tok/s prefill, 20.5 GiB peak |
| Vision, chart reading | 3 of 3 oracle facts, 22 ms vision tower |
| Ops test suite | 78 of 78 runnable tests pass on `sm_89` |

For scale: llama.cpp on the same card decodes the Qwen3.8-27B `UD-Q4_K_XL` GGUF at about
46 tok/s in a 144K-context configuration where the MTP buffers do not fit. The upstream engine
on an RTX 5090 reaches about 200 tok/s.

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
  NVFP4 stub gate now select `sm_89`. The SM86 kernel schedules run unmodified on Ada; they are
  correct but not yet retuned.
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

- Keep `--prefill-chunk` at 1024. Larger chunks abort with
  `cudaErrorCooperativeLaunchTooLarge`: the cooperative grid of the GDN gating projection scales
  with the chunk and exceeds the co-resident block limit on `sm_89`.
- Prefill reaches 1.5-1.8k tok/s and trails llama.cpp (about 2.8k tok/s) on the same card. The
  schedules are inherited from SM86 tuning; retuning them for Ada is the main open work. Decode
  is where this engine leads.
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
