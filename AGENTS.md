# AGENTS.md

本文件是工作区 agent 上下文：描述 NInfer-4090 仓库的代码现状（本仓库为自上游 fork 后建立的独立
仓库。旧仓库 `ninfer-4090` 的后续工作（jinja chat template 移植）已以 patch 形式落入本仓库并提交，见「仓库现状与分支」）。
与协作约定。

## 项目

### 概览

NInfer-4090 是 **从零实现的 C++20/CUDA 单 GPU 推理引擎**，在一张 24 GB RTX 4090（`sm_89`）上运行 **Qwen3.8-27B**（16.96 GiB 官方 groupwise-int `.ninfer` 工件）。

- 谱系：`Neroued/ninfer`（RTX 5090 / `sm_120a` 引擎）→ `Don-Chad/ninfer-3090`（SM86 兼容层、ReplaySSM 集成、Qwen3.8 运行时，v0.6.1）→ 本仓库（`sm_89` 重定向 + Ada 重调 INT8 注意力 prefill + 从 `UDPSendToFailed/ninfer-4090` 移植 E8 格 KV 量化与 1M 可见键包络 + `shantanusingh16` fork 贡献的 `timings` 字段）。
- 能力：paged KV + 兼容前缀复用、CUDA Graphs、MTP 投机解码（代码生成约 148.6 tok/s @ 81% 接受率，llama.cpp 同卡约 46 tok/s）、reasoning-effort 控制、ReplaySSM 状态事务、slot 会话保存/恢复、turn checkpoint 环、`/metrics`（llama.cpp 兼容计数名）/`/slots`/`/v1/models`（含 `context_window`）。
- 默认 KV 模式 `rk4v4-e8`（E8 Conway-Sloane 格 4-bit）在 24 GB 上放下模型完整原生 262,144 token 上下文（余量 1.37 GiB）；INT8 KV 上限约 172,032。
- 对外接口：OpenAI Chat Completions / Responses（流式 + 本地续接状态）、Anthropic Messages（含 thinking 签名）、prompt-rendered function tools、JSONL 请求日志。
- 产品边界：单进程、单 GPU、单模型；启动时固定 1–8 个活跃请求；有界 FIFO 准入、无抢占；无 continuous batching、无多卡、无权重 offload。
- License：Apache 2.0；`VERSION` 为 `0.6.1-rtx4090`（`vcpkg.json` 同步为 `0.6.1`）。

### 支持的模型身份（registry 注册）

| identity | 执行包 | 备注 |
|---|---|---|
| `qwen3.6-27b` / `groupwise-int`、`nvfp4` | `src/targets/qwen3_6_27b` | 本 fork 主目标为 Qwen3.8-27B |
| `qwen3.8-27b` / `groupwise-int`、`nvfp4` | **同样绑定在 `qwen3_6_27b` 包**（`src/targets/registry.cpp:196`、`src/targets/qwen3_6_27b/impl/package.cpp:89,95`、`load/bindings.cpp` 的 `bind_qwen38_nvfp4_text_layers`） | `src/targets` 下没有独立的 qwen3_8 C++ 目录；`tests/targets/qwen3_8_27b/` 只有 Python 测试（converter/inventory/fp8-embedding） |
| `qwen3.6-35b-a3b` / `groupwise-int` | `src/targets/qwen3_6_35b_a3b` | 额外支持 text-only DFlash 后端；在 4090 上未经测试 |

27B 与 35B-A3B 是同一 identity-free `qwen3_6` 家族运行时的**对等编译期 Variant**：家族拥有共享的 `SequencePlan<Variant>`/`RequestPlan<Variant>`/`Program<Variant>` 算法、frontend/output 语义、Text/Vision/投机调度、状态事务、workspace 组合与 CUDA Graph 捕获/重放；各包各自拥有注册身份、binder、model view、维度/存储事实、三个执行叶家族（attn 投影、GDN 投影/控制、post-mixer）与 graph frontier 数据。Program 之间不共享可变状态，家族调度内无运行时目标分支。

**sm_89 上 NVFP4 的限制**：`src/CMakeLists.txt` 无条件从 `ninfer_ops` 剔除所有 `nvfp4/*_w4a4.cu`，注入 `ops/nvfp4_stubs_sm89.cpp`（所有 W4A4 入口 `[[noreturn]] reject_nvfp4_a4()`，报错 "NVFP4 A4 execution requires a Blackwell (sm_120a) GPU; this build targets sm_89"）。即 Blackwell-only 的 NVFP4 W4A4 张量核执行不可用，A4 激活测试在无 FP4 卡的硬件上跳过而非中止。

### 目录结构（含 CMake 目标映射）

```
include/ninfer/            公共 C++ 接口：engine.h（Engine/PreparedPrompt/GenerationHandle，PIMPL）、
                           types.h；ops/*.h 为库内语义 Op 契约。不对外安装 SDK。
src/core  → ninfer_core    设备原语：arena、device、tensor、layout、dtype、decode_graph（CUDA Graph RAII）、
                           paged_kv_cache.cpp（~40KB，核心）、cyclic_kv_cache、host_kv_arena、
                           linear_attention_state（GDN）、gdn_replay_records、host_worker_pool、nvtx.h；
                           以及 runtime/engine/{admission_policy,context_cost,context_cost_defaults,kv_capacity,public_types}.cpp
                           链接 cudart + nvtx3
src/artifact → ninfer_artifact   通用 .ninfer 框架：reader、storage_layouts、binder、materializer、typed_binding
src/ops   → ninfer_ops     语义封闭 Op 全集，CMake 源清单显式列出（禁止 glob）：
                           launcher/*.cu（host 启动器：rmsnorm、rope、sampling、causal_conv1d、mtp_pack/round、
                             speculative_round、position、prepare_masked_block、vision_pos_embed 等 30+ 个）
                           softmax_attention/（dense: causal_cache/packed/context，含 fp8 变体；sliding_window）
                           kv_cache/append
                           linear_attention/gated_delta_net/（replay、recurrent、chunked: prepare_wy_wu/state_passing/output）
                           按权重量化画像分目录的投影族：attn_input_proj {bf16,fp8,nvfp4,q4_q5,w8}、
                             gdn_input_proj {fp8,nvfp4,q4_q5,w8 + conv}、gdn_gating_proj/bf16、
                             linear {bf16,fp8,nvfp4,q4,q5,q6,w8}、linear_add、linear_swiglu、linear_pair/w8
                           sparse_moe {decode,prefill,small_t}
                           wrapper/*.cpp（各 Op 的公共入口）
                           ops/nvfp4_stubs_sm89.cpp（无条件编入 ninfer_ops）
src/text  → ninfer_text    unicode.cpp + third_party/utf8proc
src/media/decode → ninfer_media_decode   FFMPEG 解码（消费已拥有的字节）
src/product/       media_acquire（CURL 获取，产品专用，不链接进 target）、prompt_input、load_progress
src/targets/       registry.cpp + runtime/contract/sampling.cpp + runtime/engine/engine.cpp 组成 ninfer_engine
                     （公共 Engine PIMPL）；子目录 qwen3_6（家族运行时：frontend/runtime/state/vision + export/ninfer/targets）、
                     qwen3_6_27b（impl/load）、qwen3_6_35b_a3b（impl/load）
src/serve → ninfer_serve   HTTP 协议/传输（third_party/cpp-httplib）：http_server、openai_chat_*、
                           openai_responses_*（含 state/store）、anthropic_messages_*（含 thinking_signature）、
                           generation_service（lane/准入）、serve_metrics、serve_options、request_log（JSONL）、
                           slot_files（会话保存/恢复）、translate、console_log
apps/                  ninfer（CLI：文本/历史/图像视频输入、采样、MTP）、
                           ninfer-serve（HTTP 服务器）、ninfer-perplexity
tests/                 CTest 套件：ops/（每 Op 一套资格测试 + 独立 oracle）、artifact/（Python 容器/布局/量化）、
                           targets/{qwen3_6, qwen3_6_27b, qwen3_6_35b_a3b, qwen3_8_27b}、
                           协议测试（openai/anthropic schema、tool_call_parser、request_log、slot_files、
                           admission_policy、state_store、public_api、serve_metrics、serve_options）
bench/                 ninfer_bench（走完整公共 Engine 路线，pp/tg/pp+tg 矩阵，语料 bench/fixtures/bench_corpus.ids）、
                           bench/ops/*.cu（~29 个算子微基准）、fixtures/ttft
tools/                 维护者工作流：convert/<target>（BF16 检查点 → .ninfer）、reference/<target>（独立 Python 参考实现）、
                           parity、artifact/inspect.py、bench（Python 矩阵）、bench/ttft、smoke（serve_contract.py、
                           serve_thinking_preservation.py）、test_kv（E8 codec 微基准）、perplexity、freq_corpus
eval/                  独立 Python 评测协调器（backends base/mock/registry；eval/configs/*.yaml；
                           自带 venv：PYTHONPATH=eval eval/.venv/bin/python -m unittest discover -s eval/tests）
docs/                  用户指南（cli.md、serving.md、performance.md、perplexity.md、rtx-4090-linux.md——本 fork 的
                           构建指南也是它，传 CMAKE_CUDA_ARCHITECTURES=89）、llamacpp-comparison.md、
                           udp-fork-comparison.md、turn-checkpoint-ring.md；
                           maintainer/ 为权威参考：engine-architecture.md（唯一顶层引擎架构文档）、
                           resource-scheduling-and-context-cache.md、paged-kv-cache.md、op-development.md、
                           replayssm-gdn.md、linear-benchmark.md、artifact-container.md、tensor-formats.md、
                           storage-layouts.md、各目标 artifact/model 文档
model-cards/           5 个已发布工件的模型卡（Qwen3.6-27B、Qwen3.6-27B-nvfp4、Qwen3.6-35B-A3B、Qwen3.8-27B、Qwen3.8-27B-nvfp4）
scripts/               下载（download-qwen38.sh/.bat → $NINFER_MODEL_DIR/qwen3_8_27b.ninfer，HF 来源）、
                           运行配置（run-qwen38-c1/c8/vision .sh/.bat）、v0.4.0–v0.6.0 发布打包、check-linux-scripts.sh
third_party/           cpp-httplib、nlohmann、utf8proc、minja（Jinja 模板引擎，随 jinja chat template 引入）
build/                 构建目录（git 忽略；docker 内全新 configure，Ninja 生成器 + sm_89）
dist/                  Windows 发布包输出目录（内容被 Git 忽略）
```

### 构建与运行

<<<<<<< HEAD
- directly contributes to the requested deliverable;
- is necessary to preserve an applicable product, semantic, or external contract;
- resolves uncertainty that could materially change the result; or
- checks a realistic regression introduced by the change.

An architectural redesign, cross-cutting refactor, or replacement of an existing path is in scope
when it is necessary to deliver the strongest solution for the requested outcome. Do not use scope
control as a reason to ship an inferior patch. Do not expand into unrelated audits, cleanup,
hardening, compatibility work, benchmark campaigns, or documentation projects. General engineering
preferences, possible future scenarios, and concerns outside the declared product model do not
create requirements by themselves.

Handle incidental findings proportionally:

- address them when they block the requested outcome or make it materially incorrect;
- include them when they are inseparable from a coherent implementation;
- otherwise leave them unchanged and mention them only when they are useful to the user.

For analysis, review, or design work, the requested explanation or design artifact is the
deliverable; experiments and code inspection serve only to resolve material questions. For
implementation work, implement the selected design completely across its affected boundaries,
remove the superseded project-owned path, and validate its supported observable behavior. For
diagnosis, establish the cause and supporting evidence without turning the task into an unrequested
fix or redesign.

## Evidence, provenance, and completion

Select evidence from the claim or decision it supports. The availability of a tool, test suite,
artifact, or profiler does not make its use necessary. Prefer representative evidence over
exhaustive evidence, and do not repeat an experiment unless the previous result is invalid or
inconclusive, or the new result could change a live decision.

Verification must match the semantic contract: use exact comparison for exact formats and
transformations, and numerical or behavioral criteria for floating-point and probabilistic work.
Do not substitute final-output plausibility for verification of an operator or state transition.

Record only the provenance needed to interpret a material result. By default, this is the relevant
target, hardware/toolchain, workload or command, and summarized outcome. Fixed hashes, clean
worktrees, full command transcripts, raw profiler inventories, byte-identical regeneration, and
exact probabilistic outputs are not validity requirements unless a concrete contract or the user
requires them.

Stop when:

- the requested deliverable exists;
- applicable contracts are satisfied;
- material claims have sufficient evidence;
- relevant checks pass, or their limitations are stated clearly; and
- no known in-scope issue prevents the result from being used.

Do not continue merely to eliminate all uncertainty, collect more metrics, complete a process loop,
improve descriptive provenance, investigate unrelated observations, or make working notes
exhaustive. The final result should lead with the deliverable, key decisions, relevant verification,
and material limitations. Raw logs, experiment diaries, exhaustive command histories, hashes, and
intermediate artifacts are excluded unless requested or themselves the deliverable.

## Current product contract

NInfer is a from-scratch C++/CUDA inference engine for maximum single-GPU inference performance on
a small set of explicitly registered checkpoint artifacts. The supported identities are
`qwen3.6-27b/groupwise-int`, `qwen3.6-27b/nvfp4`, `qwen3.8-27b/groupwise-int`,
`qwen3.8-27b/nvfp4`, and `qwen3.6-35b-a3b/groupwise-int`. The current implementation is compiled
for `sm_120a` and tuned and measured on NVIDIA GeForce RTX 5090. All identities execute Text,
image/video Vision, MTP, prefix reuse, CLI, OpenAI/Anthropic serving, and measurement through the
same public `.ninfer` Engine route; the 35B-A3B target additionally supports text-only DFlash.

The current workload is one GPU and one resident model instance with a startup-fixed one to eight
active requests. The Engine forms one compact decode batch at every round boundary and uses bounded
FIFO ingress with no request preemption. Large-scale or preemptive continuous batching, priority/QoS
scheduling, additional checkpoint targets, and retargeting the implementation to another execution
platform are outside the current product. This is a local, single-owner project. Registered models,
generated artifacts, and the local workflow are trusted.
Requirements derived from a different workload, trust model, or deployment model are out of scope
until that product contract is explicitly changed.

The 27B and 35B-A3B execution packages are peer compile-time Variants of one identity-free Qwen3.6
family runtime. The family owns the shared `SequencePlan<Variant>`, `RequestPlan<Variant>`, and
`Program<Variant>` algorithms; frontend and output semantics; Text/Vision/speculative schedules;
state transactions; workspace composition; and CUDA Graph capture/replay mechanics. Each package
separately owns its registered artifact identities and bindings, immutable model view,
dimensions/storage facts, three closed execution-leaf families, graph frontier data, and Program
instance bytes. No mutable state or device allocation is shared between Programs, neither package
is defined as a delta from the other, and there is no runtime family selection or target-dependent
branch inside family scheduling. All artifacts embed the same six frontend resources, and a
prepared prompt carries no exact-target tag.

## Engineering priorities

Prioritize functional correctness, architectural quality, clear ownership, direct code, and maximum
requested inference performance. Change size, implementation difficulty, short-term simplicity,
and backward compatibility for project-owned contracts are not quality criteria. Low maintenance
cost may distinguish otherwise equivalent designs, but it never justifies worse architecture or
performance. Generality, defensive hardening, formal completeness, broad compatibility, and test
coverage are not goals by themselves.

Prefer explicit target-specific implementation over framework-like abstraction. Do not add generic
model graphs, family base classes, plugin discovery, string-driven execution, hidden device
allocation, runtime weight repacking, or placeholders for hypothetical models or hardware unless an
explicitly changed product contract requires them.

## Sources of truth

Read only current authorities relevant to a live decision in the task. The following list is a
routing map, not a mandatory reading list:

- `README.md` and executable `--help`: delivered capabilities and exact commands;
- `docs/README.md`: public documentation map;
- `docs/cli.md`: CLI input, output, sampling, MTP, and runtime options;
- `docs/serving.md`: OpenAI/Anthropic HTTP behavior;
- `docs/performance.md`: published performance methodology and results;
- `docs/maintainer/engine-architecture.md`: Gateway/Frontend/Engine/Runtime boundaries, execution
  ownership, request/response/continuation lifecycles, admission, scheduling, output transactions,
  batched execution, and CUDA Graph semantics;
- `docs/maintainer/resource-scheduling-and-context-cache.md`: resource selection and accounting,
  continuation/checkpoint ownership, materialization transactions, and Device/Host replica policy;
- `docs/maintainer/paged-kv-cache.md`: shared KV capacity, page ownership, retention, physical
  layouts, and paged consumer contracts;
- `docs/maintainer/artifact-container.md`, `storage-layouts.md`, and `tensor-formats.md`:
  generic `.ninfer` contracts;
- `docs/maintainer/qwen3.6-27b-artifact.md`, `qwen3.8-27b-artifact.md`, and
  `qwen3.6-35b-a3b-artifact.md`: exact target inventories, conversion, and binding;
- `docs/maintainer/qwen3.6-27b-model.md` and `qwen3.6-35b-a3b-model.md`: exact model mathematics,
  dimensions, and state semantics;
- `docs/maintainer/op-development.md`: Op admission, contracts, implementation ownership,
  qualification, and performance evidence rules;
- `include/ninfer/engine.h` and `include/ninfer/types.h`: in-tree C++ product interface.

Do not survey unrelated references for completeness. Read additional documents only when they
govern a live decision in the current task.

## Product and ownership boundaries

These boundaries govern ordinary implementation work. An explicit architecture task may revise
them, but must update the corresponding active authorities and affected implementation together.

- `.ninfer` is the only C++ product artifact. Do not add extension detection, compatibility shims,
  or a second product lane.
- `include/ninfer/engine.h` and `include/ninfer/types.h` are the opaque Engine interface used by
  in-tree applications and owning host values. NInfer does not currently install or export a C++
  SDK. `include/ninfer/ops/` contains repository-internal semantic Op contracts.
- `src/core` owns device primitives, tensors/views, checked layouts, arenas, graph RAII, physical
  KV-cache containers, and raw transfer mechanisms.
- `src/artifact` owns generic `.ninfer` framing, descriptors, binding primitives, and
  materialization. It has no checkpoint execution semantics.
- `src/ops` owns every semantically closed Op implementation, including fused, fixed-shape, and
  device-specialized paths. Op ownership follows the mathematical or state-transition contract,
  not its first model caller or demonstrated cross-target reuse.
- `src/targets/qwen3_6` owns only the Qwen3.6-family invariants shared by the 27B and 35B-A3B
  targets: tokenizer/template and output semantics, media preprocessing and MRoPE prompt
  construction, owning prepared-prompt/output-session types, semantic weight-view schemas, passive
  Vision definitions, and the fixed
  planning/Program/Text/Vision/speculative/state/workspace/CUDA-Graph algorithms. It has no target
  identity, registry entry, artifact binder, target leaf
  implementation, or storage for a live Program instance.
- `src/targets/<package>` owns its registered checkpoint identities, storage profiles, binder,
  `LoadedModel`, configuration, populated family model-view values and private leaf payloads,
  diagnostics, graph frontier values, and exactly three execution-leaf families: attention
  projection, GDN projection/control, and post-mixer. It aliases and instantiates the family
  runtime types; it does not own a copied Program, Text/Vision/speculative schedule, workspace
  composition, state transaction, or graph-capture algorithm. Leaf Ops remain implemented under
  `src/ops`.
- `src/runtime` owns common contracts, generated-token transaction/publication policy, and the
  public Engine PIMPL. It does not own model mathematics or target state.
- `src/media/decode` consumes already-owned bytes. URL/path/data acquisition belongs to
  `src/product/media_acquire`, CLI, or serving and is not linked into a target.
- `src/product/prompt_input` owns the shared product-side JSON/message-to-owning-input adapter.
- `src/serve` owns protocol translation and transport. CLI, server, and benchmark call only the
  public Engine for inference.
- `tools/convert/<target>` owns target-private artifact inventories, source recipes, conversion,
  and converter-side payload verification. NInfer maintains no Python model-inference route.

## Compatibility and document lifecycle

Project-owned C++ APIs, CLIs, Python tools, fixtures, reports, formats, and active documentation do
not preserve backward compatibility. When a task replaces project-owned behavior, remove the
obsolete aliases, fallbacks, transition branches, and tests in the affected contract instead of
maintaining two paths. Do not turn that rule into unrelated repository-wide cleanup.

The advertised OpenAI and Anthropic protocol surfaces are real external contracts. A change to
their behavior must update the affected schema tests and serving documentation together.

Integrate stable requirements into the existing active reference. Use a temporary dated plan only
when active work genuinely needs one; a plan is not a substitute for the requested deliverable.
Remove completed or abandoned plans instead of retaining a historical documentation tree. Do not
create parallel `final`, `v2`, or `new-design` references.

## Numerical correctness

When a task changes numerical behavior or makes a numerical claim, identify the mathematical
oracle, represented public inputs, explicit semantic cast/quantization/state boundaries, output
criterion, and real model shapes relevant to that claim. If a route's private precision or
reduction profile matters to the evidence, describe it as an implementation profile rather than a
semantic requirement. Apply exact, tolerance-based, or behavioral comparison according to the
actual semantic contract.

Every floating-point Op has one independent naive FP32/FP64 mathematical oracle; exact transforms
and codecs have one independent exact oracle. The oracle evaluates the complete logical formula
from the represented public inputs and, for packed weights, decodes the signed code with the exact
stored scale. It does not copy a production kernel's staging casts, reduction tree, workspace dtype,
or another implementation's output.

The oracle does not prescribe a production arithmetic path. Unless an intermediate value is an
observable Op output, explicit Cast/quantize/dequantize result, registered codec value, or specified
persistent state, kernels may choose the natural intermediate precision, instruction operands,
reduction association, workspace representation, and kernel decomposition for their route. A fused
kernel is neither required to reproduce an unfused BF16 materialization nor forbidden from using a
lower-precision intermediate when that is the natural qualified implementation. Every production
route is checked directly against the same oracle with a criterion appropriate to its output and
implementation profile; pairwise implementation parity is supplementary evidence only.

Where relevant to the changed behavior, account for numeric-format decode, BF16 fusion order, FP32
GDN state, BF16/INT8 KV, MTP accept/commit state, arena lifetime, and CUDA Graph address stability.
This is a risk map, not a checklist for every numerical task.

## Performance work

Define a performance claim at the level where it matters: operator, schedule, request phase, or
end-to-end inference. Measure that level directly when practical. An isolated microbenchmark can
support an operator-level claim but does not establish an end-to-end improvement.

Use whole-inference profiling when end-to-end attribution remains unresolved. Use kernel profiling
only after a relevant kernel has been identified and a kernel-level answer could materially change
the current design or implementation decision. Do not collect additional profiling data once the
relevant alternatives can be distinguished and the requested claim has adequate support.

Retain concise context sufficient to interpret a reported result: relevant hardware/toolchain,
artifact identity at the descriptive level, workload or command, and summarized measurements. Raw
reports and fixed repository or artifact hashes are not required by default.

## Tests and verification

Add or retain a test only when it protects supported observable behavior or a realistic regression:
numerical kernel/model correctness, `.ninfer` framing/binding, external schema/report behavior, a
small real integration route, GPU lifetime, or a reproduced bug. Do not add tests for coverage,
private file/class shape, getters/constructors, deleted compatibility, source-string scans,
hypothetical failures, or test ceremony.

Run a focused set of checks sufficient to support the changed behavior and its material claims.
The following are typical choices, not a cumulative checklist:

| Change | Relevant evidence |
|---|---|
| documentation | affected active-link/stale-reference review and `git diff --check` |
| C++ runtime/API | affected explicit targets and meaningful tests |
| Python tooling | `py_compile` and affected Python tests |
| `.ninfer` reader/converter/binder | affected contract tests and a real artifact when semantics require it |
| CUDA math | independent numerical oracle at relevant shapes |
| memory/lifetime | the affected execution; sanitizer only for a concrete lifetime risk |
| performance | measurement at the claimed scope; attribution tools only when needed |
| serving | affected OpenAI/Anthropic schema tests and observable request/stream behavior |

Do not replace weak verification with low-value tests. State clearly when a relevant check could not
run and why.

## Local environment

Use unrestricted build-tool parallelism for repository compilation. Invoke CMake builds as
`cmake --build <build-dir> -j`; do not supply a numeric job limit such as `-j2` or `-j32`.

These are conventional project resources, not a checklist of resources every task must use:

| Purpose | Path |
|---|---|
| repository | current checkout |
| Python 3.11 | `python3` in the selected maintainer environment |
| BF16 source checkpoint | explicit local checkpoint directory |
| product artifact | `out/qwen3_6_27b.ninfer` |
| conversion report | `out/qwen3_6_27b.ninfer.conversion.json` |
| normal build | `build/` |
| profiler output | `profiles/ncu/`, `profiles/nsys/`, `profiles/bench/` |
| hardware/toolchain | RTX 5090, `sm_120a`, CUDA 13.1 |

Use the selected Python 3.11 interpreter explicitly. Do not install or upgrade dependencies unless
the task requires it. Never select an artifact by glob, modification time, or an unqualified
“latest” name. Large artifacts, source checkpoints, and profiler outputs are local prerequisites;
do not download or regenerate them unless that work is in scope.
=======
- 工具链：CUDA ≥ 12.8（Docker 镜像用 `nvidia/cuda:13.1.2` + Ubuntu 24.04）、CMake ≥ 3.28、C++20、Ninja 推荐。
- **`CMAKE_CUDA_ARCHITECTURES` 硬钉死 89**（根 `CMakeLists.txt` 校验，传其他值直接 `FATAL_ERROR`）。
- vcpkg manifest 依赖：`curl`、`ffmpeg`（zlib feature）、`pkgconf`；Linux 下 FFMPEG/libcurl 走 pkg-config。
- 选项：`NINFER_BUILD_APPS=ON`（默认）、`BUILD_TESTING=OFF`、`NINFER_BUILD_BENCHMARKS=OFF`；后两者派生出 `NINFER_BUILD_MEDIA_ACQUIRE` / `NINFER_BUILD_PROMPT_INPUT` / `NINFER_BUILD_SERVE`。
- Windows 特殊处理：静态 cudart（`CUDA::cudart_static`）、`/NODEFAULTLIB:LIBCMT`（CUDA 静态运行时要求 LIBCMT，与 vcpkg 依赖保持单一 CRT）、`NOMINMAX WIN32_LEAN_AND_MEAN`、`/Zc:preprocessor`。Windows 路径继承自 3090 基线，在 4090 上未测试。
- Ninja 下 CUDA 链接 job pool = 1。
- Docker 关键点：运行时镜像必须 `rm -rf /usr/local/cuda*/compat`——forward-compat 库在 GeForce 卡上触发 `cudaErrorCompatNotSupportedOnDevice`。
- 常用命令：
>>>>>>> 954a19a8 (docs: update AGENTS.md)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build --parallel          # 不限并行度，不要写 -j2 之类
ctest --test-dir build --output-on-failure
```

- 4090 服务配置（README 默认 profile）：

```bash
ninfer-serve models/qwen3_8_27b.ninfer \
  --host 0.0.0.0 --port 1234 \
  --max-context 262144 --kv-capacity 262144 \
  --max-concurrency 1 --max-pending-requests 16 \
  --pending-timeout-ms 600000 \
  --prefill-chunk 1024 --kv-dtype rk4v4-e8 \
  --spec mtp --draft-tokens 3 --lm-head-draft --preserve-thinking
```

  `--max-concurrency 2` 在 4090 上有实测收益（每 lane 约 390 MiB，KV 页池共享，双会话约 1.5× 聚合解码吞吐，prefill 仍跨 lane 串行）。
  **`--prefill-chunk` 保持 ≤ 2688**：更大值走 unsplit 调度，起效点精度约 1e-5 相对劣化。
  vision scratchpad 默认 8192 tokens（`--vision-max-tokens`，旧值硬编码 32768）；超限请求以 `media_budget_exceeded` 拒绝。

### 测试与基准现状

- README 记录：`sm_89` 上 ops 测试套件 78/78 runnable 通过（INT8 注意力重调章节提到全套 84 个通过）；根目录 `test-cpu-remote.log` 是远端 CPU-only 的 ctest 日志（87 项，GPU 项跳过）。
- 需要真实工件的 C++ 集成测试通过环境变量 opt-in，未设置时 CTest 标记 skipped：
  `NINFER_QWEN3_6_27B_WEIGHTS`（prefix/score real test）、`NINFER_QWEN3_6_35B_A3B_WEIGHTS`（real/dflash real）。
- Python 套件：`python3 -m pytest tests/artifact tests/targets/qwen3_6_27b tests/targets/qwen3_6_35b_a3b tests/test_bench_matrix.py tests/test_serve_corpus.py`。
- 算子错误记录：`NINFER_OP_REPORT_STATS=1` 让每次比较输出 `OP_ERROR_STATS`（标签、实际误差、限值、比值），只改报告不改判定。
- 服务冒烟（需常驻服务器）：`python3 -m tools.smoke.serve_contract --base-url ... --model qwen3.6-27b`、`python3 tools/smoke/serve_thinking_preservation.py --artifact ... --backend mtp|dflash`；刻意不进 CTest。

### 仓库现状与分支（截至 `914e0500`）

- 本仓库自上游 fork 后建立为**独立仓库**：`origin = https://github.com/KasoLu/ninfer-4090.git`
  （默认分支 `rtx4090-port`）；本地仅 `chat-template`（当前分支，领先 `rtx4090-port` jinja 移植与文档更新提交）和
  `rtx4090-port` 两个分支。旧仓库（`C:\Workspace\codes\ninfer-4090`，
  remotes 为 `sergiuszm/ninfer-4090` / `UDPSendToFailed/ninfer-4090`）的 `master`/`release/*`/
  `recon/*` 分支与 `k4090` 远端均已不存在。
- **旧仓库的后续工作未随 fork 带入**：其 `chat-template` 分支上的 PR #42 两笔提交（`5dac17ec`
  移植、`f81d4d32` 评论修复）在本仓库不存在（`git cat-file` 验证）。全部内容以 patch 形式重新落回本仓库
  并已提交：功能提交 `b259ac0c`（jinja chat template，含 6 处本地评论修复）、`98785557`（linear_swiglu 测试无 GPU skip 码修复）、本文件更新（明细见下）。
  提交历史与旧仓库 `rtx4090-port` 一致（见下条）。
- 近期提交（新→旧）：`test(ops): propagate the no-GPU skip code in linear_swiglu tests`、`feat(frontend): introduce a self-contained Jinja chat template`、`docs: correct the logging-replatform risk note`、`docs: record the /health port and the admission review`、`fix(serve): report the engine's real state on /health`、`docs: record the 2026-09-01 fork and upstream sweep`、`test(ops): verify i8 keys in the full-cache append`、`refactor(serve): retire --turn-checkpoints`、`test(serve): guard the effort alias and models payload`、`test: run session-persistence E2E on Qwen3.8 artifact`、`feat(runtime): persist checkpoints in slot snapshots`、`fix(ops): accept packed and E8 KV modes in append`、`feat(runtime): port session slots to catalog`、`Merge upstream neroued/master into rtx4090-port`、`docs: record inbound ports from downstream forks`、`fix(docker): remove CUDA forward-compat libs so GeForce cards can run`、`feat(serve): accept enable_thinking and reasoning_effort under chat_template_kwargs`、`fix(runtime): preserve reusable checkpoints under pressure`、`bench(ttft): cover shared prefix value scheduling`、`feat(runtime): add value-aware shared prefix scheduling`、`test(nvfp4): cover a width the fused SwiGLU route newly reaches`、`feat(profiling): instrument engine execution with nvtx`。
- **旧仓库 patch 内容**（源自旧仓库，已在本仓库逐一验证落地并提交）：
  - **上游 PR #42 移植**（Neroued/ninfer，Jinja chat template 支持，作者 Doelfke，head `3f8154ea`，
    closed 未合并；净 diff 已覆盖 #41/#43 全部，无需额外 cherry-pick）：vendored `third_party/minja/`
    （LICENSE + minja.hpp 132 KB/3069 行，与 PR head 逐字节一致 + 下方本地修复）、
    `src/targets/qwen3_6/impl/frontend/chat_template.{h,cpp}`（JinjaTemplate/CompiledChatTemplate、
    render/derive_think_parts/prompt_starts_in_reasoning、sha256 语义识别 `resolve()`）、
    `frontend.cpp` 接线与 export 头文件、`include/ninfer/types.h`（`EngineOptions::chat_template_path`）、
    `apps/cli/options.{h,cpp}` 与 `src/serve/serve_options.{h,cpp}`（chat-template 选项）、
    `src/targets/qwen3_6_27b/impl/package.cpp` 与 `qwen3_6_35b_a3b/impl/package.cpp` 各 1 行、
    `apps/cli/main.cpp`、`docs/cli.md`（## Chat templates）与 `docs/serving.md`，AGENTS.md 本体亦随 patch 更新。
  - **PR #42 评论修复**（8 条去重问题；作者已修 2 条随 diff 带入，其余 6 条为本地修复，
    均带 `local fix (upstream PR #42 review)` / `P1 fix` / `P2 fix` 注释可 grep 定位）：
    - B/C/D/F 在 `third_party/minja/minja.hpp`（与上游分叉）：`contains(Value)` 去掉
      `item.to_bool() &&` 短路（`0 in [0,1]` 的 Jinja `in` 语义）；`ForNode::do_render` 递归 `loop()`
      改用 `iter` 源而非恒遍历 `iterable_value`（仅递归 for/loop 场景触发）；`Value::pop()` 支持
      负索引；`joiner` builtin 删尾部不可达死代码。
    - G（P1，与自定义无思考前导模板直接相关）：`starts_in_reasoning` 不再按 `enable_thinking`
      选项推导（旧逻辑会把整个生成归入 reasoning、content 恒空），改由渲染后 prompt 推导——
      `chat_template.cpp:564 prompt_starts_in_reasoning()` 对 <think>、<|think|>、<thinking>
      三对标记（开/闭标签对 <think>/</think>、<|think|>/</think>、<thinking>/</thinking>）
      取最后一次出现，末标记为开标签才返回 true（三对标记互不为子串，无歧义）；
      `frontend.cpp:1382/1424/1454` 接线（`reasoning_prologue`）。默认模板行为等价（既有
      `test_frontend.cpp` 断言即回归护栏）；**残留缺口**：media 路径无渲染文本可查，仍 fallback
      到 `options.enable_thinking`（注释已写明）。
    - H（P2）：`supports_reasoning_effort_` 能力检测改为 `strip_jinja_comments(source)`
      （`chat_template.cpp:484`：删 `{# #}` 块注释（未终止则截断）+ `{{ }}`/`{% %}` 块内
      行尾 `#` 注释）后再子串查找，注释/字面量中的 `reasoning_effort` 不再误判。注意 minja
      lexer 本身只支持 `{# #}` 块注释，strip 按超集处理，测试只用 `{# #}` 用例。
  - **测试**：`tests/targets/qwen3_6/test_jinja_chat_template.cpp`（纯 CPU，docker 容器可跑）、
    `tests/targets/qwen3_6/test_frontend.cpp`（`test_custom_template_reasoning_channel()` 需
    `NINFER_QWEN3_6_27B_HF_DIR` 官方 tokenizer，无则整文件 skip，容器内只能验证编译）、
    `tests/test_cli_options.cpp`、`tests/test_serve_options.cpp`、`tests/CMakeLists.txt`
    （jinja 测试链 `ninfer_engine ninfer_core`）。
  - **未跟踪存档**（约定保持未跟踪、不提交）：`pr42.diff`（官方 diff 存档）、`pr_comments.txt`
    （#41/#42/#43 评论全文）、`HANDOFF_pr42_fixes.md`（交接文档；其中 remote 名、docker 挂载路径、
    「已提交/工作树干净」等描述均指旧仓库，引用时注意过期）、`tests/targets/qwen3_6/test_jinja_chat_template.cpp`、
    `third_party/minja/`。
- **20w 上下文 OOM 诊断结论**（4090 一键诊断实测定案，2026-09-03，与 PR #42 无关）：
  - 现象：context cache（prefix reuse）ON 时 serve 启动即死；`--no-prefix-reuse` 或砍设备侧
    （`--device-state-slots 0 --no-cuda-graph`）、降上下文（60k）、换 KV 模式（rk4v4-e8）全部无效，
    因为申请量与这些轴全无关。
  - **真凶 = pinned 主机内存**：错误行是 `cudaMallocHost failed: cudaErrorMemoryAllocation: out of
    memory`（失败时 nvidia-smi 仅 701 MiB，根本不是显存 OOM）。context cache 默认
    `kDefaultHostKvCapacityBytes=8GiB` + `kDefaultHostStateSlots=8×~147MiB` → 启动固定申请
    ~9.2 GiB pinned RAM；32GB WSL2 机器读完 18.2GB 模型文件后可支配内存不足 → 权重加载完 ~1.5s
    进程死。`--no-prefix-reuse` 只是顺带把这两项 host 容量清零（engine.cpp 规范化分支），从来不是治 GPU。
  - **修复**：host 两层改为默认 0、`--host-state-slots`/`--host-kv-mib` 显式 opt-in
    （types.h 默认值清零）；arena.cu 的 pinned 失败消息带请求字节数，低内存/WSL 机器可自解释。
    reuse ON 的设备端增量 ≈ +1 个 StateImage 缓存槽（~147MiB），200k rk8v4 预计余量 ~173MiB
    （对照 no-reuse 实测 slack=319.77MiB），100k rk4v4-e8 余量更大。
  - 4090 生产建议：去掉 `--no-prefix-reuse`，直接吃新默认（reuse ON + host=0）；确需 host 层再按
    机器空闲内存显式给 `--host-kv-mib`/`--host-state-slots`。
  - 旗标语义备忘（`src/serve/serve_options.cpp:241-271`）：`--device-state-slots` = 设备端额外
    StateImage 槽（总容量 C 活跃 + N）；`--host-state-slots`/`--host-kv-mib` = pinned CPU 内存
    （不占 VRAM）；`--max-private-continuations` = 请求结束后保留的会话延续链数；`--max-shared-prefixes`
    = 多请求共享前缀目录（单 agent 无用）；`--max-long-anchors-per-continuation` = 每链内标记点锚，
    仅在客户端显式声明 `cache_boundary_after`（kind=PrivateLongAnchor）时创建，L=0 时 marker 一律
    忽略（`program_impl.h:7610`）。
  - CLI 路径不受影响：`apps/cli/main.cpp:311` 对 CLI 硬禁用 context cache。- **已知不一致 / 陈旧点**（动手前先留意）：
  - 根目录 `test-cpu-remote.log` 是临时测试日志；`misc/` 只有 `__pycache__`。

## 约定

### 工作流与代码风格

- 改动前先读相关模块及 `docs/README.md` 路由表中的权威文档；`engine-architecture.md` 是唯一顶层引擎架构参考。
- 代码风格跟随 `.clang-format`（LLVM 基线：100 列、4 空格缩进、左对齐指针/引用、短 if/loop/block/function 可单行、include Regroup、`SeparateDefinitionBlocks`、`IndentPPDirectives: AfterHash`）；`.codex/hooks/clang_format.py` 会自动格式化。
- 提交仅在用户要求时创建；Conventional Commit 风格 + 模块 scope，类型与仓库历史一致（`feat`/`fix`/`perf`/`bench`/`test`/`build`/`refactor`/`docs`/`chore`），如 `fix(serve): report the engine's real state on /health`。
- **项目自有** C++ API、CLI、Python 工具、fixture、报告格式不做向后兼容：替换行为时删掉旧别名/回退/过渡分支与对应测试，不留双路径（但不要把这条规则扩大成无关的仓库级清理）。
- **OpenAI / Anthropic 协议面是真实外部契约**：改动必须同步 schema 测试与 `docs/serving.md`。
- 文档生命周期：把稳定要求并入现有活动参考；不建并行 `final`/`v2`/`new-design` 文档；CLI 选项拼写与默认值以可执行文件 `--help` 为准。

### 数值正确性

- 每个浮点 Op 有且仅有一个独立 naive FP32/FP64 数学 oracle：从表示的公共输入评估**完整逻辑公式**；packed 权重用精确存储的 code+scale 解码。oracle 不复制生产 kernel 的 staging 转换、归约树、workspace 精度或另一实现的输出。
- oracle 不规定生产算术路径：除非中间值是可观测量，kernel 可选用自然中间精度/指令/归约结合律/分解（例如 sm_89 INT8 prefill 的 fp16 累加 PV 折入 fp32 累加器）。融合 kernel 不要求复现未融合 BF16 物化，也不禁止更低位宽中间量。
- 精确变换/codec 用精确 oracle 精确比较；跨 Op 无共享容差预设，每个激活计算路径为整套测试集中选定一个容差。
- 有状态行为要验证**完整迁移**：KV 页所有权、GDN 线性注意力状态（FP32）、MTP accept/commit、arena 生命周期、CUDA Graph 地址稳定性。

### 性能工作

- 性能论断必须标注层级：算子 / 调度 / 请求阶段 / 端到端，并在该层级直接测量；微基准只支撑算子级论断，端到端归因未解决时才用整体 profiling（NVTX 已埋点：`src/core/nvtx.h`）。
- 端到端测量工具：`ninfer_bench`（pp/tg/pp+tg 矩阵）、`tools/bench/ttft`、serve `/metrics`（`llamacpp:` 前缀兼容计数 + `ninfer:` 系列）。
- 本 fork 的 `sm_89` 特有优化（改相关代码前必须知道）：INT8 注意力 prefill 已为 Ada 重调（128 寄存器预算、8 个成对 producer warp 走 `Bc` 列半区、byte-permute V 反量化位精确、fp16 累加 PV 每 tile 折入 fp32 累加器，64K 深度 +30%）；causal-tile 分区键块遍历（内部块无条件拷贝 + 免 masking select，边界块走精确 masked 路径，位精确）。
- 4090 上已知短板：prefill 比 llama.cpp 慢 16–24%（残余差距在自研量化 GEMM，比 cuBLAS 低约 10%；`--prefill-chunk` 不是杠杆）；解码是本引擎优势区间。

### 所有权边界

- `include/ninfer/{engine.h,types.h}`：不透明公共 Engine 接口（`Engine::prepare/prepare_tokens/tokenize_text/score_tokens/count_tokens/submit/generate/slot_* 会话持久化/slot_states/healthy/...`）；`include/ninfer/ops/` 是库内 Op 契约，均不对外安装。
- `src/core` 设备原语与物理 KV 容器；`src/artifact` 只做 `.ninfer` 框架/绑定/物化，无检查点执行语义。
- `src/ops` 每个语义封闭 Op 的实现（含融合、定形、设备特化路径）；**CMake 源清单显式枚举，新增源文件是构建边界决策**。
- `src/targets/qwen3_6` 只有家族不变量（tokenizer/template、媒体预处理与 MRoPE、prepared-prompt/output 类型、权重视图 schema、被动 Vision 定义、固定的 planning/Program/调度/状态/workspace/CUDA-Graph 算法），无目标身份/registry 项/绑定/叶实现。
- `src/targets/<package>` 拥有注册身份、存储 profile、binder、`LoadedModel`、配置、家族 model view 填充值、诊断、graph frontier 值与三个执行叶家族；叶 Op 本身仍在 `src/ops`。
- `src/runtime` 公共契约、生成 token 事务/发布策略、Engine PIMPL；`src/media/decode` 只消费已拥有字节（URL/path/data 获取归 `product/media_acquire`、CLI 或 serve）；`src/serve` 只做协议翻译与传输。
- 新代码不引入框架式抽象：无通用模型图、家族基类、插件发现、字符串驱动执行、隐式设备分配、运行时权重重打包、为假想模型/硬件的占位。

### 测试纪律

- 常驻测试只保护一个具体风险：精确工件字节/几何/绑定/变换、数值 Op 契约（独立 oracle）、家族 Frontend/Program/前缀/MTP/多模态行为、生成 token commit/stop/cancel 一致性、公共基准或 OpenAI/Anthropic 可观测行为、复现的受支持 bug。
- 不进常驻套件：性能断言（归 bench 与 profiler 评审）、源文件扫描、实现形状断言、trivial getter/配置、已退役命令面、无具体回归风险的宽泛添加。
- 验证必须与语义契约匹配（精确比较 vs 容差 vs 行为准则）；跑聚焦检查集支撑改动论断即可，相关检查跑不了要明说。

### 本地环境

- Python 用维护者环境的 `python3`（3.11）；不主动安装/升级依赖。
- 大工件、源检查点、profiler 输出都是本地前置条件，不在任务范围内不要下载或重新生成；不用 glob/mtime/“latest”选择工件。
- 常规构建目录 `build/`；profiler 输出 `profiles/ncu/`、`profiles/nsys/`、`profiles/bench/`；编译用 `cmake --build build --parallel` 不限并行度。
- 4090 + CUDA 12.8+/13.1 + CMake 3.28+ 是本仓库的目标硬件/工具链；`.ninfer` 工件是架构无关的，下载后按模型卡 SHA-256 校验。
### 开发机构建与测试（本机无 GPU）

- 本机是开发机、**无 NVIDIA 显卡**：nvcc 编译与 ctest 一律在 Docker 镜像 `ninfer-local-build:latest`（nvidia/cuda 13.1.2 基座、CUDA 13.1.2、`/usr/local/cuda/bin` 在 PATH、无默认工作目录）内执行，不在宿主机直接编译或跑 GPU 测试。
- 容器内源码挂载点为 `/src`（`build/CMakeCache.txt` 的 `CMAKE_HOME_DIRECTORY=/src`）：docker run 时把仓库根目录挂到 `/src`，复用 `build/` 树（Ninja：`build.ninja` + `compile_commands.json`），命令为 `cmake --build /src/build --parallel`、`ctest --test-dir /src/build`；`build/` 不存在时先 `cmake -S /src -B /src/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON -DCMAKE_CUDA_ARCHITECTURES=89`（容器内 cmake 默认生成器是 Unix Makefiles，必须显式 `-G Ninja`，否则 `ninja` 找不到 `build.ninja`）
- `build/` 树锁定 **sm_89（只编 4090）**，与根 CMakeLists 的 `FATAL_ERROR` 校验一致：`CMAKE_BUILD_TYPE=Release`、`CMAKE_CUDA_ARCHITECTURES=89`、`CMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc`、`CMAKE_CXX_COMPILER=/usr/bin/c++`、`NINFER_BUILD_APPS=ON`、`NINFER_BUILD_BENCHMARKS=OFF`；重新 configure 时必须继续传 `-G Ninja -DCMAKE_CUDA_ARCHITECTURES=89`
- 构建流水线（移植/功能改动均照此执行）：
  1. 先跑**语法检测门禁**（可并行；只编译 `src/` 的全部 TU、不做链接，快速暴露语法与头文件错误）；
  2. 门禁通过后再启动真正的完整编译（链接 apps 与测试）；
  3. 最后跑完 **CPU 相关的全部 ctest**，所有涉及 GPU 的 case 一律跳过（既有形态见根目录 `test-cpu-remote.log`：87 项、GPU op 项全部 Skipped）。
- 新移植/功能必须补充对应测试 case 后才算完成。

### 4090 远端测试机（密码 SSH，Win10 + Docker Desktop）

- SSH：`192.168.137.2:22`，用户 `ninfer`（home = C:/Users/ninfer，只能操作该目录）；密码 `NinFer-Build2026-x7`。仅支持密码登录（key 认证实测不可用）；本机无 plink，自动化用 Python paramiko（已装），交互可用 `ssh ninfer@192.168.137.2` 每次输密码。
- 开发容器：`ninfer-4090-kaso-dev`（ID 6d40787966745b186677bb19d4630d767c84fae764a8e204eb281e025d45c1be），镜像 `ninfer-4090-devel:0903`（内容 = 仓库根 Dockerfile.toolchain：CUDA 13.1.2-devel + apt 工具链 + python3，已剥 compat）；仓库 bind 到 /ninfer-4090-kaso（与 WORKDIR 一致），构建树在该目录 build/ 下；端口映射 1234:1234；日常 `docker start -ai ninfer-4090-kaso-dev` 回到 bash 操作。
- 容器时区：Dockerfile.toolchain / Dockerfile.runtime 已内置 `ENV TZ=Asia/Shanghai`（容器不带时区时 glibc 本地时间 = UTC，spdlog 日志会显示 UTC）。该 ENV 在容器创建时固化：重建镜像后需 `docker rm` + 重新 `docker run`（旧容器 `docker start` 不拾取新 ENV，容器 ID 也会变）；临时验证旧容器可用 `docker exec -e TZ=Asia/Shanghai -it ninfer-4090-kaso-dev date`（应显示 +0800）。
- GPU 独占约束（重要）：4090 宿主机常驻一个推理服务容器（用户本人管理）。任何会加载模型吃显存的操作（跑 ninfer-serve、引擎级 GPU 测试）之前，必须先请用户关闭该推理端容器，否则 24GB 显存必 OOM。不加载模型的操作（构建、CPU ctest、docker build）不受影响。
- 远端仓库目录只放源码与构建树；模型工件在用户侧（$NINFER_MODEL_DIR / bind 的 /models），不要往仓库目录塞权重。