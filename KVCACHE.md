# KVCACHE.md — ninfer KV Cache / Prefix 复用 / Prefill 逻辑梳理

> 分支：`prefix-v2`（HEAD `4f0faf0c`，前两条 `5a09e419`、`d24fdb90`）。
> 本文对**当前分支代码**中与 KV cache、prefix 复用、prefill 相关的逻辑做全面梳理：
> 每一层的职责、关键类型、关键函数、调用链与参数。权威规范另见
> `docs/maintainer/paged-kv-cache.md`（物理 KV 合同）与
> `docs/maintainer/resource-scheduling-and-context-cache.md`（资源调度与上下文缓存）；
> 本文侧重**代码落点与数据流**。

## 0. 分层总览

```
┌─────────────────────────────────────────────────────────────────────────┐
│ 公共 API    include/ninfer/types.h（KvCacheStorage、ContextCacheOptions、 │
│             PrefixReusePath、RuntimeStats、EngineOptions…）               │
│             include/ninfer/engine.h、include/ninfer/ops/*.h               │
├─────────────────────────────────────────────────────────────────────────┤
│ Target 层   src/targets/qwen3_6/…（qwen3_6 / _27b / _35b_a3b 同一套 impl， │
│  (接线)     经 Variant 参数化）                                            │
│             impl/runtime/{request_plan_impl,program_impl,text_context_impl,│
│                           api_impl,layouts_impl,instance}.h               │
│             impl/state/…（StateImage、GDN 线性注意力状态、slot 持久化）      │
├─────────────────────────────────────────────────────────────────────────┤
│ 引擎层      src/runtime/engine/                                            │
│  (逻辑)     engine_core.h（Engine 状态机 + 主循环）、engine.cpp             │
│             resource_manager.h（catalog/索引/事务，3518 行）               │
│             materialization_planner.h（有界 A*，1222 行）                  │
│             shared_capture_planner.h（共享前缀捕获规划）                    │
│             scheduler.h（FIFO + 单 prefill lane + 准入）                   │
│             context_cost.{h,cpp,defaults.cpp}（机器成本模型）              │
│             kv_capacity.cpp、admission_policy.{h,cpp}、                   │
│             request_record.h、slot_spill_guard.h、resource_search.h        │
│  src/runtime/contract/types.h（PrefillWork、ResourceClass、Checkpoint…）  │
├─────────────────────────────────────────────────────────────────────────┤
│ 物理层      src/core/paged_kv_cache.h（页池/租约/执行表/Host 副本）        │
│  (存储)     src/core/cyclic_kv_cache.h（环形 KV，sliding-window/MTP）      │
│             src/core/transfer_work.h（CoalescedTransferWork）             │
├─────────────────────────────────────────────────────────────────────────┤
│ 算子层      src/ops/kv_cache/append/（KV 写入 + 量化）                     │
│  (CUDA)     src/ops/softmax_attention/{dense,context,packed,causal_cache}  │
│             src/ops/sliding_window/、src/ops/sparse_moe/{prefill,decode}   │
└─────────────────────────────────────────────────────────────────────────┘
```

核心思想（一句话）：**ops 只做"给张量→算/写"，不拥有分配、frontier 与提交权；
target 层把模型执行绑定到 KV 物理视图；引擎层做全部逻辑决策（选 source、规划
转移/驱逐、提交事务），物理可行性由 Program 密封为 `ResourcePlan` 并绑定
`Program::resource_revision()`。**

---

## 1. KV 物理存储

### 1.1 两类 KV

每个 active sequence 持有一组 KV bundle（`src/targets/qwen3_6/impl/runtime/program.h:420`）：

- `text_kv`（**Main KV**）：文本注意力层的 paged KV，frontier 记在
  `SequenceState::text_kv_valid`；
- `backend_kv`（**Backend KV**）：MTP / DFlash 推测解码后端自己的 KV，frontier 记在
  `SequenceState::mtp_kv_valid`（仅 `speculative_backend==Mtp` 时存在）。

frontier 推进/收缩：

- `ProgramImplCore::commit_sequence_kv`（`src/targets/qwen3_6/impl/runtime/program_impl.h:10770`）：
  对 `text_kv_addresses` / `backend_kv_addresses` 调 `commit_frontier`；
- `trim_sequence_kv`（同文件 :10783）：`destructive_truncate` 缩回；
- `release_sequence_kv*`（:10817-:10879）：解除/归还执行映射与页面租约。

### 1.2 Paged 页池（`src/core/paged_kv_cache.h`）

- 页大小固定：`kPagedKVPageSize = 64`（:17）。
- `PagedKVLayerView`（:20）/ `PagedKVBatchLayerView`（:39）：算子消费侧视图
  （`k_pages/v_pages/k_scale_pages/v_scale_pages` + `head_dim/num_kv_heads/quant_group` +
  量化特征位 `packed_v/rotate_k/rotate_v/packed_k/e8_lattice/e8_root`）。
  物理平面的真实几何由 `KVPageGeometry{page_tokens=64, planes[]}` 与
  `KVExecutionTableSpec{logical_page_capacity, table_rows}` 描述。
- `DeviceKVPagePool`（:200）：一组 homogeneous planes；`FreePageRun` 管理空闲区间；
  `page_generations_` 防止 ABA；`reserve()/materialize()/dematerialize()` 实现
  reservation→lease 的容量语义；`copy_page/copy_to_host/copy_from_host`（含
  fork-local 调用方缓冲变体，用于 slot 文件持久化）。
- `DeviceKVPageLease`（:143，move-only 页租约）、`DeviceKVPageReservation`（:172，
  尚未绑定物理 ID 的容量预留）、`DeviceKVPageHandle`（:122，可拷贝的页凭证，
  带 generation）。
- `KVExecutionTablePool`（:351）：每 sequence 一个逻辑地址空间行
  （`KVExecutionRowLease`，带 generation）；`publish(row, logical_begin, pages…)`
  把逻辑页区间映射到物理页，GPU 侧经 block table 直接寻址。
  一个 request 的 KV **不要求物理连续**，也不与 lane 绑定；所有 active/inactive
  地址空间共享 pool capacity，active 通过 reservation 保证容量。
- `HostKVAllocationView/ConstView` + `HostKVPageLayout`：pinned host 副本（容量
  由 `EngineOptions::context_cache.host_kv_capacity_bytes` 决定，默认 0 = 关闭）。

### 1.3 环形 KV（`src/core/cyclic_kv_cache.h`）

`CyclicKVCacheLayerView`：lane-owned 固定容量环，sliding-window 注意力与
`kv_cache_append_prefix` 的 cyclic 形态使用。注册 profile 为 D=128、Hkv=8、
capacity=4096，绝对位置 p 存于 slot `p mod 4096`。

### 1.4 KV 存储格式（`include/ninfer/types.h`）

```cpp
enum class KvCacheStorage : uint8_t {
    BFloat16, Int8Group64,
    RotatedInt8KeyInt4ValueGroup64, RotatedInt4KeyInt4ValueGroup64,
    RK4V4E8, RK2V4E8, Fp8E4M3Row256,
};
```

`EngineOptions::kv_cache` 默认 `BFloat16`。量化编解码实现在
`src/ops/kv_cache/append/`：`fp8_e4m3_row_codec.cuh`（D256 行级 FP16 scale，
raw_scale=a/448，clamp `0x1p-24..65504`）、`int8_g64_codec.cuh`（每 64 值一组
FP16 scale：`scale_bits=FP16_RNE(a/127)`，`code=I8(clamp(RNE(x*inv),-127,127))`）、
`hadamard_d256.cuh` + `d256_profile.h`（K 的固定正交旋转变换，causal consumer
用匹配的 Q 准备配对）。

### 1.5 容量解析

- `KvCapacityPolicy{mode, explicit_tokens=2048, automatic_headroom_bytes=0}`；
  `kDefaultKvCapacityHeadroomBytes = 1GiB`。
- `resolve_kv_capacity(policy, curve, available_runtime_bytes)`
  （`src/runtime/engine/kv_capacity.cpp`）：
  - `Explicit`：`pages = 1 + (explicit_tokens-1)/main_page_tokens`，
    且 `automatic_headroom` 必须为 0；
  - `Automatic`：`budget = available - headroom`，
    `additional = (budget - min_resv)/bytes_per_group`，
    `pages = min(min_groups + additional, max_groups)`；
  - 溢出/非法曲线抛异常（`min<=max`，可扩展时 stride 非零）。
- `SequenceCapacityCurve`（`src/runtime/contract/types.h`）：`{main_page_tokens,
  minimum/maximum_main_page_groups, minimum_device_reservation_bytes,
  bytes_per_additional_main_page_group}`；`KvCapacityResolution` 记录
  `resolved_tokens/runtime_reservation_bytes/available_after_weights_bytes/
  automatic_headroom_bytes/planned_slack_bytes` 等，映射进
  `MemorySummary`（`kv_capacity_page_groups/kv_capacity_max_page_groups/
  kv_payload_bytes/text_kv_bytes/mtp_kv_bytes/gdn_state_bytes/dflash_kv_bytes/
  host_state_capacity_slots/…`）。

---

## 2. KV 写入算子（`include/ninfer/ops/kv_cache_append.h`，实现 `src/ops/kv_cache/append/`）

| 入口 | 签名要点 | 说明 |
|---|---|---|
| `kv_cache_append` | `(k,v,positions,PagedKVLayerView,stream)` | 单 lane 追加。k/v 连续 BF16 `[256,4\|2,T]`，positions 连续 I32 `[T]`。BF16 逐位拷贝；INT8-G64/FP8 按 1.4 的量化规则写 K 与 scale 平面，**V 保持 BF16 原值**；K 是"实现拥有的固定正交变换"后的物理表示 |
| `kv_cache_append_prefix`（paged） | `(k,v,positions,counts,table_rows,envelope,PagedKVBatchLayerView,stream)` | B 行批量前缀追加。k/v 连续 BF16 `[128,8,T,B]`，positions I32 `[T,B]`，counts/table_rows I32 `[B]`。行 b 的 `[0,counts[b])` 逐位拷贝到 `table_rows[b]` 逻辑位置 `positions[i,b]`；paged 平面 head-major `[128,64,Nphysical,8]`；被拒尾部 `[counts[b],T)` **不写任何字节**。约束 `T>0, 1<=B<=8, envelope.min<=counts[b]<=envelope.max<=T` |
| `kv_cache_append_prefix`（cyclic） | `(k,v,positions,counts,lanes,envelope,CyclicKVCacheLayerView,stream)` | lane-owned 环（D=128,Hkv=8,capacity=4096，`p→p mod 4096`）。每行现有活跃区间必须**紧接** `positions[0,b]` 之前；推进 `counts[b]` 后所有被覆盖旧 slot 变死；每行最多提交一个环容量；无两行竞争同一物理 slot |

`KVCacheAppendPrefixExecutionEnvelope{min_count,max_count}`：host launch 资源承诺
（约束 device 捕获/重放期间的 count），**不选择/发布 commit frontier**。

> 所有 append 算子**不拥有**持久分配、frontier、request identity 或 commit 权——
> 这些全部由 target/引擎层管理（KV 写入与 frontier 推进在同一 Program 事务内提交）。

`prepare_ragged_prefix`（`include/ninfer/ops/prepare_ragged_prefix.h`）：把
lane-owned BF16 前缀（GDN 线性注意力状态图像，source `[D,W,C]`）打包成 compact
ragged batch `[D,W,B]`，尾部零填充并重复最后活跃位置；`D%8==0`，BF16 16 字节对齐。

---

## 3. 注意力算子（prefill/decode 的计算路径）

`include/ninfer/ops/softmax_attention.h` 定义共享数值合同：每个 query 头 h 读 KV 头
`floor(h/(Hq/Hkv))`；理想 oracle 为 FP64 朴素 QK + 稳定 softmax + V 加权；各
storage 路线（BF16 / INT8-G64 / FP8-E4M3FN）各有独立数值判据；FP8 compute profile：
原生 E4M3FN QK MMA + FP32 累加、FP32 softmax、E4M3FN→FP16 V 转换 + 一次 FP16
scale 乘法、FP16 P/V MMA + FP32 累加、BF16 输出。

`kCausalAttentionMaximumVisibleKeys = 1048576`。所有算子带
`*ExecutionEnvelope{min_,max_}` —— **只是 host launch/workspace 资源承诺，
从不改变可见 key 集或数值结果**。

| 算子 | 注册 profile | 用途 |
|---|---|---|
| `softmax_attention` | D=72, Hq=Hkv=16, scale=1/√72，q/k/v/out BF16 `[72,16,T]` | 稠密非因果单段（vision 编码等） |
| `packed_softmax_attention` | 同上 | cu_seqlens `[S+1]` 块对角稠密（多段独立）；等长变体免 workspace |
| `causal_softmax_attention` | `[256,24,4]`(g6) / `[256,16,2]`(g8)，scale=1/√256 | **主 KV 的 append+attend 融合**：q/out `[D,Hq,W,B]`，k/v `[D,Hkv,W,B]`，B=1 任意 W，B=2..8 时 W=1..16。`valid_columns` 稠密/掩码拓扑由调用方声明。当前行先追加再被观察；尾列不写 cache 且输出精确 BF16 0 |
| `causal_softmax_attention_cached` | 同上 | 只读、cache 已填满（不追加 K/V） |
| `context_softmax_attention` | D=128, Hq=32, Hkv=8, scale=1/√128, T=1..16, B=1..8 | 持久 context + 一个 live query block 的**非因果** GQA；context 为 paged BF16（head-major `[128,64,N,8]`）只读；query K/V 为独立临时段；backend KV / MTP 路径 |
| `sliding_window_attention`（`include/ninfer/ops/sliding_window_attention.h`） | D=128, Hq=32, Hkv=8, window=4096, T=1..16, B=1..8 | 对称非因果滑窗：`positions[i,b]=L+i`，`lanes[b]` 选环 lane；committed 区间 `[max(0,L-window),L)`；可见判据 `abs(p_j-p_i)<4096`（4095 含、4096 不含），且每个 live query 看同 batch 行全部临时 query K/V |

算子实现目录：`src/ops/softmax_attention/{common,dense,context,packed,causal_cache}`
（`causal_cache/` 下 `prompt_*` 为 bf16/fp8/i8 各路线，`small_t_*` 为小 T 特化；
`common/context_query.cuh` 24.5KB 是 Q 侧准备的核心）与
`src/ops/sliding_window/`。

---

## 4. Prefix 复用：公共 API 与请求级声明

### 4.1 引擎选项（`include/ninfer/types.h`）

```cpp
struct ContextCacheOptions {
    bool enabled = true;                       // 总开关
    std::optional<uint32_t> device_state_slots;      // 额外 Device GDN 状态槽 H
    uint32_t host_state_slots = 0;             // Host StateImage 槽（pinned，默认关）
    size_t   host_kv_capacity_bytes = 0;       // Host KV pinned 字节（默认关）
    std::optional<uint32_t> max_private_continuations;  // P
    std::optional<uint32_t> max_shared_prefixes;        // S
    std::optional<uint32_t> max_long_anchors_per_continuation; // L
};
```

启用时默认（C=max_concurrency）：**H=C**（Device 状态总容量 C+H）、host 两层默认 0
（显式 opt-in，防低内存主机耗尽 pinned RAM）、**P=2C、S=C、L=2**；
`Engine::options()` 返回解析后的有效值。

`EngineOptions` 中相关项：`max_context=2048`、`kv_capacity`（默认
`explicit_capacity(2048)`）、`prefill_chunk=1024`、`turn_checkpoint_ring=0`
（每 lane 保留 N 个 host 端 turn checkpoint；每项为完整 GDN 状态图像，
Qwen3.8-27B 约 147 MiB）、`kv_cache=BFloat16`、`speculative`、`use_cuda_graph=true`、
`context_cache`、`context_cost{preset_path}`。

### 4.2 请求级 hint（`PromptInput::context_cache`）

```cpp
struct ContextCacheHints {
    std::optional<std::string> session_key;            // 会话谱系（FNV 索引）
    CacheRetentionHint retention = Default;             // Default|LiveSession|Disposable
    std::vector<PromptCacheMarker> markers;
    bool allow_engine_automatic_shared_prefixes = true; // 协议自有写策略时关掉引擎结构候选
    bool update_session_index = true;                   // 推进命名会话谱系
    uint32_t automatic_private_anchors = 0;             // 在最后 N 个 message 边界提
    // 出 PrivateLongAnchor（opportunities 而非 markers，不占显式 marker 限额，
    // 同 frontier 与显式 anchor 合并）；0 = 上游行为
};

struct PromptCacheMarker {
    uint32_t after_message_count = 0;
    PromptCacheMarkerKind kind = SharedStablePrefix;    // | PrivateLongAnchor
    SharedCandidateEvidence evidence = ExplicitBoundary;
    PromptCacheMarkerLocation location = MessageBoundary;
    // MessageBoundary|MessagePartBoundary|LeadingInstructionBoundary|ToolBoundary
    uint32_t leading_instruction_bytes = 0;
    uint32_t after_tool_count = 0;
    uint32_t after_message_part_count = 0;
};

enum class SharedCandidateEvidence : uint8_t {
    None=0, ExplicitBoundary=1, RequestedAutomatic=2, DefaultAutomatic=4,
    EngineStructural=8, EngineObserved=16   // 位标志，| 合并
};
```

其他公共类型：`ExecutionOptions.allow_prefix_reuse=true`（默认开，可 per-request 关）；
`PrefixReusePath{Root, PrivateEndpoint, PrivateTurnClosure, PrivateResponseReplay,
PrivateLongAnchor, SharedStablePrefix}`；`GenerationStart{prompt, reused_prompt_tokens}`
（流式消费者在**任何 OutputDelta 之前**、prefix 选择与物化预留提交后收到一次）；
`GenerationResult.reused_prompt_tokens/prefix_reuse_path/materialization/slot/
session_digest`。

会话持久化 API：`SlotState{processing, retained, prompt_tokens, cached_tokens,
session_digest, checkpoints}`、`SlotCheckpoint{frontier, session_digest}`
（FNV-1a 64 前缀摘要的 16 hex）、`SlotSaveResult/SlotRestoreResult`、
`SlotSessionMismatch`（`if_digest` 前置条件不匹配）、`SlotAutoSaveEvent`
（被非自愿驱逐前回写 slot 文件；`skipped_behind_tokens` 表示被 spill guard 拒绝）。

---

## 5. Prefix 识别与候选构建（target 层）

### 5.1 前缀身份

`PreparedPromptAccess` 上：`prefix_digests`（每个候选 frontier 的摘要向量）、
`prefix_identity_tag = capture_identity_tag(speculative_backend, proposal_head, kv_dtype)`
（`src/targets/qwen3_6/impl/runtime/request_plan_impl.h` 的 `publish_continuation`
分支，`base->prefix_digests.assign(prompt)`）。

`qwen3_6::PrefixShortlistKey{digests, frontier, identity_tag}` 是 prefix 索引的 key；
`RequestBasePlan::prefix_shortlist_key(frontier)`（`api_impl.h:111`）取
`digests.at(frontier)` 组装。

### 5.2 capture groups 构建（`request_plan_impl.h:300-419`）

门控：`options.allow_prefix_reuse && prompt.identity.reusable && context_cache.enabled`。

1. `rewrite_checkpoint` 先加（frontier 必须 `<= prompt frontier`，否则
   `"rewrite checkpoint frontier must lie at or inside the prompt frontier"`；
   `rewrite_execution_frontiers` 必须有序唯一，否则
   `"rewrite execution frontiers must be ordered unique prompt positions"`；
   input_order=0，evidence=None）。
2. 遍历 `base->context_cache.opportunities`
   （`qwen3_6::PreparedCacheOpportunity{frontier,input_order,kind,evidence}`，
   由 markers + automatic anchors + 引擎结构候选展开；
   `kind==SharedStablePrefix→shared 候选`，`PrivateLongAnchor→long_anchor`）。
   frontier 非法（0 或 > prompt_tokens）抛 `"capture opportunity frontier is invalid"`。
3. `add_capture(frontier,input_order,rewrite,shared,long_anchor,evidence)`：
   frontier 必须在 `(0,prompt_tokens]`；shared → `base->shared_candidates`，
   否则 → `base->capture_groups`；**同 frontier 合并**（input_order 取 min，
   shared/long_anchor/rewrite 取或，evidence 按 `|` 合并）。
4. 两组按 `(frontier,input_order)` 排序；构建 `PreparedCaptureBacking`
   （token_ids 截到 `backing_frontier=max(capture_frontier,shared_frontier)`，
   prefix_identity 同步截断）；每个 group 构建 `PreparedCaptureIdentity{backing,
   rebuild_work=rebuild_work_at_frontier(prompt, frontier, prefill_chunk,
   capture_groups, rewrite_execution_frontiers), shortlist_key}`。

`rebuild_work` 即"丢弃该 checkpoint、从 root 重建"的 `PrefillWork`，供
context cost 模型计价（见 §7.1）。

### 5.3 规划后处理（`request_plan_impl.h:600-729`）

引擎 `plan_materialization` 选中一个 candidate 后，target 层做以下收尾：

- 选中 source 的 rewrite / long_anchor **可选状态**：endpoint 落在其
  `optional_states` 内 → `source_mode=Retain`（源原地保留），否则
  `ConsumeToActive`（消费/转移），并按 residency（`DeviceOnly/Both→device.state_slots`；
  `HostOnly/Both→host.state_slots`）累计 active optional resources（去重 + valid）。
- rewrite restore 时额外做 `state_exclusive_to_sequence` 过滤，且
  `anchor.frontier > reuse_base` 的跳过；`ConsumeToActive` 时
  `state_fork_required = selected_state_requires_fork(*source, reuse,
  rewrite_disposition, selected_checkpoint, reuse_base)`。
- `plan->capture_groups` 过滤为 `frontier > reuse_base` 且（rewrite 匹配
  `ReplaceAtCommittedFrontier` 或 shared 或 long_anchor）；
  `plan->shared_candidates` 过滤为 `frontier >= reuse_base`。
- `plan->summary.reusable_prompt_tokens = plan->reuse_base`；
  `plan->summary.prefix_reuse_path = plan->reuse`（`ReusePath` 与
  `PrefixReusePath` 同序）。
- **MTP 桥接**：`reuse==Root` 或 `PrivateEndpoint` → `prepare_mtp=true`
  （PrivateEndpoint 时 `mtp_bridge = (reuse_base < prompt_tokens) ? BeforeSuffix
  : AfterExactHit`）；rewrite / LongAnchor / SharedStablePrefix → 同样
  `prepare_mtp=true` + 同规则 bridge（桥接步在 base-1 位置跑一步 MTP 前向，
  用 source 的 tail hidden 作为 previous_hidden，见 §9.3）。
- **vision**：`uses` 过滤 `token_end > reuse_base` 的项；`prepare_mtp` 时
  `begin = token_begin - 1`（对齐 shifted embedding），`max_merged_count` 取 max。

MTP 可行性预检（:550-:560）：`reuse_base==0 || source->mtp_kv_valid >= reuse_base-1`，
否则带 MTP 的 candidate 不可行。

---

## 6. 逻辑调度核心：`ResourceManager`（`src/runtime/engine/resource_manager.h`，3518 行）

`ResourceManager<Package>` 只拥有**逻辑策略**；每个物理可行性决策都密封为
`Package::ResourcePlan` 并绑定 `Program::resource_revision()`。

### 6.1 状态与数据结构

| 结构 | 状态 | 说明 |
|---|---|---|
| `lanes_[kMaximumConcurrency=8]` | `LogicalLaneState{Free,Materializing,Active,TerminalPending}` | 并发 lane |
| `catalog_`（private 延续槽，容量 P） | `CatalogState{Vacant,Catalogued,Claimed,ReservedForActive}` | `CatalogEntry{state,id,revision(≥1),summary:ContinuationSummary{endpoint,rewrite,long_anchors,active_references},handle:ContinuationHandle,session:CacheSessionKey,observations,retention}`；`observation_capacity = 3 + max_long_anchors` |
| `shared_catalog_`（容量 S） | `SharedCatalogState{Vacant,Catalogued,Claimed,ReservedCapture}` | `SharedCatalogEntry{…,summary:SharedPrefixSummary{checkpoint,active_references,handle},observation{SharedStable},transaction_pins,explicit_credit,credit_expiry_epoch}` |
| `session_index_` | `{Empty,Occupied,Deleted}` | 开放寻址线性探测；session_hash 用 FNV-1a（prime 1469598103934665603 / 1099511628211）；`SessionIndexEntry{state,key,slot,owner_id,revision,publication_order}`；Deleted 可复用 |
| `prefix_index_` | 位图式 | `PrefixIndexEntry{occupied,shared,key:PrefixShortlistKey,slot,owner_id,revision,checkpoint}`；容量 = `private_capacity*(max_long_anchors+2) + shared_capacity`（`checked_prefix_index_capacity`）；每次需要前 `rebuild_prefix_index()` 重建（遍历 catalog/shared catalog 的 Catalogued+handle 条目的 endpoint/rewrite/long_anchors + shared checkpoint） |
| `demand_window_` | 滑动窗口 | `kDemandWindowCapacity=32`；`PrefixDemandRecord{domain:ReuseDomainId{low,high},candidate_keys,exact_resident_keys,selected_source_key}`；满则 erase 最旧；`commit_demand` 顺带衰减 shared 的 `explicit_credit`（key 已在 exact_resident_keys 中→credit=false；否则 `demand_epoch_>=credit_expiry_epoch`→credit=false） |
| `transaction_` | variant | `monostate | MaterializationRecord | ActiveCaptureRecord`（每时刻至多一个事务） |

`reuse_domain`：无 session 时 `{publication_order, publication_order^0xD6E8FEB86659FD93}`；
有 session 时 FNV 双哈希。`matching_reuse_domains(key)` 统计匹配该 key 的不同
domain 数（≥2 视为"多会话都在用"，是 shared 捕获的 pressure evidence 之一）。

### 6.2 `inspect()`（候选生成 + 选择，line ~1044）

1. `transaction_` 非空或 `program.has_context_transaction()` → `TemporarilyBlocked`；
2. `publication_order==0` → 抛异常；
3. `!program.isolated_request_feasible(base)` → `PermanentlyInfeasible`；
4. 找第一个 `Free` lane 作 destination；没有 → `TemporarilyBlocked`；
5. `rebuild_prefix_index()`；
6. 构建 `provisional_demand`（domain 按 §6.1；candidate_keys 取
   `context_cache().opportunities` 中 `kind==SharedStablePrefix` 的
   `base.prefix_shortlist_key(frontier)`）；
7. `current_session_cell = find_session_cell`（当 `session_key && update_session_index`）；
8. **root 候选** = `program.inspect_admission(prompt, base, *destination,
   nullptr, nullptr, nullopt, false)`（失败抛
   `"Program rejected isolated root planning"`）；
9. 对每个有效 `PrefixIndexEntry`：
   - **private 路径**：要求 `CatalogState::Catalogued && handle &&
     !private_has_active_edge(slot)`；`retain = entry.session &&
     (!req_session || 不同 || !update_session_index)`；调
     `inspect_admission(prompt,base,dest,&*entry.handle,nullptr,index.checkpoint,retain)`；
     `reusable_prompt_tokens==0` 或 `(retain && source_mode!=Retain)` → 抛
     `"Program returned an invalid private candidate"`；
     `current_session_binding = (cell.slot==index.slot && cell.owner_id==entry.id &&
     cell.revision==entry.revision)`；key 记入 `exact_resident_keys`；
   - **shared 路径**：要求 `SharedCatalogState::Catalogued && handle`；调
     `inspect_admission(prompt,base,dest,nullptr,&*entry.handle,index.checkpoint,false)`；
     `reusable==0` 或 `source_mode!=Retain` → 抛
     `"Program returned an invalid shared candidate"`；
10. `plan_materialization(program, prompt, base, dest, candidates,
    publication_order, planning_started, provisional_demand)` 选择；未选 →
    `TemporarilyBlocked`；否则 `readiness = needs_transfer ? NeedsTransfer : Ready`。

### 6.3 `reserve_materialization()`（line ~474）

- `choice.plan_` 必须存在且其 `resource_revision == program.resource_revision()`，
  否则 `Stale`；`validate_choice`；`take_materialization_record`；
- **顺序关键（fork-local）**：`observe_planned_evictions(record.private_claims)`
  **必须发生在** `program.start_resource_transaction(...)` **之前**
  （驱逐观察者要读 record 的 claim spans，事务会改变它们）；
- `reserve_logical_materialization`（Catalogued→Claimed 等状态迁移）；
- `program.start_resource_transaction(std::move(*plan), std::move(prompt),
  cancellation)`；`Aborted` → `rollback_logical_materialization` + 按
  `cancellation.requested()` 返回 `Aborted` 或 `Stale`；成功 →
  `observe_planner_diagnostics` + `Reserved`。

### 6.4 `reserve_active_capture()`（line ~434，共享前缀捕获，最复杂）

- lane 必须 `Active`；manager/program 事务必须一致；program 已有事务 →
  `skip_capture` + `Skipped`；
- `rebuild_prefix_index()`；
- `private_baseline = program.inspect_capture(offer,nullptr,nullptr,nullopt,false)`；
  若 `private_replacement_candidates` 非空，取
  `min_element`（按 `tuple{kind,frontier,ordinal}`）再
  `inspect_capture(...,private_replacement,false)` 得 `private_replacement`；
- `candidate = inspect_capture(...,true)`（publishes_shared）；
- 若 `candidate.publishes_shared`：遍历 prefix_index 找 **exact_shared**
  （`index.shared && valid && index.key==candidate.shortlist_key &&
  program.shared_capture_matches(offer,*entry.handle)`）；
  - **exact 命中**：`!private_baseline.publishes_private ||
    !physically_feasible` → `Skipped`；否则 `ActiveCaptureRecord{lane,
    publishes_private=true}` +
    `program.reserve_active_capture(offer, exact_shared, nullptr,
    private_replacement, false, cancellation)`；
  - **否则 scenarios**：
    - `pressure_evidence = candidate.shared_evidence 含 ExplicitBoundary|
      RequestedAutomatic，或 matching_reuse_domains(shortlist_key) >= 2`；
    - scenarios = [第一个 Vacant shared slot（stable_ordinal 0）] +
      （若有 pressure_evidence）所有 `Catalogued && !transaction_pins &&
      shared_active_edge_count==0` 的 shared slot 作为 replacement
      （stable_ordinal = 1+slot），各自
      `inspect_capture(offer,nullptr,&*entry.handle,private_replacement,true)`，
      `!publishes_shared` 的跳过；
    - owner_policies（private: `private_retention_weight(entry.retention)`；
      shared: `explicit_credit`）、checkpoint_policies（per private
      endpoint/rewrite/long_anchor + shared checkpoint，
      `demand_mask=committed_demand_mask_for(shortlist_key)`，
      `rebuild_ns=cost_model.prefill_ns(rebuild_work)`，
      `baseline_recovery_ns=price_checkpoint_recovery_work(cost_model,
      program.checkpoint_recovery_work(handle, ref))`）；
    - 每个 scenario 调 `capture_planner_.plan(program, cost_model, input{capture,
      owners, owner/checkpoint policies, direct_shared_victim,
      candidate_demand_mask, candidate_rebuild_ns=prefill_ns(protected_rebuild_work),
      private_baseline_immediate_ns=price_context_transfer_requirements(baseline),
      blocked_runnable_requests, stable_scenario_ordinal,
      target_budget=CapturePlanner::kTargetBudget/scenarios})`；
      选 **max net_gain**（平手按 stable ordinals 字典序）；
- 未选：退回私有捕获 reserve（若 baseline 可行）；
- 选中：构建 `ActiveCaptureRecord`（publishes_private/publication_slot/
  replacement_id/revision/shared_evidence）；对每个 owner_outcome 校验 owner 存在
  且非 stale（private 需 Catalogued+handle+id/revision 匹配+无 active edge；
  shared 需 Catalogued+handle+id/revision+无 transaction_pins+active edge 0），
  计算 `dropped_checkpoints`（`selected_checkpoint_drops`）填入 claims；
  `!plan.pressure` → 抛 `"selected shared capture has no pressure plan"`；
  校验 publication slot（replacement_id==0 需 Vacant，否则需 Catalogued+匹配）；
  `observe_planned_evictions` → `reserve_logical_active_capture` →
  `program.reserve_active_capture_with_pressure(offer,nullptr,scenario.replacement,
  private_replacement,true,*plan.pressure,cancellation)`；`Aborted` → rollback。

### 6.5 进度采纳与终结

- `adopt_materialization_progress`（line ~2791）：校验 `result.victims` 与
  `record.private_claims` 对齐、`shared_victims` 与 `shared_claims` 对齐、
  private/shared result ID 唯一；`validate_private_action/validate_shared_action`
  （disposition/dropped_checkpoints/pressure_committed/final_summary 一致性）；
  source/shared_source 语义校验（Retain 不要求 source_mode 匹配；ConsumeToActive
  要求 final_summary 空）；published 时校验 publication 可释放（source_cell 或
  victim_cell Evicted+pressure_committed）；`observe_selected_hit`；
  Aborted → `restore_unreported_materialization` + lane Free + finalize；
  否则 `publication.state=ReservedForActive` + 分配 id + 填 active
  （`retained_private_source/shared_sources` edges）+ `commit_demand(record.demand)`
  → 返回 `PublishedActivation`。
- `adopt_active_capture_progress`（line ~2955）：publishes_shared 时要求
  `publication.state==ReservedCapture` 且（replacement_id==0 →
  `!capacity_preparation_committed`；否则匹配 replacement）；published 时若
  publishes_shared 则填 publication（分配 id，summary=shared->summary，handle，
  observation{SharedStable}，`explicit_credit = evidence 含
  ExplicitBoundary|RequestedAutomatic`，`credit_expiry_epoch = demand_epoch_ +
  kDemandWindowCapacity`，shared_sources 加 edge）。
- `finish()`（line ~1065）：lane → `TerminalPending`；
  `result.status != Consumed` → `program.abort` + release + `Released`；
  `cache_enabled && Catalogued`：要求 disposition==Catalogued && continuation &&
  `valid_continuation_summary` && `publication.state==ReservedForActive` &&
  `publication.id==active.continuation_id`，否则 release + 抛
  `"Program returned an invalid terminal continuation"`；随后
  `publication.state=Catalogued`、`assign_continuation_summary`、
  handle/session/retention 迁移、`migrate_observations`、`advance_revision`、
  `publish_session`（失败则 session.reset + retention=RecentPrivate）、lane Free。
- `abort()`：→ TerminalPending、program.abort、清 catalog entry。
- `apply_commit`：Active / Finishable(→TerminalPending) / CancelledReleased
  （`release_cancelled_lane`）。

### 6.6 校验规则与权重

- `valid_continuation_summary`：endpoint 必须是 `SessionEndpoint + Private scope`；
  rewrite 不能是 SessionEndpoint/SharedStablePrefix/LongAnchor；long_anchors 必须
  `LongAnchor + Private + ordinal!=0 + ordinal<=max + frontier!=0 + main_pages!=0 +
  tokens!=0`，ordinal 不重复。
- `valid_shared_prefix_summary`：`kind==SharedStablePrefix + frontier!=0 +
  ordinal==0 + Shared scope + shortlist_key.frontier==ref.frontier +
  main_pages!=0 + tokens!=0`。
- `private_retention_weight`：`Disposable=1, RecentPrivate=4, LiveSession=16,
  SharedStable=0`（SharedStable 的代价走 shared 侧的 rebuild/baseline 计价）。
- `checkpoint` 语义（`src/runtime/contract/types.h`）：
  `CheckpointKind{SessionEndpoint, TurnClosure, ResponseReplay, SharedStablePrefix,
  LongAnchor}`；`CheckpointRef{kind, frontier, ordinal}`（singleton 类 ordinal=0，
  LongAnchor 为 per-continuation slot）；`CheckpointScope{Private,Shared}`；
  `RetentionClass{SharedStable,LiveSession,RecentPrivate,Disposable}`。

### 6.7 磁盘 slot 持久化（fork-local）

`catalog_slot()` 视图、`adopt_restored(slot,handle,summary)`（Catalogued+
RecentPrivate）、`take_catalogued(slot)`、`set_eviction_observer/
set_slot_release_observer`（session 的 slot 文件绑定必须随 cell 不再持有
session 而死）、`clear_after_program_cleanup`。slot 文件写前由
`slot_spill_guard.h` 的高水位 token 标记把关：`blocks()` 拒绝比已记录深度更浅的
自动 spill（D3 事故，2026-09-04：31,505 token 的拷贝差点覆盖 78,020 token 的文件；
`note_authoritative` 显式 save/restore 才 SET 标记，`note_spilled` 只升不降）。

---

## 7. 规划器

### 7.1 成本模型（`src/runtime/engine/context_cost.{h,cpp}` + `context_cost_defaults.cpp`）

- `ContextTransferCost{batch_ns, operation_ns, ns_per_byte_q32}`：
  `transfer_ns = max(batch_ns + ops*operation_ns, q32(bytes, ns_per_byte_q32))`；
  q32 定点，`kContextCostQ32One = 1<<32`。
- `ContextPrefillCost{chunk_ns, token_ns_q32, attention_pair_ns_q32,
  vision_item_ns, vision_patch_ns_q32}`；`prefill_ns(work)` 按
  `PrefillWork{chunks, tokens, attention_pairs, vision_items, vision_patches}` 计价。
- 解析优先级：`GenericDefault → CompiledDefault → External`（JSON
  `schema_version 2, artifact_type "ninfer_context_cost_presets"`；transfer 与
  prefill 组件**独立**解析；`ContextCostOptions.preset_path` 指定）。
  硬件类 = `context_cost_hardware_class()` → `"nvidia-<slug>-sm<major><minor>"`
  （编译默认硬件 `"nvidia-geforce-rtx-5090-sm120"`；preset 如
  `qwen3.6-27b/groupwise-int`、`qwen3.8-27b/nvfp4`）。
- 计价函数：`price_materialization_machine_work`
  （`optimistic_request_ns = prefill + optimistic transfers`，
  `immediate_ns = prefill + pressure + candidate`，`transferred_bytes/
  copy_operations` 汇总 `MaterializationMachineWork{pressure_transfers,
  candidate_transfers, optimistic_candidate_transfers, remaining_prefill_work,
  reused_prompt_tokens}`，`CoalescedTransferWork = array<TransferWork,3>`
  按 `ContextResourceClass{State,MainKV,BackendKV}` ×
  `ContextTransferDirection{DeviceToHost,HostToDevice,DeviceToDevice}` 合并）；
  `price_checkpoint_recovery_work`（取 alternatives 的 min，
  `CheckpointRecoveryAlternativeWork{transfers,prefill}`）；
  `price_context_transfer_requirements`（按方向合并）。
- 原子 upsert：`upsert_context_transfer/prefill_cost_atomic`（JSON 原子重写）。

### 7.2 `MaterializationPlanner`（`src/runtime/engine/materialization_planner.h`，1222 行）

常量：`kTargetBudget=4096`（`kGuidedBeamWidth=16`、`kGuidedAssessmentBudget=32`）。

`plan(program, prompt, machine_cost, candidates, root_candidate_index,
pressure_inputs, logical_goal, final_schedule, planning_started)` 流程：

1. 校验 candidates 非空、ID 唯一；重置 frontier/ledger
   （`BoundedTargetLedger` 状态位 `Discovered=1/Assessed=2/Expanded=4`，线性扫描、
   无分配，见 `resource_search.h`）；
2. **identity 阶段**：对每个 candidate 调 `identity_assessment()` 得
   `IdentityMaterializationAssessment{physical_status{Feasible,Infeasible,
   StructuralInvalid}, source_mode{Retain,ConsumeToActive},
   machine_work, pressure_may_change_machine_work, expandable, projection_work,
   assessment_digest}`，`fold_identity` 得 `FoldedCost`；`Feasible` 者生成
   `LogicalGoal{publication_slot}`；选 `identity_best`（`cost.less`）；
3. **快速路径**：`identity_best` 存在且无 candidate 需要扩展
   （`needs_pressure` = 不可行或 Infeasible 但 expandable；`pressure_can_improve`
   = 可行且 `pressure_may_change_machine_work`）→ `final_schedule` 给出
   `shared_capture_frontiers`（`price_split` 用
   `program.shared_capture_split_prefill_work` 的增量计价）→
   `program.seal_identity(candidate, prompt, FinalScheduleIntent{shared_capture_frontiers})`
   → `diagnostics.stop_reason = NoPressure` → 返回；
4. **有界压力搜索**（否则）：`program.begin_pressure_planning(candidate_handles,
   candidate_ids, private_owners/ids, shared_owners/ids)`；
   - 无 `identity_best` 时，先对 root candidate 的 `root_maximal_target` 做
     `assess`（`assessed` 必须仍指向同一 candidate，否则抛
     `"maximal pressure target changed admission candidate"`）；不可行则返回
     nullopt；
   - `search_budget_ns = min(5'000'000, incumbent.cost.total_ns / 20)`
     （guided watchdog 同值）；
   - 每个 expandable root 推入 queue + guided beam（
     `session.guidance(target)` 折叠为 `GuidanceCost`）；
   - guided closure 阶段：按 `closure_order`（lower_bound 升序）对
     `session.guided_closure_target(id, preferred_owner_ids)` 评估；
     偏好 owner 按 `tuple{selected_hit_count, explicit_shared_credit,
     private_retention_weight, last_hit_epoch, owner.value}` 排序；
   - 主搜索：A* 式——pop queue（`FoldedCost.key()` 字典序），
     `session.assess(target)`，可行则 `logical_goal` 生成目标、更新 incumbent，
     否则扩展出新的压力目标（`mark_target` 去重；budget/queue/ledger 耗尽即停）；
   - **`FoldedCost` 排序键**（越靠前越优）：
     `{total_ns, affected_selected_hits↓, newest_affected_hit_epoch↑,
     owner_evictions, checkpoint_drops, copy_operations, transferred_bytes,
     remaining_text_prefill, remaining_vision_prefill,
     max_reused_prompt_tokens↓（用 uint32 max 反转）,
     !current_session_binding, candidate_ordinal, target_ordinal}`；
5. 产出 `Result{plan(密封的 ResourcePlan), candidate, publication_slot,
   source_mode, owner_outcomes, checkpoint_outcomes, diagnostics}`。

`MaterializationDiagnostics`（`include/ninfer/types.h`）：
`predicted_now_ns/predicted_future_loss_ns/predicted_total_ns/targets_evaluated/
projection_work/planning_elapsed_ns/search_elapsed_ns/stop_reason
{NoPressure,QueueExhausted,TargetBudget,ExpansionCapacity,TimeBudget,
ValueOfNextExpansion}/budget_exhausted/selected_degradation_units/
selected_maximal_fallback` + **`best_reuse_prompt_tokens`**——搜索实际评估过的
candidate 中最大的 `reusable_prompt_tokens`（由规划**上游**决定，与搜索选择无关）：
`0` 指向 prefix 匹配环节（无候选），大值伴随 root plan 则指向规划器（候选被
计价弃用）；2026-09-01/09-02 线上反复出现该歧义，故显式落字段。

### 7.3 `SharedCapturePlanner`（`src/runtime/engine/shared_capture_planner.h`）

`kTargetBudget=4096`；`Input{capture, owners, owner_policies, checkpoint_policies,
direct_shared_victim, candidate_demand_mask, candidate_rebuild_ns,
private_baseline_immediate_ns, blocked_runnable_requests, stable_scenario_ordinal,
target_budget}`；有界目标搜索后 `Result{plan, net_gain, …}`——
`net_gain` 是"捕获该共享前缀的净收益（未来重建节省 − 立即代价 − 对
active/blocked 请求的干扰）"，`reserve_active_capture` 在多 scenario 间取 max。

---

## 8. 调度与准入（`src/runtime/engine/scheduler.h` + `admission_policy.{h,cpp}`）

- `Scheduler<Request>`：FIFO `pending_`；`ExecutionAction{Prefill,Decode,Wait}`；
  **单条 staged prefill lane**（`set_prefill_lane/clear_prefill_lane`，重复设置抛异常）；
  `should_attempt_admission = have_pending && admission_check_pending &&
  !context_transaction && !prefill_lane && (!have_decode ||
  previous_unit_was_decode)`（prefill 进行中的空档不插准入）；
  `choose_execution`：prefill runnable 时，`have_decode && !prev_decode → Decode`
  （交织一轮回 decode），否则 `Prefill`；
  `build_round_membership`（decode 就绪 lane + per-lane budget，row_stride；
  缺 budget/sequence 抛异常）/ `build_control_membership`（uniform row_stride 的
  target control spans）；`active_admission_set`；`consume_service_work`
  （0 或超 remaining 抛异常）。
- 准入协议：`AdmissionGrant{request_id, backfill_class, protection_epoch,
  resource_revision, service_work_quanta}`（`grant_head` 绑定观察到的 FIFO head）；
  `AdmissionProtection{epoch_id, head_request_id, resource_revision,
  donor_ids[kMaximumConcurrency], donor_count}`（`make_admission_protection`
  冻结全部 active 为 donors；`rebind` 重新验证——当前 epoch 的 Persistent
  borrower 永不升为 donor）；`BackfillClass{None,Persistent}`；
  `persistent_backfill_is_authorized`（revision 必须匹配 + 存在 live donor +
  非 donor 必须为当前 epoch Persistent）；`protect_blocked_head`
  （创建/rebind 保护）；`qualify_backfill`；`commit_admission`
  （head grant 清 fifo_head + protection）；`observe_fifo_head/
  on_waiting_removed` 重置保护。
- `service_work_quanta` 来源：`RequestPlanSummary{prompt_tokens,
  reusable_prompt_tokens, requested/effective_output_tokens,
  effective_limit_reason, prefix_reuse_path, service_work_quanta,
  publish_continuation}`；target 层按 `PrefillWork`
  （`src/runtime/contract/types.h`）换算：
  `chunks = (suffix==0 || prefill_chunk==0) ? 0 : 1 + (suffix-1)/prefill_chunk`
  （suffix = prompt_tokens - reuse_base），
  `attention_pairs = prefix*suffix + suffix*(suffix+1)/2`
  （**饱和** u128→u64），再加 vision items/patches 成本。

---

## 9. Prefill 执行

### 9.1 引擎状态机（`src/runtime/engine/engine_core.h`，`request_record.h`）

`EngineRequestState{Waiting, Materializing, Prefill, DecodeReady, ControlReady,
ModelFinished}`（lane 侧 `LogicalLaneState` 见 §6.1）。`RequestRecord` 关键字段：
`prompt/output/prompt_summary/options/budget/admitted_begin(可选
BeginSummary{prompt_tokens,reused_prompt_tokens,prefix_reuse_path})/
begin(实际 BeginSummary)/lane/sequence/cancelled(atomic)/capture_pending/
post_capture_state/terminal_reason/base_plan(RequestBasePlan)/
remaining_service_work/backfill_epoch/class/retained_slot/
retained_session_digest/stream/events/result/RequestHostTiming`。

主循环每轮：`choose_execution` →
- **Prefill 分支**：`run_prefill_step`（`engine_core.h:1554`）——
  取 `scheduler_.prefill_lane()`（无则抛 `"no request owns staged prefill"`），
  校验 request 为 Prefill 且非 capture_pending，
  `instance_.program->advance_prefill(*sequence, &failed_timing)`
  （计时进 `prefill_seconds_total` / `ExecutionTimingRecorder`
  submit/wait/post 三相）→ `resolve_prefill_progress`；
- **Decode 分支**：`build_round_membership` 组装 compact batch →
  `program->run_round` → 行级结果分发（finish / capture / 转
  DecodeReady|ControlReady）。

### 9.2 准入 → 物化 → Prefill（`engine_core.h:1762+`、:1660+）

`admit_planned_request(request, choice, grant)`：

1. 过 deadline → `QueueTimeout` 错误出队；已取消 → detached cancel；
2. 校验 `grant`（request_id/service_work_quanta/`validate_grant`），否则抛
   `"admission choice lost its Scheduler grant"`；
3. 预分配 `GenerationBudget(effective_output_tokens, effective_limit_reason)`；
4. 构造 `MaterializingRequest{request, destination, budget, summary,
   backfill_class, protection_epoch, started}`；
5. `resources_.reserve_materialization(...)`（§6.3）。

`progress_context_transaction`（`engine_core.h:1685+`，每轮至多推进一个事务）：
`Materialization` 事务 → `resources_.progress_context_transaction` 返回
`MaterializationOutcome`：
- `Aborted` → 清 `materializing_`、`complete_detached_cancelled`；
- `Published` → 必须带 `activation`（否则抛
  `"published materialization has no adoption token"`）→
  `resources_.adopt(...)` → 填 `request->sequence/budget/lane/
  remaining_service_work/backfill_*/materialization_diagnostics`，
  `model_state = Prefill`，记 `queue_wait_ns`，`slots_[lane]=request`，
  `record_prefix_selection(control.summary)`（`engine_core.h:815`：按
  `summary.prefix_reuse_path` 累加 `root/private_endpoint/private_turn_closure/
  private_response_replay/private_long_anchor/shared_stable_prefix_selections` +
  `reused_prompt_tokens += summary.reused_prompt_tokens` +
  `last_selected_frontier_tokens`），
  `scheduler_.set_prefill_lane(lane)`。
`ActiveCapture` 事务 → `active_captures_completed/aborted` 计数，
`capture->capture_pending=false`，`model_state = post_capture_state`。

### 9.3 Prefill 逐步推进（target 层，`src/targets/qwen3_6/impl/runtime/program_impl.h:11369`）

`ProgramImplCore::advance_prefill(sequence, request, failed_timing)`
（`api_impl.h:415` 暴露为 `Program::advance_prefill(SequenceHandle, …)`）：

1. `staged = *request.prefill`（`RequestControl::Prefill{prompt, prompt_tokens,
   base(=reuse_base), reuse(=PrefixReusePath), cursor, capture_groups,
   next_capture, pending_capture_offer, prepare_mtp, mtp_bridge,
   initial_mtp_extent, vision, …}`）；pending capture offer 未清则抛
   `"prefill cannot advance while a capture offer is pending"`；
2. `BeginSummary{prompt_tokens, reused_prompt_tokens = staged.base,
   prefix_reuse_path = staged.reuse}`；
3. **零 prefill 捕获**：若下一 capture group 的 frontier == cursor == base 且非
   shared-base 提升（非 shared 或 rewrite 或 long_anchor）→ 抛
   `"zero-prefill capture is not a shared base promotion"`；否则置
   `pending_capture_offer`（`next_capture_offer_id_++`），本步只返回 summary
   （引擎侧 `resolve_prefill_progress` 里经 `reserve_active_capture` 走
   §6.4 流程）；
4. **MTP bridge（BeforeSuffix）**：`cursor==base && base>0 && cursor<prompt_tokens`
   （否则抛 `"staged MTP bridge is outside the reusable suffix"`）：用 source 的
   `sequence.tail_hidden` 作 previous_hidden、位置 `base-1`
   （`prompt_rope_position(prompt, base-1)`）跑
   `mtp_bridge_and_propose`（多模态走 `mtp_bridge_multimodal`），
   `sequence.mtp_kv_valid = staged.base`，`commit_sequence_kv`，bridge 置 None；
5. **chunk 循环**（`cursor < prompt_tokens`）：`nominal = min(prefill_chunk,
   prompt_tokens - cursor)`；workspace 按 `prepare_mtp? mtp_prefill : text_prefill`
   （+ DFlash 时 `dflash_context`）标记；
   - `materialize_sequence_kv(sequence, max(text_kv_valid, end), end)`
     （确保页容量覆盖本步终点）；
   - `state_selectors(sequence)` 选出 source/destination StateImage 槽
     （GDN 线性注意力状态的读/写位置，含 fork/restore 语义）；
   - `schedule_state`（`PrefillContext`：kv 视图、MTP 缓存、cursor、
     sampling 行、rewrite checkpoint hidden slot、state 槽、mtp extent）；
   - 调用 `TextContext::prefill*`（§9.4）跑**一个 chunk**（或 chunk 被
     capture/rewrite split 截断的前缀段）；
   - `sequence.text_kv_valid = staged.cursor`；`prepare_mtp` 时
     `mtp_kv_valid = cursor`；`commit_sequence_kv`；
6. **完成**：`text_kv_valid = prompt_tokens`（MTP 的 backend 同步到
   prompt_tokens，必要时补跑最终 chunk 的 MTP 对齐）；
   `prompt_frontier_capture`（capture group 恰在 prompt 末端）→ 发 capture offer
   并 `request.prefill.reset()`；否则 `request.prefill.reset()` +
   `PrefillStepResult{summary, round, processed_prompt_tokens, complete=true,
   pending(token), timing}`。

引擎侧 `resolve_prefill_progress`（`engine_core.h:1517+`）：
- capture offer → `reserve_active_capture`（Prefill 态中间捕获）；
- `complete` → 必须有 lane + pending token，且
  `progress.summary == *request->admitted_begin`（否则抛
  `"runtime Begin summary differs from committed admission"`）→
  清 prefill lane → `request->begin = progress.summary`（`GenerationStart`
  此刻下发给流式 sink）→ `commit_pending` 提交首 token。

### 9.4 单 chunk 计算（`src/targets/qwen3_6/impl/runtime/text_context_impl.h:1055`）

`TextContext::prefill`（`prefill_impl`）：

- 入参：`ids`（本 chunk token，绝对下标 `base` = `text_kv_base_`）、
  可选 `text_prefill`（整 prompt 对齐校验）、可选 `multimodal`
  （3×T positions + Vision session + rope_delta）；
  **"prefix-append prefill 继续已有 cache：positions 是绝对位置（从 resident
  长度起），KV/GDN 状态不重置；reset prefill base==0"**；
- `prefill_split_frontier_`：若 `(base, base+T]` 内有 split（capture frontier
  或 rewrite execution frontier），chunk 在 split 处截断（split 后继续下一 chunk）；
- 每 chunk：
  1. `visible = base + t0 + len`，
     `CausalAttentionExecutionEnvelope{visible, visible}` + scoped 注入
     （envelope 是本 chunk 可见 key 上界的 host 承诺）；
  2. embedding（+ vision scatter：`scatter_indices` 二分定位本 chunk 的视觉 patch）；
  3. `run_layers(x, Phase::Prefill, tap)` —— 每层内部：主 KV 层调
     `causal_softmax_attention`（append+attend 融合，§3）+
     `kv_cache_append` 语义由该融合算子承担；GDN 线性注意力层更新
     StateImage（source→destination 槽，`state_fork_required` 时先 fork）；
     sliding-window / MTP backend 层走对应 cyclic/context 算子；
  4. `rmsnorm` → 最终 chunk（`is_last`）：`lm_head` +
     `ops::sample(..., kSamplePurposePrefill, ...)`（`io_.pos = base+T`，
     使采样 RNG 与首 decode step 的 key 区分）得到 **bonus token**；
  5. MTP prompt（`prepare_mtp`）：按 `plan_mtp_alignment_window`
     构造 shifted mtp_ids（末列用刚采样的 token 时做 D2D copy），
     `mtp_prefill_chunk`；最终 chunk 且 `mtp_proposal_extent>0` 时先跑
     `mtp_prefill_chunk(..., draft0)` 再逐位 `mtp_forward_ar_step`
     （AR envelope `ar_visible = base+T+i`）出 draft 序列；
  6. split 恰在本 chunk 末端且配置了 `rewrite_checkpoint_hidden_output_`
     → 把 `xf` 末列 hidden 拷出（rewrite checkpoint 的 hidden 捕获）；
- **每轮 `advance_prefill` 只处理一个 chunk**（`t0 += len; break;`），
  从而 prefill 与 decode/其他请求交替、可被取消/抢占于 chunk 边界。

`prefill_chunk(full_ids, begin, nominal_length, finalize_at_end[, sink])` /
`prefill_chunk(input, begin, nominal_length, vision, finalize_at_end)` 是
`prefill_impl` 的三个公开包装（纯文本 / +DFlash sink / 多模态）。

### 9.5 成本与容量如何进入 prefill

- `PrefillWork` 的 `attention_pairs` 公式与 `rebuild_work_at_frontier`
  （§5.2）同源——"从 frontier 重建"的注意力对数 = `prefix*suffix +
  suffix*(suffix+1)/2`；
- `ContextPrefillCost` 把 `PrefillWork` 折算为 ns（§7.1），用于：
  candidate 的 `prefill_ns`（`price_materialization_machine_work`）、
  checkpoint 的 `rebuild_ns`、shared capture 的 `candidate_rebuild_ns`、
  split prefill 的增量计价（`shared_capture_split_prefill_work`，
  `api_impl.h:366`）；
- `kv_capacity`（§1.5）决定 page pool 规模；prefill 每步前
  `materialize_sequence_kv` 按终点 `end` 扩容（reservation→materialize）；
  `prefill_chunk` 与 target 的 `prefill_chunk_alignment`
  （`src/targets/qwen3_6/impl/runtime/instance.h:58`）对齐，
  `layouts_impl.h:271` 校验 `chunk = min(prefill_chunk, capacity)` 非 0。

---

## 10. 可观测性（`RuntimeStats`，`include/ninfer/types.h`）

- **prefix 选择**：`root/private_endpoint/private_turn_closure/
  private_response_replay/private_long_anchor/shared_stable_prefix_selections`、
  `reused_prompt_tokens`（累计）、`last_selected_frontier_tokens`；
- **物化/捕获**：`materializing_requests`、`capture_pending_requests`、
  `active_captures_completed/aborted`；
- **状态与 KV 转移**：`state_{moves,forks,restores}`、
  `state_{d2h,h2d,d2d}_{count,bytes,seconds}`、
  `main_kv_{d2h,h2d,d2d}_{pages,bytes,seconds}`、`backend_kv_*` 同构、
  `pressure_spill_pages`、`partial_tail_cow_pages`、`historical_fork_hits`、
  `actual_context_transfer_seconds`；
- **压力规划**：`pressure_{private,shared}_owners_{degraded,evicted}`、
  `pressure_checkpoints_dropped`、`pressure_searches`、
  `pressure_search_budget_exhaustions`、`pressure_maximal_fallback_selections`；
- **驻留**：`device_{state_occupied_slots, main_kv_occupied_pages,
  backend_kv_occupied_pages}`、`host_state_occupied_slots`、
  `host_kv_occupied_bytes`、`shared_active_references`；
- **吞吐/时延**：`computed_prefill_tokens`（**扣除**复用 checkpoint 前缀）、
  `committed_decode_tokens`（**不含** prefill bonus token）、
  `prefill_seconds_total/decode_seconds_total`、`decode_rounds/decode_row_rounds`、
  `running/prefilling/decode_ready/waiting/terminal_pending_requests`、
  `RuntimeHostWorkStats`（`prefill_host_ns/prefill_device_wait_ns/
  prefill_units` 等，单调纳秒计数）；
- per-request：`RequestHostTiming`（queue_wait、engine 各相、program
  submit/post/device_wait、decode host/device、prefill_units、decode_rounds、
  control_units）+ `GenerationTimings`（prepare/first_token/vision/prefill/
  decode/total 秒）+ `GenerationEngineTiming`（exposed 时延口径 +
  prefill_units/decode_rounds/control_units）+ `ExecutionTiming`
  （submit_host_ns/device_wait_ns/post_host_ns 三相）。

---

## 11. 关键调用链（端到端）

**A. 准入与物化**
```
Engine 主循环 should_attempt_admission
 → Scheduler::choose_execution / grant_head（AdmissionProtection 冻结 donors）
 → request->base_plan = program.plan_request(prompt, execution_options)
    [target: request_plan_impl 构建 capture groups / shared candidates /
     PreparedCaptureIdentity{rebuild_work, shortlist_key}（§5）]
 → resources_.inspect(program, prompt, base_plan, publication_order)
    [§6.2：root + prefix 候选 → plan_materialization →
     MaterializationPlanner::plan（§7.2）]
 → admit_planned_request（grant 校验 → GenerationBudget → MaterializingRequest）
 → resources_.reserve_materialization
    [observe_planned_evictions → reserve_logical → program.start_resource_transaction]
 →（后续轮）progress_context_transaction → Published
 → resources_.adopt → request.{sequence,budget,lane} = …，model_state=Prefill
 → record_prefix_selection（统计 + BeginSummary 已随 grant 记录为 admitted_begin）
 → scheduler_.set_prefill_lane(lane)
```

**B. Prefill 执行**
```
run_prefill_step（单 lane）
 → program.advance_prefill
    [BeginSummary{prompt_tokens, reused=staged.base, path}；
     零 prefill shared-base 捕获 offer；
     MTP bridge（BeforeSuffix，base-1 位置）；
     每轮一个 chunk：materialize_sequence_kv → state_selectors（GDN 槽）→
     TextContext::prefill（绝对 positions，causal envelope{visible,visible}，
     层内 causal_softmax_attention 融合 KV append，GDN 状态更新，
     末 chunk lm_head+sample bonus token，MTP prompt/AR，
     split 处捕获 rewrite checkpoint hidden）→
     text_kv_valid/mtp_kv_valid = cursor → commit_sequence_kv]
 → resolve_prefill_progress
    [capture offer → reserve_active_capture（§6.4）；
     complete → BeginSummary 一致性校验 → begin=summary（GenerationStart 下发）
     → commit_pending 提交 bonus token → 清 prefill lane]
 → request 转 DecodeReady（或 ControlReady）
```

**C. 活跃捕获（共享前缀发布）**
```
（decode 或 prefill 中 commit 的 CaptureOffer / 零 prefill offer）
 → reserve_active_capture（engine_core.h:1488：统计 blocked_runnable_requests）
 → resources_.reserve_active_capture（§6.4：baseline / replacement /
    exact_shared / scenarios × SharedCapturePlanner → max net_gain →
    program.reserve_active_capture_with_pressure）
 →（后续轮）progress_context_transaction → adopt_active_capture_progress
    [publishes_shared → publication{id,summary,handle,observation,
     explicit_credit, credit_expiry_epoch=demand_epoch+32, shared_sources edge}]
 → capture_pending=false，回 post_capture_state
```

**D. 终结与发布**
```
finish → TerminalPending
 → status==Consumed && Catalogued → publication.state=Catalogued、
   summary/handle/session/retention 迁移、advance_revision、publish_session
   （失败 → session.reset + retention=RecentPrivate）、lane Free
 →（session 有 slot 文件时）保留 retained_slot/session_digest →
   GenerationResult.slot / session_digest
 → status!=Consumed → abort + release（Released）
```

**E. 驱逐 / 自动保存**
```
pressure 规划选中 Eviction（§7.2 owner_outcomes）
 → observe_planned_evictions（eviction_observer：slot 文件回写，
   auto_save_evicted 时走后台 writer 线程，SlotAutoSaveEvent；
   slot_spill_guard 高水位拒绝更浅覆盖）
 → apply_private_action（Evicted → clear entry + notify_slot_released）
```

---

## 12. 关键参数速查

| 参数 | 默认 | 落点 |
|---|---|---|
| `max_concurrency` C | 1（上限 `kMaximumConcurrency=8`） | `EngineOptions` |
| `max_context` | 2048 | `EngineOptions` |
| `kv_capacity` | `explicit_capacity(2048)` | `KvCapacityPolicy` |
| `prefill_chunk` | 1024 | `EngineOptions` |
| `turn_checkpoint_ring` | 0（每 lane host 端 GDN 状态快照，~147 MiB @27B） | `EngineOptions` |
| `auto_save_evicted` / `auto_save_listener` | false | `EngineOptions` |
| `kv_cache` | `BFloat16` | `EngineOptions::kv_cache` |
| `use_cuda_graph` | true | `EngineOptions` |
| context cache | enabled=true；H=C、host 两层=0（opt-in）、P=2C、S=C、L=2 | `ContextCacheOptions` |
| `host_kv_capacity_bytes` / `host_state_slots` | 0 / 0 | `ContextCacheOptions` |
| `allow_prefix_reuse` | true（per-request 可关） | `ExecutionOptions` |
| `automatic_private_anchors` | 0 | `ContextCacheHints` |
| `allow_engine_automatic_shared_prefixes` | true | `ContextCacheHints` |
| `kPagedKVPageSize` | 64 | `src/core/paged_kv_cache.h:17` |
| `kCausalAttentionMaximumVisibleKeys` | 1048576 | `include/ninfer/ops/softmax_attention.h` |
| sliding-window | D=128/Hq=32/Hkv=8/window=4096 | `include/ninfer/ops/sliding_window_attention.h` |
| `kDemandWindowCapacity` | 32 | `resource_manager.h` |
| `MaterializationPlanner::kTargetBudget` | 4096 | `materialization_planner.h:650` |
| 搜索时间预算 | `min(5ms, incumbent.total_ns/20)` | `materialization_planner.h` |
| `kDefaultKvCapacityHeadroomBytes` | 1 GiB | `include/ninfer/types.h` |
| retention 权重 | Disposable=1 / RecentPrivate=4 / LiveSession=16 / SharedStable=0 | `resource_manager.h` |
| 编译默认硬件 | `nvidia-geforce-rtx-5090-sm120` | `context_cost_defaults.cpp` |

---

## 13. 文件索引

| 文件 | 角色 |
|---|---|
| `include/ninfer/types.h` | 公共类型：`KvCacheStorage`、`KvCapacityPolicy`、`ContextCacheOptions`、`EngineOptions`、`PromptCacheMarker(Hints)`、`SharedCandidateEvidence`、`PrefixReusePath`、`MaterializationDiagnostics`、`RuntimeStats`、slot API |
| `include/ninfer/engine.h` | Engine 公共门面（slots/requests） |
| `include/ninfer/ops/kv_cache_append.h` | KV 追加算子合同（lane / paged prefix / cyclic prefix） |
| `include/ninfer/ops/softmax_attention.h` | 注意力算子合同 + 数值 oracle |
| `include/ninfer/ops/sliding_window_attention.h` | 滑窗注意力合同 |
| `include/ninfer/ops/attention_geometry.h` | `AttentionHeadGeometry` |
| `include/ninfer/ops/prepare_ragged_prefix.h` | GDN 状态 ragged 打包合同 |
| `include/ninfer/ops/prepare_masked_block.h` | 掩码 block 准备 |
| `src/ops/kv_cache/append/` | `kernel.cuh`(33.7KB)/`launch.cu`/`kv_cache_append.cpp` + `fp8_e4m3_row_codec.cuh`/`int8_g64_codec.cuh`/`hadamard_d256.cuh`/`d256_profile.h` |
| `src/ops/softmax_attention/` | `common/{context_query.cuh, head_mapping.cuh}`、`dense/`、`context/`、`packed/`、`causal_cache/`（`prompt_*` bf16/fp8/i8、`small_t_*`、`geometry.cuh`、`causal_softmax_attention.cpp`） |
| `src/ops/sliding_window/`、`src/ops/sparse_moe/{prefill,decode,small_t}` | 滑窗 / MoE 前后向 |
| `src/core/paged_kv_cache.h` | 页池/租约/执行表/Host 副本（`kPagedKVPageSize=64`） |
| `src/core/cyclic_kv_cache.h` | 环形 KV（4096 槽） |
| `src/core/transfer_work.h` | `TransferWork` / `CoalescedTransferWork` |
| `src/runtime/contract/types.h` | `PrefillWork`、`ContextResourceClass`、`ContextTransfer*`、`Checkpoint*`、`RetentionClass`、`Readiness`、`RequestPlanSummary`、`Materialization*`、`SequenceCapacityCurve`、`ExecutionTiming` |
| `src/runtime/contract/sampling.{h,cpp}` | 采样合同（prefill bonus token 用） |
| `src/runtime/engine/resource_manager.h` | catalog/索引/事务/捕获（3518 行，§6） |
| `src/runtime/engine/materialization_planner.h` | 有界 A* 物化规划（1222 行，§7.2） |
| `src/runtime/engine/shared_capture_planner.h` | 共享前缀捕获规划（§7.3） |
| `src/runtime/engine/scheduler.h` | FIFO + 单 prefill lane + 准入（§8） |
| `src/runtime/engine/admission_policy.{h,cpp}` | 准入保护 / 回填授权（§8） |
| `src/runtime/engine/engine_core.h` / `engine.cpp` | 状态机 + 主循环（§9.1-9.3） |
| `src/runtime/engine/request_record.h` | 请求记录（§9.1） |
| `src/runtime/engine/context_cost.{h,cpp}` + `context_cost_defaults.cpp` | 成本模型（§7.1） |
| `src/runtime/engine/kv_capacity.cpp` | 容量解析（§1.5） |
| `src/runtime/engine/slot_spill_guard.h` | slot 文件高水位防覆盖（§6.7） |
| `src/runtime/engine/resource_search.h` | `BoundedTargetLedger`（§7.2） |
| `src/runtime/engine/causal_score_core.h` / `context_portfolio_value.h` / `public_types.cpp` / `generation/generation_budget.h` | 打分 / 组合价值 / 公共类型实现 / 输出预算 |
| `src/targets/qwen3_6/impl/runtime/request_plan_impl.h` | prefix 候选构建与 plan 后处理（§5） |
| `src/targets/qwen3_6/impl/runtime/program_impl.h` | `advance_prefill`（:11369）、KV commit/trim（:10770+）、`text_kv_view`（:10850）、状态事务 |
| `src/targets/qwen3_6/impl/runtime/text_context_impl.h` | 单 chunk 计算（:1055+，§9.4） |
| `src/targets/qwen3_6/impl/runtime/api_impl.h` | `Program` 公开 API（`inspect_admission`:313、`advance_prefill`:415、`checkpoint_recovery_work`:433/461、`shared_capture_matches`:467） |
| `src/targets/qwen3_6/impl/runtime/{layouts_impl,instance,session_snapshot_impl}.h` | 布局 / 常量 / 会话快照与恢复 |
| `src/targets/qwen3_6/impl/state/` | StateImage / GDN 状态 / slot 持久化实现 |
| `src/targets/qwen3_6/{_27b,_35b_a3b}` | 同 impl 的 Variant 特化 |
| `apps/{cli,perplexity,serve}` | 入口（serve 暴露 slots/captures 的 HTTP 面） |
