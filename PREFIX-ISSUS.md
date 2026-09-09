# PREFIX-ISSUS.md —— 实际使用问题归因分析

> 基线：分支 `prefix-v2`（HEAD `4f0faf0c`）。代码证据均给出 `文件:行号`；机制全景见本仓库 `KVCACHE.md`（§1 物理存储、§6 ResourceManager、§7 规划器、§8 调度准入、§9 prefill 执行）。
>
> 本文回答三个问题：
> 1. `max_tokens` 会提前预分配对应容量的 page，即使实际生成的 token 数到不了 `max_tokens`，也占用这么多 page；
> 2. 当 `max_tokens` 占用的 page 数过大时，会把当前已经产生的 prefix 直接清掉，导致模型每轮都触发全量 prefill；
> 3. 每发送一条消息 / 一个工具调用结果，都会迫使引擎跑一次（近）全量 prefill；预期应只有 prefix 变化点之后的内容需要 prefill。（用户后续澄清：并非指"上下文压缩改写历史"，而是指普通多轮追加场景——历史保持不变，只追加新消息。）
>
> 结论先行：**问题 1 是设计使然（容量确定性换利用率）**；**问题 2 是问题 1 与"host 层默认关闭"叠加的必然结果（压力只能靠驱逐解决）**；**问题 3 的引擎侧设计（endpoint 续延 + 模板自动生成 TurnClosure checkpoint）恰好实现了"只 prefill 变化点之后"——实测每轮（近）全量 prefill 是复用链在某处断掉的结果：thinking 模型下服务端默认 `preserve_thinking=false` 使历史响应的 `<think>` 块被丢弃，稳态退化为每轮重 prefill 上一轮响应（`PrivateTurnClosure`），叠加问题 2 的驱逐后升级为真正的 `Root` 全量 prefill**。

---

## 0. 三个问题共用的底层事实

### 0.1 页池容量由启动参数一次性决定

- 逻辑页大小固定为 64 token（`src/core/paged_kv_cache.h:17`，`kPagedKVPageSize=64`）。
- 页池总容量 = `kv_capacity`（Explicit 模式 `explicit_tokens`，必须 ≥ `max_context`；Automatic 模式按 `available_after_weights - headroom` 推导），见 `src/targets/qwen3_6/impl/runtime/layouts_impl.h:615-630` 与 `KVCACHE.md` §1.5。
- **host 层（D2H/H2D 搬迁目标）默认全部关闭**：`ContextCacheOptions.host_state_slots=0`、`host_kv_capacity_bytes=0`（`include/ninfer/types.h`）。这意味着压力处理**没有"搬到 host"这个选项，只剩驱逐（Evicted）或降级（丢 checkpoint）**。

### 0.2 一个请求的 KV 需求 = 整个 prompt + 整个 max_tokens（entitlement）

`src/targets/qwen3_6/impl/runtime/request_plan_impl.h`：

```cpp
// :257-270
capacity_output = capacity - prompt_tokens + 1;            // capacity = 引擎 max_context
effective_output_tokens = min(requested_output_tokens, capacity_output);  // 即 max_tokens 被 context 截断
reserved_context_tokens = prompt_tokens + (effective_output_tokens == 0 ? 0 : effective_output_tokens - 1);
text_kv_page_entitlement = pages_for_tokens(reserved_context_tokens);     // 页 = 1 + (tokens-1)/64
// MTP 后端另有 backend_kv_page_entitlement = pages_for_tokens(min(capacity, reserved + draft_window - 1))
// :278-282  root_active.main_kv_pages = text_kv_page_entitlement  ← 准入/压力规划的物理需求
```

**这个 entitlement 在准入时就作为候选的物理需求参与可行性判断，在物化采纳时被真正"占住"页池**（见 §1）。它与"本轮实际会生成多少 token"无关——`max_tokens` 给多大就预留多大。

### 0.3 前缀复用的匹配是逐 token 精确匹配

- 摘要（digest）只是**候选短名单**："One rolling digest per token frontier. This is only a content shortlist: exact token and ResidentPrefixIdentity comparison remains authoritative for reuse."（`src/targets/qwen3_6/impl/runtime/prefix_identity.h:57-59`）
- 权威判定 `prefix_matches`（`src/targets/qwen3_6/impl/runtime/prefix_identity.cpp:426-432`）：`[0, count)` 的 `token_ids` 逐一相等，**且** `ResidentPrefixIdentity`（token_types、3 条 position 轴、vision items）匹配。
- 复用起点只有几种可能（`PrefixReusePath`，`include/ninfer/types.h`）：
  - `PrivateEndpoint` / 私有续延：**从 position 0 开始**必须完全一致；
  - `PrivateLongAnchor`：从锚点 frontier 开始一致即可（部分复用）；
  - `PrivateTurnClosure` / `PrivateResponseReplay`：rewrite checkpoint 恢复——**frontier 之前**必须精确一致，frontier 之后可以改写；
  - `SharedStablePrefix`：共享前缀目录条目整体匹配。

任何改写 frontier 之前 token 的变化（上下文压缩/摘要、响应回传的再序列化失真、渲染选项漂移、max_context 截断）都会使上述所有路径失效。

### 0.4 压力驱逐的优先级（谁先被清）

- 私有 retention 权重（`ResourceManager::private_retention_weight`，`KVCACHE.md` §6.6）：`Disposable=1 < RecentPrivate=4 < LiveSession=16`（`SharedStable=0` 不参与私有驱逐）。
- retention 解析（`src/targets/qwen3_6/impl/frontend/frontend.cpp:735-739`）：`CacheRetentionHint::Default` → **有 `session_key` 时是 `LiveSession`，无 `session_key` 时是 `RecentPrivate`**；`LiveSession` 必须带 `session_key`（否则抛 `"LiveSession retention requires a session_key"`）。
- 共享前缀有 `explicit_credit` 机制，经 32 槽 demand window 衰减（`KVCACHE.md` §6.1/§6.6）：持续被复用的共享前缀保持信用，长期不命中的会被降权。

---

## 1. 问题一：max_tokens 提前预分配 page

### 1.1 机制（代码链路）

1. **规划期**：`plan_request` 按 §0.2 公式算出 `text_kv_page_entitlement`，写入 `root_active.main_kv_pages`（`request_plan_impl.h:270,281`）。即使走前缀复用（reuse_base>0），entitlement 也按**本轮完整新 prompt + max_tokens** 计算，而不是 `reuse_base + max_tokens`——被复用部分与已驻留页共享，但总容量口径不变。
2. **采纳期**：物化采纳时 `resize_sequence_kv_entitlement(sequence, text_kv_page_entitlement, backend_kv_page_entitlement)`（`src/targets/qwen3_6/impl/runtime/program_impl.h:9818-9819, 9866-9867, 10692-10701`）。
3. **页池层**：`prepare_activation`（`src/targets/qwen3_6/impl/runtime/logical_kv_store.h:905-930`）：
   ```cpp
   required_pages = pages_for_tokens(activation_frontier);          // 当前已驻留前沿所需
   growth = entitlement - required_pages;                            // 尚未物化的增长段
   required = missing_replicas + growth;
   pages_->physical_pool().resize_reservation(reservation, required); // 从空闲页池一次性扣掉
   ```
   即：**增长段（≈ `max_tokens` 对应的页）以 `DeviceKVPageReservation`（tail reservation）的形式立即从空闲页池扣除**，挂在序列地址空间上（`entitlement_` / `tail_reservation_`，`logical_kv_store.h:223-235`）。
4. **运行期**：页是**惰性物化**的——prefill 每 chunk、decode 每 step 只把 frontier 推进一步所需的页真正写满（`materialize_sequence_kv`，`KVCACHE.md` §9.3）。但 tail reservation 期间这些页对其他任何序列都不可用；越界会抛 `"KV materialization exceeds active entitlement"`（`logical_kv_store.h:1442`）。
5. **释放时机**：未用满的增长段**只在请求终结（finish 路径）时归还**：`release_sequence_growth_entitlement(state)`（`program_impl.h:9206`，把 entitlement 收缩到 `page_count`），随后解绑 KV、释放 lane。生成提前结束（stop/length/工具调用）不会在中途归还剩余 reservation。
6. **可观测**：`RuntimeStats` 的设备 KV 页数统计把整个 entitlement 都算进去（`device_pages += entitlement - mapped`，`program_impl.h:6723-6729`）——所以你从 stats 看到的"占用"就是 max_tokens 口径，与"真实写满的页"不同口径。

### 1.2 定性

- **这不是 bug，是容量确定性设计**：为"一次请求生成中途绝不因页池耗尽而失败"付出的代价。页池是启动期固定大小（无运行时扩缩），若按需增长，并发场景下谁先拿到页谁活，规划器（`KVCACHE.md` §7）就失去了"物理可行性"这个硬约束。
- **放大因素**：`max_tokens` 接近 `max_context - prompt_tokens` 时，单个请求的 entitlement ≈ 整个页池（`create_active` 甚至要求 `entitlement ≤ page_capacity_`，`logical_kv_store.h:864`）。此时池子里**没有任何余量**留给：已 catalog 的 prefix 驻留、并发请求、MTP backend KV。

### 1.3 缓解建议

| 措施 | 说明 |
|---|---|
| **把 `max_tokens` 设成真实上界** | 收益最直接：entitlement 随 `max_tokens` 线性缩小。长文任务用分段调用而不是一个巨大的 `max_tokens`。 |
| 用 Automatic 容量模式 + 合理 headroom | `KvCapacityPolicy{Automatic, automatic_headroom_bytes}`（默认 headroom 1GiB，`kDefaultKvCapacityHeadroomBytes`）让池子按显存实际余量推导，避免"显式 tokens 远小于 prompt+max_tokens"的结构性不可行。 |
| 降低 `max_concurrency` | 每个并发请求都带一份完整 entitlement；并发 × 大 max_tokens 是池耗尽的乘数。 |
| 用 stats 校准 | 观察 `RuntimeStats` 的 `device_main_kv_pages`（含增长段）与 `main_kv_pages` 实际物化量的差值，量化"预留而没用"的规模，再决定 `max_tokens` 口径。 |

---

## 2. 问题二：max_tokens 过大 → 已产生的 prefix 被清掉 → 每轮全量 prefill

### 2.1 机制（为什么 prefix 会被清）

单轮多轮会话的时序是：`start`（准入+物化）→ prefill/decode → `finish`（续延 **Catalogued** 进目录，KV 页继续驻留）→ 下一轮 `start`。下一轮请求 = 旧 prompt + 旧生成 + 新用户消息，引擎本应复用上一轮 catalog 的私有续延（`PrivateEndpoint`），把 prefill 限制在新增部分。但在页池紧张时链路变成：

1. **准入时**新请求的 root 候选物理需求 = 新 prompt 全跨度 + `max_tokens` 的 entitlement（§0.2/§1.1），加上目录中已驻留 prefix owner 的页，**超过空闲页**。
2. **压力规划**（`MaterializationPlanner` 有界 A*，`KVCACHE.md` §7.2）：host 层关闭（§0.1）→ 没有 D2H 转移目标 → 规划器只能在 **驱逐 owner（`VictimDisposition::Evicted`，清目录条目+释放页，`KVCACHE.md` §6.5）** 或 **降级（丢 checkpoint）** 之间找可行解。
3. 规划器按成本（`KVCACHE.md` §7.2 的 FoldedCost 排序键：驱逐/降级代价 vs 省下的 prefill vs 传输代价）选出"驱逐某个 prefix owner 后走 Root/复用"的 plan；被驱逐的 owner 页被 `clear_catalog_entry` 释放（触发 slot release observer，`KVCACHE.md` §6.7）。
4. 本轮若落到 **`Root`**（`reuse_base=0`）→ **全量 prefill**。`finish` 时新一轮续延重新 Catalogued。
5. 下一轮：池子依然紧张（新 prompt 更大 + `max_tokens` 不变 + 新 owner 驻留）→ **再次驱逐** → 再次全量 prefill。这就是"每轮都触发全量 prefill"的循环。

### 2.2 为什么默认配置下特别容易发生

- **无 `session_key` 时默认 retention 是 `RecentPrivate`（权重 4）**，低于 `LiveSession`（16）。多个会话混跑或共享前缀混存时，普通会话的 prefix 是**最先被牺牲**的（`frontend.cpp:735-739`；权重表见 `KVCACHE.md` §6.6）。
- **`auto_save_evicted=false` 且无 host 层**：被驱逐的 prefix 直接蒸发（不落盘、不搬 host），下一轮想恢复只能整体重 prefill。
- **prompt 逐轮增长**：`capacity_output = max_context - prompt_tokens + 1` 逐轮变小，但 `reserved_context_tokens = prompt + max_tokens - 1` 逐轮变大——张力随会话变长单调恶化，前几轮能共存，十几轮后必然进入每轮驱逐状态。
- 共享前缀同理：`explicit_credit` 经 32 槽 demand window 衰减（`KVCACHE.md` §6.6），不持续命中的共享 prefix 先被清。

### 2.3 缓解建议（按优先级）

1. **压低 `max_tokens`**（治本，见 §1.3）——让"prefix 驻留 + 新轮 entitlement"在池子内共存。
2. **给会话传 `session_key`**（或显式 `retention: LiveSession`）：`Default` 有 `session_key` 即升为 `LiveSession`（权重 16），压力规划会优先保住你的会话，牺牲 Disposable/RecentPrivate 的别人。
3. **开启 host 层**（`ContextCacheOptions.host_kv_capacity_bytes > 0`、`host_state_slots > 0`）：压力处理从"驱逐"升级为"D2H 转移"——prefix 搬到 host 内存驻留，下轮 H2D 换回（成本 = 传输 ns，`KVCACHE.md` §7.1），不再清空。4090 上显存紧、主机内存通常富余，这是性价比最高的结构性修复。
4. **扩池**：`kv_capacity.explicit_tokens` 提高到 ≥ 常驻 prefix 总量 + 最坏单请求 entitlement；或 Automatic 模式减小 headroom。
5. **兜底**：`auto_save_evicted=true` + slot 目录，被驱逐的 slot 落盘、可恢复（恢复=完整 H2D，慢于 host 驻留，好过重 prefill 的算力成本）。
6. **验证手段**：
   - 每个 `GenerationResult.prefix_reuse_path`：持续是 `Root` 而 `reused_prompt_tokens=0` → 问题 2 成立；
   - `RuntimeStats`：`pressure_private_owners_evicted` / `pressure_shared_owners_evicted` / `pressure_checkpoints_dropped` / `pressure_spill_pages`（`KVCACHE.md` §10）——非零即发生过压力驱逐/降级；
   - `GenerationResult.materialization`（`MaterializationDiagnostics`）：`stop_reason`、`best_reuse_prompt_tokens`（>0 说明有候选但被规划器放弃，=0 说明根本没有可复用候选）。

---

## 3. 问题三：每发一条消息 / 工具结果就触发（近）全量 prefill

### 3.1 问题重述（用户澄清后的语义）

> "只要我发送一条消息，或者是工具调用结果发送，都会迫使引擎跑一次全量的 prefill。预期应该是只有在 prefix 的变化点之后的内容才需要 prefill，而实际上是每轮对话输入内容后，都会强制执行 prefill。"

即：多轮追加场景（历史不变、只加新消息/工具结果）下，本应复用上一轮的 KV、只对新增 suffix 做 prefill，实测却每轮都付出接近全量的 prefill 代价。

### 3.2 引擎侧设计：正是用户期望的复用链（设计没有缺口）

- **v22_4 模板自动生成 TurnClosure checkpoint**：每次 chat 渲染都会放置一个 rewrite checkpoint（`RewriteCheckpointKind::TurnClosure`），位于**历史末尾、`<|im_start|>assistant` 生成前缀之前**（`src/targets/qwen3_6/impl/frontend/chat_template.cpp:878-891`）。模板注释明确这就是为多轮场景设计的（`:879-883`："An immediate successor may replay the response, close the turn, or branch by appending a different user message... The rolling private checkpoint must therefore precede the assistant opener"）；若 `preserve_thinking` 开启则为 `ResponseReplay`（`:886-887`）。
- **finish 时发布 endpoint**：`CheckpointRef{kind=SessionEndpoint, frontier=sequence.execution_frontier}`（`src/targets/qwen3_6/impl/runtime/program_impl.h:7276-7284`）——frontier 是**整个 ledger 末尾（prompt + 本轮采样响应的全部 token）**，随 ContinuationSummary 一起 Catalogued（`KVCACHE.md` §6.5）。
- **稳态时序**（记第 N 轮 ledger = 历史渲染 H_N + 采样响应 R_N）：
  - 第 N 轮 catalog = { endpoint E_N（H_N+R_N 末尾）, TurnClosure R_N（H_N 末尾）}；
  - 第 N+1 轮 prompt = H_N + R_N（客户端回传渲染）+ 新消息 + 生成前缀；
  - **客户端把响应逐字节回传（含 thinking）** → 命中 E_N → **prefill 只有新消息**（= 用户期望）；
  - **响应回传有损** → E_N 失配，退而命中 R_N（H_N 部分客户端渲染是确定性的、自洽的）→ prefill = 上一轮响应 + 新消息（部分，**不是全量**）。

所以"引擎强制全量 prefill"并非引擎行为——引擎每轮产出的是上面两级复用中的某一级；用户观察到的"（近）全量"来自下面的断点。

### 3.3 断点清单（按嫌疑度排序）

**B1 响应回传失真 —— thinking 模型的默认配置陷阱（头号嫌疑）**

- 复用要求新 prompt 与旧 ledger **逐 token 一致**（§0.3）。旧 ledger 里的响应是引擎**实际采样**的 token，含完整 `<think>...</think>` 块（thinking 模型）；
- 模板对历史 assistant 消息是否渲染 `<think>` 块由 `keep_thinking = preserve_thinking || (i > last_query_index)` 决定（`chat_template.cpp:844`，`<think>` 追加在 `:855-860`）；历史消息（`i <= last_query_index`）只在 `preserve_thinking=true` 时保留 think 块——**即使客户端把 `reasoning_content` 原样回传也照样被丢弃**；
- 服务端 `preserve_thinking` 默认 **false**，需 `--preserve-thinking` 显式打开（`src/serve/serve_options.cpp:346-347`），或按请求传 `preserve_thinking` / `chat_template_kwargs`（`src/serve/translate.cpp:117`：请求值 > 服务端值）；
- 后果（thinking 模型 + 默认配置）：每轮 E_N 必然失配 → 稳态 = **每轮 `PrivateTurnClosure`，重 prefill 上一轮完整响应（含 thinking）+ 新消息**。长 thinking 响应的 prefill 量与"全量"观感无异；
- 其他回传失真源（同样打穿 E_N，但 R_N 兜底仍在）：**tool_calls 再序列化**——`render_tool_call` 按结构化字段渲染（`chat_template.cpp:863-873`），客户端回传的 arguments JSON 若在空白/键序/数值格式上与引擎采样文本不同即失配；content 被 trim/归一化/摘要同理；
- 协议通道是齐备的：OpenAI assistant 消息接受 `reasoning_content`/`reasoning`（`src/serve/openai_chat_request.cpp:424-457` 解析、`:535` 映射；冲突校验 `"conflicting assistant reasoning and reasoning_content values"`），响应回传 `reasoning_content`（`src/serve/openai_chat_response.cpp:196-197`）；Anthropic 协议同（`src/serve/anthropic_messages_request.cpp:189,332`）；`turn.reasoning_content → message.reasoning_content`（`src/serve/translate.cpp:189`）。**通道在，但两端都要做对**：服务端开 `preserve_thinking`，客户端回传 `reasoning_content`。

**B2 压力驱逐（问题 2 叠加）——从"近全量"升级为真 `Root`**

- 即使匹配命中，上一轮续延的 KV 页若被页池压力驱逐（§2 机制：host 层关 + max_tokens entitlement + 无 `session_key` 时 `RecentPrivate` 权重低），catalog 条目消失 → 本轮无任何候选 → `Root` 全量 prefill；
- 与 B1 的区分：B2 的轮次里 `materialization.best_reuse_prompt_tokens > 0`（候选存在但被规划器放弃）；B1 单独作用时 path 是 `PrivateTurnClosure` 而非 `Root`。

**B3 轮次未正常终结**

- 续延只在 `finish()` 路径 Catalogued（`KVCACHE.md` §6.5）；轮次被取消（客户端 abort）或超时（`EngineOptions.pending_timeout_ms` 默认 30000）→ 未发布 → 下一轮无从复用。

**B4 渲染选项跨轮漂移**

- `enable_thinking`、`preserve_thinking`、reasoning effort、模板版本等任何跨轮变化 → 同一段历史的渲染 token 改变 → 连 R_N 的自洽性也被打破 → 真 `Root`。会话全程保持选项恒定。

**B5 max_context 截断**

- 历史增长超过 `max_context` 时：`encode_rendered_chat(tokenizer, rendered, max_context + 1)`（`src/targets/qwen3_6/impl/frontend/frontend.cpp:1434`）截断，或直接抛 `context_length_exceeded`（`frontend.cpp:1439`）；截断后 digest 全变 → 复用永久失效，此后每轮全量。长 agent 会话里这是最"突然"的断点（`prompt_tokens` 顶在上限）。

**B6 响应尾 token 失配（只降级、不到全量）**

- 采样响应若以裸 EOS 结束而非 `<|im_end|>`，E_N 差 1~2 个 token → 只丢 endpoint，R_N 仍可用 → 表现为"每轮重 prefill 上一轮响应"，不是全量。

### 3.4 为什么"一次失真会污染整个会话"（归纳放大）

所有复用 frontier（除 LongAnchor/Shared 外）都要求**从 position 0 精确一致**。任何一段历史响应渲染不一致，后续每一轮的所有复用起点都失效；且新轮 catalog 的 ledger 继承旧轮前缀——缓存**不会自我修复**。因此观测到"每轮全量"后，应优先排查协议回传的字节稳定性与配置漂移，而不是怀疑缓存机制本身。

### 3.5 排查决策表（判断当前处于哪种断点）

| 观测（每轮） | 结论 |
|---|---|
| `prefix_reuse_path=PrivateEndpoint`，`reused_prompt_tokens`≈历史长度 | 正常，引擎按预期工作 |
| `prefix_reuse_path=PrivateTurnClosure` 稳定复现，reused≈历史末尾 | **B1**：响应回传失真（thinking 未保留 / tool_calls 再序列化）→ 每轮重 prefill 上一轮响应 |
| `prefix_reuse_path=Root` 且 `materialization.best_reuse_prompt_tokens=0` | 无候选 → **B1(深度)/B4/B5**：匹配阶段就断了（选项漂移、截断、历史改写） |
| `prefix_reuse_path=Root` 且 `best_reuse_prompt_tokens>0` | 候选存在但被压力规划放弃 → **B2**（§2） |
| `prompt_tokens` 顶在 `max_context` 附近 | **B5** 截断 |
| 上一轮被 cancel/超时 | **B3** |

（字段来源：`GenerationResult.prefix_reuse_path` / `reused_prompt_tokens` / `materialization`，`RuntimeStats.pressure_*`，`KVCACHE.md` §10。）

### 3.6 缓解建议（按优先级）

1. **响应逐字节回传**（治 B1 根因）：客户端把引擎返回的 `content`、`reasoning_content`、`tool_calls` **原样**放回下一轮 assistant 消息（最稳妥是保留 assistant 消息的原始文本/结构再发送）；不要 trim/归一化/摘要历史；
2. **服务端开 `--preserve-thinking`**（或按请求 `preserve_thinking: true`）：thinking 模型下这是让历史响应的 `<think>` 块进入渲染、从而命中 endpoint 的**必要条件**——默认关闭时 thinking 会话结构性地只能走到 `PrivateTurnClosure`；
3. **渲染选项跨轮恒定**（治 B4）：`enable_thinking`/effort/`preserve_thinking`/模板版本会话全程不变；
4. **会话控制在 `max_context` 内**（治 B5）：为 agent 工作集留足；确需截断时接受一次性重建并冻结此后历史；
5. **不打断轮次**（治 B3）：依赖复用的下一轮之前保证上一轮正常 finish；`pending_timeout_ms` 给足；
6. **叠加 §2 防线**（治 B2）：`session_key`（`LiveSession`）+ 开 host 层 + 收敛 `max_tokens`——回传再忠实，新建的缓存下一轮被驱逐照样回到 `Root`；
7. **客户端确实无法保证字节级忠实时**（第三方协议有损再序列化）：用 `SharedStablePrefix` marker 固定稳定头 + `automatic_private_anchors=N` 固定稳定尾（`include/ninfer/types.h:466-479`，上限 `max_long_anchors_per_continuation` 默认 2），把每轮重 prefill 范围限制在不稳定的中段；rewrite checkpoint / `turn_checkpoint_ring`（`layouts_impl.h:688-689`，DFlash 强制 0）适用于"头稳、近期尾段每轮改写"形态。

### 3.7 小结

- 引擎的多轮复用链（endpoint 覆盖 prompt+响应 + 模板自动 TurnClosure 在历史末尾）**在机制上完整实现了用户期望的"只 prefill 变化点之后"**；
- 实测"每轮（近）全量 prefill"的标准解释：**thinking 模型 + 服务端默认 `preserve_thinking=false` + 客户端未回传 `reasoning_content`（或 tool_calls 再序列化有损）→ 每轮 `PrivateTurnClosure` 重 prefill 上一轮响应**；再叠加页池压力驱逐（问题 2）/ 截断 / 打断 / 选项漂移 → 升级为真正的 `Root` 全量 prefill。
---

## 4. 三问题的关系与排查顺序

- **问题 1 是根因放大器**：entitlement 按 `prompt + max_tokens` 全额预留（§1），直接制造池张力；
- **问题 2 是张力在默认配置（host 层关、无 session_key 时 retention 偏低）下的兑现形式**：prefix 被压力驱逐，循环全量 prefill（§2）；
- **问题 3 是复用链断链问题**：引擎侧多轮复用设计（endpoint + 模板自动 TurnClosure）本就实现"只 prefill 变化点之后"；实测每轮（近）全量 prefill 来自响应回传失真（thinking/tool_calls，默认 `preserve_thinking=false`）、渲染选项漂移、`max_context` 截断、轮次打断或压力驱逐等断点（§3.3）。
- 实际环境里三者常叠加：回传失真（B1）或压力驱逐（B2）打断复用后，下一轮全量 prefill 重建的缓存又可能再次被池张力（问题 1+2）驱逐，形成"每轮全量"的观感。

**建议排查顺序**：先看 `prefix_reuse_path` 分布（`PrivateTurnClosure` 稳定出现 = 回传失真 B1；`Root` + `best_reuse_prompt_tokens=0` = 匹配断链；`Root` + `best_reuse>0` = 压力放弃）与 `materialization` 诊断（`KVCACHE.md` §10）→ 再看 `pressure_*` 驱逐统计（确认问题 2/B2 是否在发生）→ 然后核对协议层响应回传是否字节级忠实（`reasoning_content`/`tool_calls` 原样回传）、渲染选项跨轮是否恒定、`prompt_tokens` 是否顶到 `max_context`（§3.5 决策表）。最后按 §1.3 / §2.3 / §3.6 的措施组合落地：`max_tokens` 收敛 + `session_key`/`LiveSession` + host 层开启 + `--preserve-thinking` + 响应字节级回传（必要时加 marker/anchors 约束重 prefill 范围）。

## 5. 证据索引

| 事实 | 位置 |
|---|---|
| `pages_for_tokens = 1 + (tokens-1)/64` | `src/targets/qwen3_6/impl/runtime/request_plan_impl.h:44-46` |
| `capacity_output` / `effective_output_tokens` / `reserved_context_tokens` / entitlement 公式 | `src/targets/qwen3_6/impl/runtime/request_plan_impl.h:254-276` |
| root 候选物理需求 = 完整 entitlement | `src/targets/qwen3_6/impl/runtime/request_plan_impl.h:278-282` |
| 页池 `prepare_activation`：`growth = entitlement - required_pages`，`resize_reservation` 一次性扣减 | `src/targets/qwen3_6/impl/runtime/logical_kv_store.h:905-930` |
| `resize_entitlement` / `release_growth_entitlement`（`"KV materialization exceeds active entitlement"`） | `src/targets/qwen3_6/impl/runtime/logical_kv_store.h:1423-1442` |
| 物化采纳时 resize entitlement | `src/targets/qwen3_6/impl/runtime/program_impl.h:9818-9819, 9866-9867, 10692-10701` |
| 增长段只在 finish 路径归还 | `src/targets/qwen3_6/impl/runtime/program_impl.h:9206` |
| stats 把整段 entitlement 计入设备 KV 页 | `src/targets/qwen3_6/impl/runtime/program_impl.h:6723-6729` |
| 页池容量来自 `max_context`/`kv_capacity`（显式须 ≥ `max_context`） | `src/targets/qwen3_6/impl/runtime/layouts_impl.h:606-630, 775-779` |
| retention 解析：`Default` → 有 `session_key` ? `LiveSession` : `RecentPrivate` | `src/targets/qwen3_6/impl/frontend/frontend.cpp:735-739` |
| `CacheRetentionHint{Default,LiveSession,Disposable}` 与 `ContextCacheHints` 默认值 | `include/ninfer/types.h:403-408, 456-479` |
| digest 仅为短名单；精确 token+identity 匹配才权威 | `src/targets/qwen3_6/impl/runtime/prefix_identity.h:57-59` |
| `prefix_matches`：token_ids 逐一相等 + `ResidentPrefixIdentity` 匹配 | `src/targets/qwen3_6/impl/runtime/prefix_identity.cpp:426-432` |
| rewrite checkpoint（`TurnClosure`/`ResponseReplay`）与 `rewrite_execution_frontiers` | `src/targets/qwen3_6/export/ninfer/targets/qwen3_6/prepared_prompt.h:70-90` |
| `can_retain_rewrite_checkpoint`（frontier 之前精确一致才可续） | `src/targets/qwen3_6/impl/runtime/program_impl.h:6938-6950` |
| rewrite checkpoint hidden 截取 | `src/targets/qwen3_6/impl/runtime/text_context_impl.h:1290-1294` |
| `turn_checkpoint_ring`（DFlash 强制 0） | `src/targets/qwen3_6/impl/runtime/layouts_impl.h:688-689` |
| v22_4 模板自动生成 TurnClosure（历史末尾/assistant 前缀前）与 ResponseReplay；设计注释 | `src/targets/qwen3_6/impl/frontend/chat_template.cpp:878-891` |
| `keep_thinking = preserve_thinking || (i > last_query_index)`；`<think>` 块条件渲染 | `src/targets/qwen3_6/impl/frontend/chat_template.cpp:844, 855-860` |
| 历史 assistant 消息：`reasoning_content` 回传 / `derive_think_parts` 兜底；`render_tool_call` | `src/targets/qwen3_6/impl/frontend/chat_template.cpp:833-843, 863-873` |
| endpoint frontier = `sequence.execution_frontier`（含响应 token 的 ledger 末尾） | `src/targets/qwen3_6/impl/runtime/program_impl.h:7276-7284` |
| serve `--preserve-thinking`（默认 false）、`--no-prefix-reuse` | `src/serve/serve_options.cpp:340-347` |
| `preserve_thinking` 解析：请求值 > 服务端值 | `src/serve/translate.cpp:117` |
| OpenAI：assistant `reasoning_content`/`reasoning` 解析回传；响应回传 `reasoning_content` | `src/serve/openai_chat_request.cpp:424-457, 535`；`src/serve/openai_chat_response.cpp:196-197` |
| `turn.reasoning_content` → chat message | `src/serve/translate.cpp:189` |
| `identity.reusable=true`（chat 路径）与 `context_cache` 组装 | `src/targets/qwen3_6/impl/frontend/frontend.cpp:1450-1453` |
| `max_context` 截断 / `context_length_exceeded` | `src/targets/qwen3_6/impl/frontend/frontend.cpp:1434-1439` |
| 压力规划/驱逐/retention 权重/demand window/统计字段 | `KVCACHE.md` §6、§7、§10 |
