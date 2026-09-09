# PREFIX-PLAN.md 可行性审查报告

> 审查对象：`PREFIX-PLAN.md`（298 行，§0–§10，三支柱：动态 KV 页 / prefix 硬保护 + root gating / StateImage 槽解冻）
> 基线：prefix-v2 @ `4f0faf0c`。本报告中所有 `文件:行号` 均为对当前分支的**实测**行号（与方案引用不一致处在 §4 列出）。
> 审查维度：① 需求满足度（三目标）② 破坏性变更 ③ 新 bug 风险 ④ 实施可行性（落点 / API 面 / 协议）。
> 方法：方案中每条代码断言逐条对照仓库源码核验；静态审查，不含编译 / 运行验证（编译在 Docker `ninfer-local-build:latest`，GPU 验证需 4090 容器 `ninfer-4090-kaso-dev`，跑前须先请用户关闭常驻推理容器）。

---

## 1. 结论总览

**判定：条件通过（可行）。** 三支柱的全部改动落点在基线中真实存在，新增 API 面窄（5 个函数 + 1 个 span 字段 + 1 个结构字段），协议零新增；**完成 M1–M7 修正（§5）后即可满足三目标，且不引入引擎级破坏性变更**。其中 M1 / M2 / M5 是"不引入引擎级回归"的**实施前置**——未修正直接实施，会分别复现引擎整体停摆（恰是方案要消灭的故障模式）、成功→失败回归（QueueTimeout 替代 root 全量 prefill）、prefix 静默丢失（违反目标 2）。

### 1.1 需求满足度

| 目标（用户原话） | 判定 | 依据 |
|---|---|---|
| 1. max_tokens 只是最大值的约束，不做强制性 page 占用，动态申请，申请失败 = 上下文窗口到极限 = 正常逻辑 | ✅ 可达（M1/M5/M6 后） | `entitlement = pages(prompt)`（`request_plan_impl.h:259-264` 改 `reserved_context_tokens`→`prompt_tokens` 即可）+ 增长闸门改 `KvGrowth{Ok,PoolExhausted,InvalidTarget}`（`logical_kv_store.h:1442` 落点实测成立）+ 池耗尽 → 请求级 `ContextCapacity` stall 链（复用 `mark_terminal_pending` `resource_manager.h:928-932` + 既有 `settle_terminal_requests` 链，实测全链成立）→ 引擎继续服务 |
| 2. 渲染结果与 prefix 匹配时无论如何不得丢 prefix（除非一开始就不匹配才全量 prefill） | ⚠️ 可达（M2/M3 后） | 硬保护 owner 集 P（`build_pressure_inputs` `resource_manager.h:1967-2100` 落点成立）+ root gating + `note_pending_demand` 均可落地；**但方案 §3.3 root gating 谓词过强（F-2）**：幽灵匹配（key 匹配但物理副本已合法丢失）下"永不选 root"会死等到 `pending_timeout`（serve 默认 **600s**，`serve_options.h:33-34`）→ `QueueTimeout` 失败，而基线反而 root 全量 prefill 成功——M2 修正后才真正满足 |
| 3. 所有 page 按需分配，prefix 稳定必复用，禁止 prefix 匹配时触发全量 prefill | ✅ 可达（M7 后） | 机制 A/B/C 全部骑在基线**已存在**的管道上：`state_slots==2` 绑定模式已在 `program_impl.h:9474-9575` 实现、`checkpoint_references`（`state_image_store.h:209-211`）/ `can_recycle_checkpoint_destination`（`program_impl.h:7474-7477`）/ skip 分支 `!pressure && !physically_feasible`（`program_impl.h:7822-7826`）均为现成落点 |

### 1.2 破坏性变更清单

**有意（目标本意，4 项）**
1. 池耗尽：引擎 `fail_all`（worker 永久停摆，`engine_core.h:2073-2105`，注释 :268 明言 permanent）→ 请求级 `ContextCapacity` 正常终结（目标 1 本意）。
2. root gating：存在匹配候选时 root 由"立即全量 prefill"变为 `TemporarilyBlocked` 等待（目标 2/3 本意）。
3. entitlement 由全程跨度（`prompt+max_tokens-1`）收缩为 `pages(prompt)`：准入检查变松 → 高负载下并发更满（有界，无泄漏；峰值页占用仍由页池容量封顶）。
4. root 请求 `state_slots=1→2`（`request_plan_impl.h:272` 一处）：root run 激活期状态槽池可用数 −1（M7 定义争用 = 请求级拒绝）。

**副作用（2 项，文档化即可）**
5. E1 reason 覆盖：池先枯竭时 `OutputLimit` 被覆盖为 `ContextCapacity`（`preview_terminal` 用 stall 自身 reason）——Anthropic 协议要求区分两类上限，语义**正确**。
6. E4 统计口径：`device_pages` 由整段 entitlement 计（`program_impl.h:6723-6729`）改为按 mapped 计——可观测性口径变化，无行为影响。

**意外（1 项，M2 消除）**
7. **F-2**（方案 §3.3 过强）：`best_offered_reuse > 0`（`materialization_planner.h:72-80`，注释明言"reuse was ON THE TABLE"，与物化可行性无关）只证明"逻辑上曾匹配"。幽灵匹配场景下压力搜索结构性不可行 → 方案"永不选 root" → 死等 600s → `QueueTimeout`：基线成功（慢）→ 方案失败，是唯一一处**成功→失败**的行为回归。

### 1.3 新 bug 风险（全部有消除手段，详见 §5/§6）

| ID | 风险 | 严重度 | 消除 |
|---|---|---|---|
| R-a | `mark_capacity_stalled` 前置条件漏 `!capture_pending` → `settle_terminal_requests` 所有权校验（`engine_core.h:1128-1133`）throw `logic_error` → worker `catch(...)`（:2185-2187）→ `fail_all_locked`（:2073）引擎永久停摆 | **最致命** | M1 |
| R-b | `resources_.finish` 返回非 Consumed → 自动 fallback `program.abort` → `clear_catalog_entry` → 该请求自身 endpoint 目录条目丢失 → 下轮 prefix 匹配断裂（stalled 请求带 `fork_pending` 且 `abort_fork` 失败时触发，`program_impl.h:9149-9219`） | 高（违反目标 2） | M5 |
| R-c | F-2：root gating 过强 → `QueueTimeout` 替代 root 全量 prefill | 中高 | M2 |
| R-d | E3（prefill 期枯竭）欠规约：`advance_prefill` 仅有 `PrefillProgress`/异常通道（`engine_core.h:1554-1601`），engine `catch(...)` 无法区分容量与 bug | 低（实测结构性自消） | M4 |
| R-e | demand window（32 槽，`resource_manager.h:1423`）：pending 需求 > 窗口时最旧被静默丢弃 → 保护降级；`commit_demand`（:2457-2480）的 `explicit_credit` 清除侧效应只看最新记录，请求 B 的登记会误清请求 A 的匹配 credit；`PrefixDemandRecord`（:122-126）无 request-id → 定点出窗不可能 | 中（降级而非失败） | M3 |
| R-f | E2 行内增长预检 throw → `fail_all` | **刻意设计**（暴露 bug 而非静默跳过） | 文档化 |

---

## 2. 三支柱逐条核验

### 2.1 支柱一：动态 KV 页（§1–§2）

| # | 方案断言 | 代码证据（实测） | 判定 |
|---|---|---|---|
| 1 | 增长闸门在 `logical_kv_store.h:1438-1446` | :1442 `target < address.page_count \|\| target > entitlement(address)` → `throw std::invalid_argument("KV materialization exceeds active entitlement")`；:1417-1469 `materialize_to_tokens` 全貌（闸门后 `pages_->materialize(reservation, added, predecessor)`，publish 有 try/catch 回滚 `dematerialize`） | ✅ |
| 2 | 增长原语 `resize_entitlement` 可直接复用 | **已存在** `logical_kv_store.h:1414-1425`：`entitlement < address.page_count \|\| > page_capacity_` → `invalid_argument`，否则 `pages_->physical_pool().resize_reservation(address.reservation, entitlement - address.page_count)`；另有 `release_growth_entitlement`（:1427-1435，resize 到 0） | ✅（比方案预期更顺） |
| 3 | `prepare_activation` / `create_active` 通道 | `prepare_activation`（:905-945）：`required = missing_replicas + growth` → `resize_reservation(reservation, required)`；`create_active`（:860-880）：`entitlement==0 \|\| > page_capacity_` → nullopt，否则激活异常透传 | ⚠️ nullopt 通道**只覆盖静态容量**（地址空间）；运行期池耗尽是 `throw std::bad_alloc`（`paged_kv_cache.cpp:268`，`paged_kv_cache.h:226-228` 声明）→ E3 / 准入拒绝路径须按 `can_resize_reservation` 预检或 `bad_alloc` 捕获处理（§4 小误 4） |
| 4 | `probe_decode_capacity` 新 API | `can_resize_reservation(reservation, new_pages) noexcept → bool`（`paged_kv_cache.h:226-228`；实现 :268 区：`used = allocated + (reserved - reservation.pages) + new ≤ capacity`）；`reserve(pages)` noexcept 返 nullopt；`available_pages() = capacity - allocated - reserved` → probe 可**无异常**实现 | ✅ |
| 5 | entitlement 计算改 `pages(prompt)` | `request_plan_impl.h:252-264`：`capacity_output = capacity - prompt_tokens + 1`；`effective_output_tokens = min(requested, capacity_output)`；`reserved_context_tokens = prompt + (effective==0 ? 0 : effective-1)`（:257-260）；`text_kv_page_entitlement = pages_for_tokens(reserved)`（:262）→ 改为 `pages_for_tokens(prompt_tokens)` 即方案值；`pages_for_tokens = 1 + (tokens-1)/64`（:44-46，`kPagedKVPageSize=64` `paged_kv_cache.h:17`） | ✅ |
| 6 | MTP / DFlash backend entitlement | MTP = `pages_for_tokens(min(capacity, reserved + draft_window - 1))`（:266-268）；**DFlash 基线 = `pages_for_tokens(reserved_context_tokens)`（:270-271），不是方案 §2.1/§2.2 所写的 `pages_for_tokens(prompt_tokens)`** | ⚠️（§4 小误 3；方案新值 `pages_for_tokens(prompt)` 不受影响） |
| 7 | `FinishReason::ContextCapacity` 与协议映射 | `include/ninfer/types.h:555`（方案引 :552）；映射实测齐备：`src/serve/openai_chat_response.cpp:46` → `"length"`、`openai_responses_response.cpp:22`、`anthropic_messages_response.cpp:64` → `"model_context_window_exceeded"` | ✅ |
| 8 | `GenerationBudget` 语义 | `generation_budget.h:11-45`：构造 `(effective_tokens, limit_reason)`，`limit_reason ∉ {OutputLimit, ContextCapacity}` → `std::abort()`；`commit(tokens)` 若 `tokens > remaining` → `std::abort()`；`budget.commit(remaining)` 清零安全 | ✅ |
| 9 | stall 终结链 | `engine_core.h:1110-1170` `settle_terminal_requests` 实测全链：事务一致性（不一致 → throw `"Engine and Program disagree before terminal settlement"`；program 有事务 → 返回 false 延到下边界）→ 按 `publication_order` 最小选 lane → 所有权校验（`is_model_finished() && !capture_pending && sequence && lane 匹配 && lane_state==TerminalPending`，否则 throw `"terminal-pending request has invalid ownership"` :1128-1133）→ `resources_.finish`（Catalogued 时记 `retained_slot` + `retained_session_digest` :1145-1153）→ `complete_success(request, reason)`（:1021 对任意 reason 通用，自动携带 `reused_prompt_tokens`/`prefix_reuse_path`）→ `remove_completed_slot`（:1091，内部 :1093 调 `request_admission_check` → 排队者立即补位） | ✅ 链完整成立 |
| 10 | `note_capacity_stall` 落点 | `mark_terminal_pending(LaneId)` **已存在**（`resource_manager.h:928-932`，Active→TerminalPending）→ 新 API = 复用 + 统计计数 | ✅ |
| 11 | `mark_capacity_stalled` 前置条件 | 方案列 `is_model_finished / sequence / lane`，**漏 `!capture_pending`**（settle 校验含它，`engine_core.h:1129`）。结构性保护（decode-ready 判定排除 `capture_pending`，:2021 区）仅在 probe 与 membership 构建**同一边界**时成立 | ❌ M1 修正 |
| 12 | 池耗尽 → finish 保 Catalogued | `resources_.finish`（`resource_manager.h:932-1006` 区）要求 TerminalPending + 无事务，否则 `throw logic_error("terminal finish overlaps an open resource transaction")`；**非 Consumed → 自动 fallback `program.abort` → `clear_catalog_entry`**（丢该请求 endpoint 条目）。`ProgramImplCore::finish`（`program_impl.h:9149-9219`）前置：`has_context_transaction() \|\| pending_transaction_ \|\| !valid_sequence` → 空；**`lifecycle != Lifecycle::Finishable` → 空**；`!publish_continuation` → Released（Consumed，不 catalog，合法）；catalog 路径：`fork_pending` → `abort_fork` + `release(destination)`（失败 → 非 Consumed）；`role(state.state.read)` 必须 ActiveMutable(→freeze) 或 CheckpointImmutable；`release_sequence_growth_entitlement`（:9206 确认在 finish 内）；`unbind_sequence_kv`；`lifecycle=Empty`。stalled 请求须镜像 terminal commit 的两项迁移：`lifecycle = Finishable` + `pending = {}`（基线 `program_impl.h:12231`） | ⚠️ M5（stall 专用 settle 分支断言 Consumed） |
| 13 | E3（prefill 期枯竭同 stall 语义） | `run_prefill_step`（`engine_core.h:1554-1601`）：`advance_prefill` 异常 → worker catch → `fail_all`（基线行为）。**关键实测**：新 `entitlement = pages(prompt)` 后 prefill 只从**激活期已预留**页（`tail_reservation_`，`logical_kv_store.h:223-235` 持有、他者不可取）物化 → prefill 点池枯竭**结构性不可能** | ⚠️ M4（改为"结构性自消 + 防御 throw"） |
| 14 | E2 行内预检 | 行内 `can_resize_reservation` 预检失败 throw → `fail_all`——暴露 bug 而非静默跳过，**刻意设计**（与 I3 不变量一致） | ✅（文档化） |
| 15 | E4 统计 | `program_impl.h:6723-6729`：`device_pages += entitlement - mapped`（整段计）→ 动态页下高估，改按 mapped 计正确；增长点 :11740（ordinary 每步）/ :11889（MTP）/ :12076（DFlash）；`materialize_sequence_kv` :10755 | ✅ |
| 16 | 零预算 lane 必须排除出 membership | `"ordinary batch row is not decode-ready"` 实测 `program_impl.h:11695`（方案 :11688）；decode 就绪校验含 `generated_tokens_remaining == 0` → 进 membership 必 throw → `fail_all` | ✅（并入 M1） |

### 2.2 支柱二：prefix 硬保护 + root gating（§3）

| # | 方案断言 | 代码证据（实测） | 判定 |
|---|---|---|---|
| 1 | 容量核算 8+16=24≤32 | `kMaximumConcurrency = 8`（`types.h:20`）；serve 默认 `max_pending_requests = 16`、`pending_timeout_ms = 600000`（`src/serve/serve_options.h:33-34`）；`kDemandWindowCapacity = 32U`（`resource_manager.h:1423`）→ 核算成立。⚠️ 方案 / PREFIX-ISSUS 记的 30000 是**部署值**，非 serve 默认；600s 默认会放大 F-2 最坏等待 | ⚠️（§4 小误 5） |
| 2 | 压力目标空间禁 Evicted / 丢 P 中 owner checkpoint，host 开时允许 D2H | `build_pressure_inputs`（`resource_manager.h:1967-2100`）：private 条目须 Catalogued && handle && `!private_has_active_edge`（active edge 的 owner 本就整 run 保护）；checkpoint policy 覆盖 endpoint + rewrite + long_anchors，各带 `demand_mask = demand_mask_for(key, provisional_demand)`；shared 条目单 checkpoint、`retention_class=SharedStable`、`explicit_shared_credit`；`plan()` 调用点 :2201。`PressureInputs`（`materialization_planner.h:85-94`）= `{private_owners, private_owner_ids, shared_owners, shared_owner_ids, owner_policy, checkpoint_policy}` → 落点 = 加 protected 标记 / `protected_owners` span + 目标扩展剪枝 + 快速路径守卫。D2H 降级通道基线已存在（residency :1474-1572；HostSnapshot 落位 `program_impl.h:7543-7550`） | ✅ |
| 3 | 保护 owner 集 P 由候选 key 集 K 推导 | inspect 阶段 `resource_manager.h:295-330` 构建 `provisional_demand`（`exact_resident_keys` 来自 `prefix_index_`）+ 候选集（含 root）→ K = 候选 key 集；P = {private 条目：endpoint/rewrite/long_anchors key ∈ K} ∪ {shared 条目：`shortlist_key` ∈ K}——与 :1967-2100 的条目/键结构一一对应 | ✅ |
| 4 | root gating：`best_offered_reuse > 0` → 禁选 root | `best_offered_reuse(candidates)`（`materialization_planner.h:72-80` 静态）= 各候选 `summary().reusable_prompt_tokens` 的 max；`plan()` 模板（:100-354）：identity 快速路径 `seal_identity(FinalScheduleIntent)` 可直接让 root 胜出（**须同样加守卫**）；否则 bounded A*（`search_budget_ns = min(5'000'000, incumbent.total_ns/20)` :187-188）；落点 = 选后规则（`result.candidate == root && best_offered_reuse > 0` → nullopt → `TemporarilyBlocked` 既有路径）+ 快速路径守卫 | ⚠️ 落点成立，**谓词过强 → F-2（M2）** |
| 5 | **F-2**：谓词语义 | `best_offered_reuse > 0` 只表"逻辑上曾匹配"。幽灵匹配场景：key 仍匹配但物理副本已合法丢失（无设备页、host 关、无可驱逐 owner）→ 压力搜索后结构性不可行 → root 唯一可行 → 方案"永不选 root" → 死等到 `pending_timeout`（**serve 默认 600s**）→ `QueueTimeout` 失败；基线反而 root 全量 prefill 成功。planner 诊断可用：`observe_planner_diagnostics`（`resource_manager.h:2482-2497`，`stop_reason` / `budget_exhausted` / `selected_maximal_fallback`）可区分"预算耗尽（可进展）"与"结构性死亡" | ❌ M2 修正（方案 §3.3 是唯一"意外"破坏性变更来源） |
| 6 | `note_pending_demand` 复用 demand window | `commit_demand`（`resource_manager.h:2457-2480`）：`capacity < 32` → `std::terminate()`；满 → `erase` 最旧（**pending 需求 > 窗口时最旧被静默丢 → 保护降级**，须文档化运维约束 `max_pending_requests + active ≤ 32`）；随后有 `explicit_credit` 清除侧效应（只看 `demand_window_.back()` 最新记录 → 请求 B 的登记会误清请求 A 的匹配 credit）→ `note_pending_demand` 须为**纯窗口插入**、跳过该侧效应 | ⚠️ M3 |
| 7 | `PrefixDemandRecord` 结构 / "零结构改动" | `resource_manager.h:122-126` = `{candidate_keys, exact_resident_keys, selected_source_key}`——**无 request-id 字段** → 取消 / 超时 / 终结时定点出窗不可能；"零结构改动"仅在 lazy expiry（保护随窗口滑动自然过期、可超出请求生命周期，轻度放宽 I5）语义下成立 | ⚠️ M3（加 owner 字段最小改动，或显式文档化 lazy 语义） |
| 8 | §3.5 decode 轮后补 `request_admission_check` | 盲区**实测确认存在**：`should_attempt_admission`（`scheduler.h:233-237`）= `have_pending && admission_check_pending && !context_transaction && !prefill_lane_ && (!have_decode \|\| previous_unit_was_decode)`；worker 每轮 :2133-2134 消费 flag、:2156-2163 评估；`request_admission_check()`（:910-911）仅置 atomic flag（:2219）；现有触发点 :245/875/884/890/982/1093/1230/1545/1687/1714/1732/1749/1756/1776/1793/1802——**`run_decode_round` 完成后无 setter**（其尾部 :2050-2066 = budget commit + append_output + `model_state=DecodeReady` + stats）→ flag 已消费后 `TemporarilyBlocked` 者最长等 `pending_timeout`。实现 = `run_decode_round` 尾部加一次 `request_admission_check()`（一行；既有 gate 含 `previous_unit_was_decode=true` 已能放行）——比方案措辞改动更小 | ✅ |

### 2.3 支柱三：StateImage 槽解冻（§4）

| # | 方案断言 | 代码证据（实测） | 判定 |
|---|---|---|---|
| 1 | 机制 C 七项校验原语 | `state_image_store.h`：`StateImageRole{Free, ActiveMutable, CheckpointImmutable, ReservedDestination}`（:69-74）；`checkpoint_references(StateImageHandle)`（:209-211）；`can_release_after_checkpoint_references`（:686）→ `!has_context_transaction` / slot<capacity / Catalogued / generation 匹配 / `kind==LongAnchor` / 锚点在条目册 / `checkpoint_references(anchor.state)==1` 全部可用现成原语 | ✅ |
| 2 | 机制 A（root `state_slots=2`） | **基线已具备 state_slots==2 管道**：`SequenceState::reserved_state`（`program.h:409`）；`program_impl.h:9474-9575` 激活/事务代码已处理 `state_slots==2`（`reserved_states[0]`→active read/write 绑定、`reserved_states[1]`→`sequence.reserved_state`）；`finish()` 释放 `reserved_state`（:9180-9182）；:9823/:9871 `reserve_state_entitlement`。方案改动仅 `request_plan_impl.h:272`（`root_active.state_slots = 1U` → `2U`） | ✅（直接骑在既有管道上） |
| 3 | 机制 A 细节：capture 目的序 `prepared→recycled→reserved_state→new` | 基线 `device_destination_available = recycles_private_state \|\| (state_store->device_occupied() - replaced_shared.device.state_slots < state_store->device_capacity())`（`program_impl.h:7539-7545`）——**无 reserved_state 分支** → 须在 placement 选择加该分支；reserved 槽在激活期已计入 active entitlement（`state_slots=2` 计入 `optional_resources`）→ **无新增 peak**，`physically_feasible = physical_peak_fits(demand.physical_peak_additional)`（:7605-7607；`fits_u32` 含 device/host state_slots）公式只需 OR 一个条件。无设备空槽且有 host 层 → HostSnapshot（`added.host.state_slots=1` :7550，transfer DeviceToHost） | ⚠️ 落点真实，placement 分支列入实施项 |
| 4 | 机制 B（锚点图像回收为 fork 目的） | 基线 `recycles_private_state` 模式已存在：`can_recycle_checkpoint_destination`（:7474-7477，仅 `rewrite_state` 可回收）；drops 校验 `checkpoint_references(drop.state) != count` → 跳过（:7487-7490）；`install_private_capture`（:7974）/ `prepare_active_capture`（:8041）命名 API 真实存在。扩展 = 把回收模式扩到 long-anchor 图像 + 新不变量"引用迁移后 `checkpoint_references` 恒 ≥ 1"（可测试） | ✅ |
| 5 | 机制 C skip 分支改造 | `if (!pressure && !assessment.physically_feasible) { skip_capture; return Aborted; }`（`program_impl.h:7822-7826`，方案引 :7823 正确；即 prefill_v1.md §6 锚点槽冻结根因实测点）→ "先释放被支配源锚点再重评估一次"落点 `reserve_active_capture_impl`，释放原语 = `checkpoint_references==1` + `release` + catalog 侧刷新 | ✅ |
| 6 | M7（新增发现）：root `state_slots=2` 槽争用 | 状态槽池紧张时（如池=2、两个并发 root run），第二个 root 激活期 `reserve_state_entitlement` 失败 → 须定义为**请求级拒绝**（与目标 1"申请失败 = 正常逻辑"语义一致），不得阻塞引擎 / 不得 `fail_all` | ⚠️ 方案须补此句 |

---

## 3. stall 终结链完整走查（M1 后形态）

1. **探测**：membership 构建点（`engine_core.h:2152` 区，`build_round_membership` **之前**）对每个在跑 lane 调 `probe_decode_capacity`（`can_resize_reservation` 无异常版）。
2. **标记**：`probe == PoolExhausted` → `mark_capacity_stalled(lane)`：前置校验 `is_model_finished 候选（model_state 迁移）+ sequence + lane + !capture_pending + 无开放事务`（防御不变量，违反 → 视为 bug throw，**不得**静默）→ 镜像 terminal commit 两项迁移（`lifecycle = Finishable`、`pending = {}`，基线 :12231）→ `budget->commit(remaining)` 清零 → `preview_terminal(ContextCapacity)` → `note_capacity_stall`（= 复用 `mark_terminal_pending` `resource_manager.h:928-932` + 统计）→ `model_state = ModelFinished`。
3. **排除**：stall lane 从本轮及后续 membership 排除（零预算进 decode membership → `program_impl.h:11695` throw → `fail_all`）。
4. **终结**：下一边界 `settle_terminal_requests`（:1110-1170）选 lane → **stall 专用分支**：`resources_.finish` 必须返回 Consumed（Catalogued 或 Released），否则 `throw logic_error` 暴露（**不 fallback abort**——abort → `clear_catalog_entry` 丢该请求 prefix，违反目标 2；与 E2 风格一致：bug 走 `fail_all` 暴露而非静默）→ `complete_success`（携带 `reused_prompt_tokens` / `prefix_reuse_path`）→ `remove_completed_slot`（:1091 → :1093 `request_admission_check`，排队者立即补位）。
5. **线程安全**：所有触碰 program/resources 的引擎入口均取 `execution_mutex_`（`engine_core.h:251/283/302/317/361/395`）；worker_loop 跨 try 块持锁（含 `run_decode_round` / `run_prefill_step` / settle）；auto-save 写线程同锁（:2226-2227 注释）→ 页池 / 槽池计数器单线程（worker）+ 锁序列化 → `probe_decode_capacity` 无竞态 ✅。

---

## 4. 方案小误清单（不影响可行性，随 M 修正一并订正）

| # | 方案引用 | 实测 | 位置 |
|---|---|---|---|
| 1 | `FinishReason::ContextCapacity` `types.h:552` | `include/ninfer/types.h:555` | §2.3（方案 :59） |
| 2 | `"ordinary batch row is not decode-ready"` `program_impl.h:11688` | `program_impl.h:11695` | §1.4 |
| 3 | DFlash 基线 entitlement = `pages_for_tokens(prompt_tokens)` | 基线 = `pages_for_tokens(reserved_context_tokens)`（`request_plan_impl.h:270-271`） | §2.1/§2.2 |
| 4 | "运行期池耗尽走 `create_active` 返回 nullopt 通道" | nullopt 仅覆盖静态容量（`entitlement > page_capacity_`，`logical_kv_store.h:860-880`）；运行期池耗尽是 `throw std::bad_alloc`（`paged_kv_cache.cpp:268`）→ 须 `can_resize_reservation` 预检或异常捕获 | §2.2 |
| 5 | `pending_timeout_ms` "默认 30000" | serve 默认 `600000`（`src/serve/serve_options.h:33-34`）；30000 为部署配置值 | §3.5 / R2 |
| 6 | §2.2 limit_reason 预计算"`pages_for_tokens(prompt+max_tokens-1) ≤ 页池容量` → OutputLimit，否则 ContextCapacity；`max_tokens==0` → OutputLimit" | 基线规则（`request_plan_impl.h:254-256`）= `requested_output_tokens <= capacity_output ? OutputLimit : ContextCapacity`（**输出预算 vs 上下文窗口**的比较，与页池无关）。方案规则的两处偏差：(a) 用页池容量代替上下文窗口——span 装得下页池 ≠ 装得下上下文窗口，会错发 OutputLimit（Anthropic 协议要求区分两类上限，E1 覆盖只兜池枯竭，兜不了这种"预计算就错"的情形）；(b) `max_tokens==0`（无显式上限）→ OutputLimit 与基线 `ContextCapacity` 相反。**预计算 reason 只用于"预算自然耗尽"，池维度全由 stall 覆盖** | §2.2 → M6 |
| 7 | §3.4 "现有 `demand_window_` 结构零改动" | 仅在 lazy expiry 语义下成立（`PrefixDemandRecord` `resource_manager.h:122-126` 无 owner 字段） | §3.4 → M3 |

---

## 5. 必须修正 M1–M7（实施前置）

- **M1**（消除 R-a，最致命）：`mark_capacity_stalled` 前置条件 += `!capture_pending` + 无开放事务（作为 API 防御不变量，违反 → bug throw）；probe/stall 必须在**首次** `build_round_membership`（`engine_core.h:2152` 区）之前执行；stall lane 从本轮及后续 membership 排除（零预算进 membership → `:11695` throw → `fail_all`）。
- **M2**（消除 R-c / F-2）：root gating 谓词细化——**仅当匹配候选 feasible 或"可通过进展变得 feasible"**（等页 / 等释放；用 planner diagnostics `stop_reason` / `budget_exhausted`，`resource_manager.h:2482-2497` 可辨）时禁选 root；**结构性死亡**（前缀物理已丢：无副本、无可驱逐 owner——root 全量不再多丢任何东西）→ 放行 root；快速路径 `seal_identity` 同样加守卫。保留 `pending_timeout` 作最终兜底（运维侧注意 serve 默认 600s）。
- **M3**（消除 R-e）：(a) `PrefixDemandRecord` 加 owner（request id）字段（最小结构改动）以支持定点出窗，或显式文档化 lazy expiry 语义（保护可超出请求生命周期，I5 放宽说明）；(b) `note_pending_demand` 为纯窗口插入，**不触发** `commit_demand` 的 `explicit_credit` 清除侧效应（否则请求 B 的登记误清请求 A 的匹配 credit）；(c) 文档化运维约束 `max_pending_requests + active ≤ 32`（窗口溢出静默丢最旧）。
- **M4**（消除 R-d）：E3 改为——`entitlement = pages(prompt)` 后 prefill 只从激活期已预留页（`tail_reservation_` `logical_kv_store.h:223-235`）物化，**结构性自消**；prefill 路径增长失败保留为防御 throw（bug 语义走 `fail_all`，与 E2 一致）；P0 **不**为 prefill lane 加 stall 信号通道（`advance_prefill` 仅有 `PrefillProgress` / 异常通道，engine `catch(...)` 无法区分容量与 bug）。
- **M5**（消除 R-b）：stall 专用 settle 分支：`resources_.finish` 非 Consumed → `throw logic_error` 暴露（**不 fallback `program.abort`**，否则 `clear_catalog_entry` 丢该请求 endpoint 目录条目 → 下轮 prefix 匹配断裂，违反目标 2）；单测必须覆盖 stalled + `fork_pending` 组合（`abort_fork` 失败路径）。
- **M6**（小误 6）：limit_reason 预计算保留基线语义（`request_plan_impl.h:254-256`：`requested <= capacity_output ? OutputLimit : ContextCapacity`），不与页池容量纠缠；池维度全由 stall 覆盖（E1）。
- **M7**（新增）：root `state_slots=2` 槽争用（激活期 `reserve_state_entitlement` 失败）= **请求级拒绝**（nullopt / 异常通道，与目标 1 语义一致），不得阻塞引擎 / 不得 `fail_all`；文档化最坏情形（状态槽池=2 + 两个并发 root run → 第二个被拒）。

---

## 6. 风险汇总表（含缓解与阶段归属）

| ID | 风险 | 严重度 | 缓解 | 阶段 |
|---|---|---|---|---|
| R-a | `capture_pending` 交互 → settle throw → `fail_all` 停摆 | **最致命** | M1 | P0（与支柱一同批） |
| R-b | finish 非 Consumed → abort 回退 → prefix 丢失 | 高 | M5 + 单测 | P0 |
| R-c | F-2 root gating 过强 → `QueueTimeout` 替代 root prefill（成功→失败） | 中高 | M2 | P1（与支柱二同批） |
| R-d | E3 prefill stall 欠规约 | 低（结构性自消） | M4 | P0（文档 + 防御 throw） |
| R-e | demand 窗溢出 / 误清 credit / lazy 出窗 | 中（降级非失败） | M3 + 运维文档 | P1 |
| R-f | E2 行内预检 `fail_all` | 刻意设计（暴露 bug） | 文档化 | P0 |
| R-4 | root `state_slots=2` 占槽 → 极端槽池下第二个 root 被拒 | 低 | M7（请求级拒绝语义） | P2（与支柱三同批） |

阶段归属与方案 §6 一致（P0 = 支柱一 + §3.5 decode 轮重准入；P1 = 支柱二；P2 = 支柱三），M1/M5/M6 随 P0、M2/M3 随 P1、M7 随 P2、M4 为 P0 文档项。

---

## 7. 判定

1. **需求**：目标 1 ✅（M1/M5/M6 后：池耗尽 → 请求级 `ContextCapacity` 正常终结，引擎持续服务，`fail_all` 仅保留在刻意的 bug 暴露路径）；目标 2 ⚠️→✅（M2/M3 后：硬保护集 P + 细化 root gating + pending 保护全部落地，"匹配不得丢 prefix" 成立；唯一反例幽灵匹配经 M2 放行 root 全量，与基线行为一致）；目标 3 ✅（M7 后：三机制全部骑在既有管道上，槽争用语义明确）。
2. **破坏性变更**：4 项有意（池耗尽语义 / root gating 等待 / 准入变松 / 槽占用）+ 2 项副作用（E1 语义正确、E4 口径变化）+ 1 项意外（F-2，M2 消除）。**未发现其他破坏面**；协议零新增（`ContextCapacity` 三家映射实测齐备），参数零新增（方案 §8 成立）。
3. **新 bug 风险**：6 项，全部有消除手段（M1–M5）或为刻意设计（R-f）。最致命的 R-a 在 API 前置条件 + membership 规则层面消除（改动量小），**不需要**改动引擎错误处理架构。
4. **实施可行性**：全部改动落点实测存在；新 API 面 = `probe_decode_capacity` + `mark_capacity_stalled`（`note_capacity_stall` 复用 `mark_terminal_pending`）+ `drop_superseded_anchor` + `protected_owners` span + `note_pending_demand`（+ `PrefixDemandRecord.owner`），均为窄接口；编译验证在 Docker `ninfer-local-build:latest`（仓库挂 `/src`、`build/` 复用、Ninja），GPU 验证在 4090 容器（跑前须先请用户关闭常驻推理容器）。
5. **最终判定：可行（条件通过）**——方案的方向、落点、机制全部正确；**完成 M1–M7 并落实方案 §4.3 八类不变量测试表 + §6 分阶段回归（P0 的 ①超池 max_tokens 请求正常终结且引擎不停摆 ②A 占大池 → B stall 终结 → 引擎持续 ③双协议 finish_reason 一致性）后即可进入实施**。

---

## 附录 A：本审查实测证据索引

| 证据 | 位置 |
|---|---|
| 增长闸门 / `resize_entitlement` / `materialize_to_tokens` / `prepare_activation` / `create_active` / `tail_reservation_` | `src/targets/qwen3_6/impl/runtime/logical_kv_store.h:223-235, 860-880, 905-945, 1414-1435, 1417-1469（闸门 :1442）` |
| `can_resize_reservation` / `resize_reservation`（bad_alloc）/ `reserve` / `available_pages` / `kPagedKVPageSize=64` | `src/core/paged_kv_cache.h:17, 226-228` + `src/core/paged_kv_cache.cpp:268` |
| entitlement / limit_reason / MTP / DFlash / `root_active.state_slots=1U` | `src/targets/qwen3_6/impl/runtime/request_plan_impl.h:44-46, 252-272` |
| `FinishReason::ContextCapacity`；`kMaximumConcurrency=8` | `include/ninfer/types.h:555, 20` |
| 协议映射 | `src/serve/openai_chat_response.cpp:46`、`src/serve/openai_responses_response.cpp:22`、`src/serve/anthropic_messages_response.cpp:64` |
| `GenerationBudget` abort 语义 | `src/runtime/generation/generation_budget.h:11-45` |
| serve 默认值 | `src/serve/serve_options.h:33-34`（`max_pending_requests=16`、`pending_timeout_ms=600000`） |
| worker_loop / `fail_all_locked` / `settle_terminal_requests` / `cancel_active_requests` / `remove_completed_slot` / `complete_success` / admission flag | `src/runtime/engine/engine_core.h:2073-2105, 2106-2204（catch :2185-2187）, 1110-1170, 1167-1191, 1091-1094, 1021` |
| `is_model_finished` / `capture_pending` 置位清除 | `src/runtime/engine/request_record.h:146-148`；`engine_core.h:1512, 1730, 2021, 2163` |
| `should_attempt_admission` / `request_admission_check` | `src/runtime/scheduler/scheduler.h:233-237`；`engine_core.h:910-911, 2133-2134, 2156-2163` |
| `mark_terminal_pending` / `resources_.finish` / `PrefixDemandRecord` / `kDemandWindowCapacity` / `commit_demand` / `observe_planner_diagnostics` / `build_pressure_inputs` | `src/runtime/resource/resource_manager.h:928-932, 932-1006, 122-126, 1423, 2457-2480, 2482-2497, 1967-2100` |
| `ProgramImplCore::finish` / terminal commit 迁移 / 统计 / 增长点 / decode 就绪 throw / capture 可行性与 skip | `src/targets/qwen3_6/impl/runtime/program_impl.h:9149-9219, 12158-12236（:12231）, 6723-6729, 11740/11889/12076, 11695, 7420-7660（:7539-7545, :7543-7550, :7605-7607）, 7804-7826, 7974, 8041, 9474-9575` |
| `StateImageRole` / `checkpoint_references` / `can_release_after_checkpoint_references` / `SequenceState::reserved_state` | `src/targets/qwen3_6/impl/runtime/state_image_store.h:69-74, 209-211, 686`；`src/targets/qwen3_6/impl/runtime/program.h:409` |
| `best_offered_reuse` / `plan()` / `PressureInputs` / `search_budget_ns` | `src/targets/qwen3_6/impl/runtime/materialization_planner.h:72-80, 85-94, 100-354（:187-188）` |

*报告生成：静态代码对照审查（prefix-v2 @ 4f0faf0c）；行号以当前分支为准。*
