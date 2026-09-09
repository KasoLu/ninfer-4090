# PREFIX-PLAN.md — max_tokens 动态 page 申请与 prefix 稳定必复用方案

> 目标分支：prefix-v2（HEAD `4f0faf0c`）。
> 参考材料：`KVCACHE.md`（本分支 KV/prefix/prefill 全景，984 行）；`PREFIX-ISSUS.md`（三问题归因，v2 250 行）；prefix-v1 分支（**仅作方向参考**，见 §0.2；完整 diff 与文档导出件在 `.tmp-prefix-plan/full.diff`（1241 行）、`.tmp-prefix-plan/prefill_v1.md`（271 行））。
> 本文是设计规划，不含代码改动。所有 file:line 以 prefix-v2 基线为准。
> **v2（可行性修订）**：已并入完整可行性审查（`PREFIX-PLAN-REVIEW.md`，181 行）——修正文中 7 处小误（§11.3）、落实 7 项 M 设计修正（§11.4）、新增审查记录章节 §11；文中代码引文均已逐行对照 prefix-v2 基线核验（实测行号）。

---

## 0. 设计原则

### 0.1 目标（用户原话）

> 1. 我期望 max_tokens 只是一个最大值的约束，不做强制性的 page 占用。整体采用动态申请的模式，如果申请失败，则表明上下文窗口已经到达极限，因此失败属于正常的逻辑情况。
> 2. 只要每次输入的 message 列表经过 chat-template 渲染成的结果与 prefix 是匹配的，那么无论什么情况，都不应该将 prefix 丢掉，也就是上述问题 2 的针对，除非是从一开始就匹配不上，才能触发全量的 prefill。
> 3. 总结一下就是：所有的 page 都按需分配；只要 prefix 是稳定的，就一定要复用，禁止在 prefix 匹配时，还触发全量 prefill。

### 0.2 prefix-v1 的地位（用户补充约束，原话）

> "prefix-v1 的实现我认为可能是很粗糙的。因为一开始我将 prefill 这块理解成了 bug，所以整体的改动是以修 bug 的方式进行……prefix-v1 只能作为参考，不能直接照搬……有十多次的提交，都是因为改出了 bug 或者是效果不佳。因此 prefix-v1 只能作为一个参考点，给你一个方向，但本次的方案分析和设计，还是要结合 KVCACHE.md，进行全面细致的规划，力求改动最小，且不要引入新的 bug。"

执行规则：

- **R-A 机制方向可参考，落地必须重推**：prefix-v1 的三个机制方向（动态 entitlement、root 预留捕获槽、轮转锚点图像回收、superseded 锚点释放）逐一基于 KVCACHE.md 全景在 prefix-v2 基线上重新推导落点（§2/§4），不复用其代码。
- **R-B prefix-v1 的 15 笔提交 = 缺陷史**：每类 bug（§4.3 表）在本方案中预先转化为防御不变量 + 回归测试，而不是"改了再测"。
- **R-C 失败语义重设计**：prefix-v1 池耗尽 → `runtime_error` → 引擎级 `fail_all`（全引擎停摆），与目标 1 直接冲突，**明确放弃**，改为请求级正常终结（§2.4/§2.5）。
- **R-D 改动最小**：不新增 EngineOptions、不新增协议字段（`ContextCapacity` 的协议映射已存在，§2.4）；改动集中在 §1.2 表列出的文件。

### 0.3 三条不可妥协的语义（目标的规范化表述）

- **S1 上限不预留**：准入时不预占 prompt+max_tokens 的全跨页；增长逐步申请（每步 1 token / 每 chunk），申请失败 = 上下文窗口到极限 = **该请求**以 `ContextCapacity` 正常终结，引擎继续服务。
- **S2 prefix 稳定必复用**：渲染后 prefix 与任一在册条目匹配 → 该条目及其 checkpoint **永不被驱逐/丢弃**（host 层开启时允许 D2H 迁移，迁移后仍可复用）→ 绝不触发全量 prefill；只有"一开始就匹配不上"（无匹配候选）才允许 root。
- **S3 按需分配**：所有 page 经既有单点 `materialize_sequence_kv` 惰性推进；预留的增长跨度在请求终结时归还页池。

---

## 1. 基线现状与改动落点

### 1.1 三个目标在 prefix-v2 上的现状归因（细节见 PREFIX-ISSUS.md 与 KVCACHE.md）

| 目标 | 现状机制（prefix-v2） | 违反点 |
|---|---|---|
| 1 | 计划期全跨预留：`request_plan_impl.h:246-282`（`reserved_context_tokens = prompt + (effective==0 ? 0 : effective-1)`，`entitlement = pages_for_tokens(reserved)`，`root_active.main_kv_pages = entitlement`）；`logical_kv_store.h:905-945` `prepare_activation` 一次性 `resize_reservation` 扣空闲页（`tail_reservation_` 挂地址空间 `:223-235`）；增长闸门 `logical_kv_store.h:1442`（`materialize_to_tokens` `:1417-1469` 内）：`target < page_count \|\| target > entitlement` → `throw std::invalid_argument("KV materialization exceeds active entitlement")` → `program_impl.h:8768-8806` decode rethrow → `engine_core.h:2185-2187` worker_loop catch → `fail_all_locked`（`:2073-2105` 语义：pending 全部 complete_error，引擎停摆） | max_tokens 直接转成物理页占用（用户问题 1）；唯一容量失败路径是引擎崩溃，而非"正常的逻辑情况" |
| 2 | 软保护：`demand_mask`（`materialization_planner.h:26`）→ `ContextPortfolioValue::fold`（`context_portfolio_value.h:55-101`，demand bit → baseline/target 公共价值，驱逐被 demand 的 checkpoint 只是 net_gain 变差，**非硬禁止**）；checkpoint policy 只用当前请求的 `provisional_demand`（`resource_manager.h:1811/1829/2017-2018/2075-2076`）；`demand_window_`（`kDemandWindowCapacity=32`，`resource_manager.h:1423`）仅在 adopt 时 `commit_demand` → **排队（pending、未 admit）请求零保护**；planner 中 root 与 reuse 候选同台竞争（`materialization_planner.h:81-340`：identity 评估 → Feasible → `logical_goal` → `identity_best`），reuse 不可行且压力预算（`search_budget_ns = min(5'000'000, incumbent.total_ns/20)`，bounded A*，`kTargetBudget=4096`）内修不好时 **root 胜出 → 全量 prefill**；准入重检是事件触发（`request_admission_check()` 仅于 `engine_core.h:245/875/884/890/982/1093/1230/1545/1687/1714/1732/1749/1756/1776/1793/1802`），decode 轮完成不触发 → TemporarilyBlocked 的 pending 最长等 `pending_timeout_ms`（serve 默认 **600000ms**，`src/serve/serve_options.h:33-34`；30000ms 是部署配置值） | 匹配 prefix 可被驱逐 → 下轮 root（用户问题 2）；"prefix 匹配还触发全量 prefill"正是 planner 选 root 的结果 |
| 3 | StateImage 槽冻结（实测：prefix-v1 PREFILL.md §6，25 轮 `--request-log`）：req1(root, 71199 tok) 发布私有锚点 @70878 占唯一 extra 槽（池 = 活跃 1 + extra 1）→ 后续轮自身新锚点 @~71.9k 无槽 → infeasible → 纯私有 capture 静默 skip（`program_impl.h:7823` `!pressure && !feasible`，该路径无压力分支，刻意设计）→ 每轮采纳 @70878 走 Retain（源保留 + 活跃边整 run 保护，`resource_manager.h:3036-3039`）→ 目录永不 churn → reuse 前沿冻结在 70878、每轮重 prefill 尾 28.6k tok | 新锚点发布依赖"新空槽"，而首锚点永不被驱逐 → 复用前沿永久停滞（用户问题 2 的另一半，与目标 2/3 交织） |

### 1.2 改动落点总表

| 文件 | 改动 | 支柱 |
|---|---|---|
| `src/targets/qwen3_6/impl/runtime/logical_kv_store.h` | `materialize_to_tokens` 增长闸门改 `resize_reservation` + 失败返回 `KvGrowth::PoolExhausted`（不抛）；`prepare_activation` 接受小 entitlement | 1 |
| `src/targets/qwen3_6/impl/runtime/request_plan_impl.h` | entitlement = pages(prompt)；MTP/DFlash 后端预留；prompt-fit guard 改请求级拒绝；root `state_slots=2`（机制 A） | 1/3 |
| `src/targets/qwen3_6/impl/runtime/program.h` | 新 API：`probe_decode_capacity`（基于 `can_resize_reservation` 无异常预检）/ `mark_capacity_stalled`（前置含 `!capture_pending`，镜像 `pending = {}`，M1）；`drop_superseded_anchor` 落点声明（机制 C） | 1/3 |
| `src/targets/qwen3_6/impl/runtime/program_impl.h` | 新 API 实现；`decode_raw` 行内增长预检（防御不变量）；capture 目的顺序 prepared→recycled→reserved→new（机制 B）；`release_superseded_source_anchor` | 1/3 |
| `src/runtime/engine/resource_manager.h` | 硬保护 owner 集（inspect → planner 输入）；`note_pending_demand`（排队需求登记，纯窗口插入，M3）+ `PrefixDemandRecord` 增 owner 字段；锚点释放钩子（机制 C 管理层校验） | 2/3 |
| `src/runtime/engine/materialization_planner.h` | 压力目标空间硬约束（禁 Evicted/丢 checkpoint 保护 owner）；root gating（§3.3，M2 细化谓词） | 2 |
| `src/runtime/engine/shared_capture_planner.h` | capture 场景 victim 选择排除保护 owner | 2 |
| `src/runtime/engine/engine_core.h` | 轮前容量探测 + stall 终结链（§2.5，探测/stall 早于 `build_round_membership`）；submit 时排队需求登记；decode 轮完成 `request_admission_check()`（§3.5） | 1/2 |
| `tests/test_resource_manager.cpp`（+ FakeProgram） | §4.3 八类不变量回归 + §3 新场景（含 M2/M5/M7 场景） | 2/3 |
| `include/ninfer/types.h` / `src/runtime/generation/generation_budget.h` | **预期不改**：`FinishReason::ContextCapacity`（`types.h:555`）与协议映射已齐备；`GenerationBudget(effective_tokens, limit_reason)`（`generation_budget.h:11-45`，limit_reason ∈ {OutputLimit, ContextCapacity}，构造/commit 越界 → `std::abort()`）已足够 | — |

---

## 2. 支柱一：动态 KV 页申请（目标 1）

### 2.1 现状机制链（KVCACHE.md §1/§9 对应）

1. **计划期**：`request_plan_impl.h:246-282`：`capacity_output = capacity - prompt_tokens + 1`；`effective_output_tokens = min(max_tokens, capacity_output)`；`effective_limit_reason = (requested <= capacity_output) ? OutputLimit : ContextCapacity`（基线已预计算）；`reserved_context_tokens = prompt_tokens + (effective==0 ? 0 : effective-1)`；`text_kv_page_entitlement = pages_for_tokens(reserved_context_tokens)`；MTP backend = `pages_for_tokens(min(capacity, reserved_context_tokens + draft_window - 1))`；DFlash backend = `pages_for_tokens(reserved_context_tokens)`（**实测基线**；旧版误引为 `pages_for_tokens(prompt_tokens)`，§11.3 #3）。`pages_for_tokens = 1 + (tokens-1)/64`（`request_plan_impl.h:44-46`；`kPagedKVPageSize=64`，`src/core/paged_kv_cache.h:17`）。
2. **激活期**：`logical_kv_store.h:905-945` `prepare_activation`：`growth = entitlement - required_pages`，`resize_reservation` 一次性扣走 growth 值空闲页（`tail_reservation_` 挂地址空间，`:223-235`）；`:860-880` `create_active` 在 `entitlement == 0 || > page_capacity_` 时返回 nullopt——注意该 nullopt 通道**仅覆盖静态（地址空间）容量**；**运行期**页池耗尽是 `std::bad_alloc` 异常（`paged_kv_cache.cpp:268`），可用 `can_resize_reservation`（`paged_kv_cache.h:226-228`，noexcept）无异常预检（§11.3 #4）。
3. **采纳期**：`program_impl.h:9818-9819/9866-9867/10692-10701` `resize_sequence_kv_entitlement`（reuse 候选采纳后按源重挂增长跨度）。
4. **增长期**：单点 `materialize_sequence_kv` 覆盖 prefill（text_context_impl.h 每 chunk）、decode（`program_impl.h:11740` 每步 1 页）、MTP（`:11889` `frontier + extent + 1`）、DFlash（`:12076`）、session 回放。闸门 `logical_kv_store.h:1442`（`materialize_to_tokens` `:1417-1469` 内）越界 throw。
5. **终结期**：`program_impl.h:9206` `release_sequence_growth_entitlement`（增长段只在终结归还）；`:6723-6729` 统计 `device_pages += entitlement - mapped`（**整段计入**——动态页下会系统性高估占用，须改为按 mapped 计，§2.5-E4）。

### 2.2 改动 1：计划期 entitlement = 仅 prompt

- `text_kv_page_entitlement = pages_for_tokens(prompt_tokens)`；MTP backend = `pages_for_tokens(min(capacity_tokens, prompt_tokens + draft_window - 1))`；DFlash backend = `pages_for_tokens(prompt_tokens)`（注意与基线不同：基线为 `pages_for_tokens(reserved_context_tokens)`，动态化后三者统一收敛到 prompt 维度）。
- `effective_output_tokens / effective_limit_reason` 字段语义保留（协议统计不变），但**不再进入预留量**；`limit_reason` **保持基线预计算语义**（M6：`requested <= capacity_output ? OutputLimit : ContextCapacity`，`request_plan_impl.h:246-282`），**不引入**与页池容量纠缠的新规则（旧版该规则已废弃，§11.3 #6）——池维度完全由 §2.5 stall 覆盖（E1 覆盖规则不变：池先枯竭 → stall reason = ContextCapacity）。
- **prompt-fit guard**：prompt 本身放不下（`prompt_tokens ≥ capacity` 或 prompt 页超出页池）从 throw 改为**请求级正常拒绝**（与前端既有 `> max_context` 的 `throw_context_length_exceeded`（frontend.cpp:1438-1439）同一语义层；池级经 `can_resize_reservation` 预检（`paged_kv_cache.h:226-228`）作请求级拒绝，**不抛异常**——`create_active` nullopt 通道仅覆盖静态容量，§11.3 #4）。
- `GenerationBudget` 构造不变。

### 2.3 改动 2：动态增长闸门（单点）

`logical_kv_store.h` `materialize_to_tokens`：

- 保留 `target < page_count` → throw（不变量 I1：物化单调，违反是 bug 不是容量问题）；
- `target > entitlement` 不再 throw，改为 `resize_reservation(reservation, target - page_count)`（原子批：全部页预留成功或一个不留）；
- `bad_alloc`（页池耗尽）**不再向上抛**，改为返回新结果枚举 `enum class KvGrowth { Ok, PoolExhausted, InvalidTarget }`；
- 该单点是全系统唯一的"池不足"来源（I3）：`:860-880` 的 `entitlement ≤ page_capacity_` 预检保留，仅作计划期静态快速拒绝（运行期池耗尽走 `KvGrowth::PoolExhausted` / stall 链，§2.4-§2.5）；运行期拒绝经 `can_resize_reservation` 预检，无异常上抛。

prefix-v1 在此单点的改法方向一致（commit `52a6b787`），但失败语义按 §2.4 重设计。

### 2.4 改动 3：失败语义 = 请求级正常终结（与 prefix-v1 的关键差异）

- **无需新增 FinishReason**：`FinishReason::ContextCapacity`（`include/ninfer/types.h:555`）协议映射已存在——OpenAI → `"length"`（`openai_chat_response.cpp:46-47`、`openai_responses_response.cpp:22`），Anthropic → `"model_context_window_exceeded"`（`anthropic_messages_response.cpp:64-65`）。
- 语义：页池（含其他会话的合法占用）无法容纳本 sequence 的下一步增长 → **该请求**携带已生成响应正常终结，lane 释放、页归还，引擎继续服务。这就是用户所说"失败属于正常的逻辑情况"。
- 区分：max_tokens 自然耗尽 → `OutputLimit`；池先枯竭 → `ContextCapacity`（E1 覆盖规则见 §2.5）。

### 2.5 改动 4：stall 终结链（本支柱核心）

**Program 侧新 API**（`program.h` / `program_impl.h`）：

1. `KvGrowth probe_decode_capacity(SequenceHandle sequence) const` —— 纯查询，无变更、无设备工作：本 sequence 能否再长一步。（a）text KV 池空闲 + 未预留页数 ≥ 所需增长页（普通 decode 1 页；MTP 含 draft-window 增长）；（b）backend（MTP/DFlash）池同判；（c）MTP 时 StateImage 槽可用。实现用 `can_resize_reservation`（noexcept）+ `available_pages()`，无异常路径。
2. `bool mark_capacity_stalled(SequenceHandle sequence)` —— 纯 host 状态迁移，镜像 terminal commit 迁移的两项（`program_impl.h:12231`）：`requests[lane].lifecycle = Finishable` + `request.pending = {}`；前置条件（M1，**最致命前置**）：无 pending batch、`!has_context_transaction()`、**`!capture_pending`**（settle 的 ownership 校验 `engine_core.h:1128-1133` 要求 `is_model_finished() && !capture_pending && lane_state==TerminalPending`，违反 → `throw std::logic_error("terminal-pending request has invalid ownership")` → worker catch → `fail_all` 引擎停摆）、lifecycle == Active；**无设备工作、不产 token**。sequence 的 ledger / `text_kv_valid` / `prefix_identity` 全部不变——响应 token 已在 ledger 中，`SessionEndpoint` 合法，可正常 catalog。

**Engine 侧流程**（`engine_core.h`；M1：轮内容量探测与 stall **必须早于该轮首次 `build_round_membership`（`:2152`）** 执行，stall 后的 lane 从该轮起排除出 membership）：

3. 逐 DecodeReady lane 探测；`probe == PoolExhausted` 时：
   - 预算清零（`budget->commit(remaining)`，`RoundBudget.generated_tokens_remaining = 0`，`contract/types.h:598-601`）→ 该 lane 自动排除出**后续所有** membership（零预算 lane 进 membership 会抛 `"ordinary batch row is not decode-ready"`，`program_impl.h:11695`，故排除是必须的，M1）；
   - `request->output.preview_terminal(FinishReason::ContextCapacity)`（镜像 cancel 路径 `engine_core.h:1178` 的 `preview_terminal(Cancelled)`）；
   - `program->mark_capacity_stalled(sequence)`；
   - 新窄钩子 `resources_.note_capacity_stall(...)`：lane Active → TerminalPending（复用既有 `mark_terminal_pending`，`resource_manager.h:928-932`，镜像 `apply_commit` 的 Finishable 迁移）；
   - `request->model_state = ModelFinished`；`request->terminal_reason = ContextCapacity`。
4. `settle_terminal_requests`（`engine_core.h:1109-1185`）下一轮自然交付：`resources_.finish`（`:1142`）→ `program.finish`（Finishable lifecycle 与 terminal commit 同等处理，应返回 Consumed）→ disposition Catalogued 时记录 `retained_slot`（`:1145-1153`）→ `complete_success(request, ContextCapacity)`（`:1158`，`:1021` 对任意 reason 通用）→ `remove_completed_slot`（`:1091-1094`，内部 `:1093` 触发 `request_admission_check`，排队者立即补位）。**M5 stall 专用 settle 断言**：stall 分支校验 `program.finish` 必须返回 Consumed，否则 `throw std::logic_error`（暴露缺陷），**不走自动 `program.abort` 回退**——abort 回退的 `clear_catalog_entry` 会丢失该请求自己的 endpoint 目录条目 → 下一轮 prefix 匹配断裂，直接违反目标 2（stalled 请求带 fork_pending 且 `abort_fork` 失败时触发）；回归单测必须覆盖 fork_pending 场景。

**为什么必须走 finish 而非 cancel**（`cancel_active_requests`，`:1167-1191`）：`resources_.abort` 丢弃 continuation（不 catalog）→ 本轮响应 token 不进 prefix 目录 → 下一轮 prefix 匹配断裂 → 直接违反目标 2。stall 必须 Catalogued。

**边缘情况**：

- **E1 reason 覆盖**：预计算 reason = OutputLimit 但池先枯竭 → stall 路径的 `terminal_reason` 恒为 ContextCapacity（`preview_terminal` 使用 stall 自身 reason，不读 `budget->limit_reason()`）。Anthropic 协议要求区分 max_tokens 与上下文窗口，此覆盖是语义正确性的需要。
- **E2 轮内防御**：轮前探测是正常路径；行内（`decode_raw` 行循环前）再做一次确定性预检作为**防御不变量**——同轮内页池不被其他执行单元消费，若预检失败即真实 bug，`throw std::logic_error` 走 fail_all 暴露，**不静默降级**（刻意设计：暴露 bug 不掩盖，已文档化）。
- **E3 prefill 期枯竭（M4：结构性自消）**：新 entitlement = pages(prompt) 后，prefill 只从激活期已预留页（`tail_reservation_` `logical_kv_store.h:223-235` 持有、他者不可取）物化 → **prefill 点池枯竭结构性不可能**；prefill 路径增长失败保留为防御 throw（bug 语义走 fail_all，与 E2 一致）；**P0 不为 prefill lane 增加 stall 信号通道**（`advance_prefill` 仅有 PrefillProgress/异常通道，`engine_core.h:1554-1601` 异常 → worker catch → fail_all；`catch(...)` 无法区分容量与 bug）。
- **E4 统计修正**：`program_impl.h:6723-6729` 的 `device_pages` 统计从 `entitlement - mapped` 改为按 `mapped` 计（动态页下原口径系统性高估；另可新增 growth_span 观测字段，可选）。

### 2.6 支柱一验证点

- 单测（logical_kv_store）：`resize_reservation` 耗尽 → `KvGrowth::PoolExhausted`、异常不外泄；单调性 I1；原子性 I2（失败后 `page_count` / reservation 不变）。
- 集成：A 占大池（大 max_tokens）+ B decode 增长失败 → B 以 ContextCapacity 正常终结，A 不受影响，引擎持续服务；finish_reason 协议字段（OpenAI/Anthropic 双通道）。

---

## 3. 支柱二：prefix 硬保护与 root gating（目标 2）

### 3.1 现状缺口（四连）

1. **软而非硬**：`demand_mask`（`materialization_planner.h:26`）经 `ContextPortfolioValue::fold`（`context_portfolio_value.h:55-101`）只使净收益变差；压力足够大时规划器仍会驱逐被 demand 的 checkpoint。
2. **覆盖仅当前请求**：materialization checkpoint policy 只用 `provisional_demand`（`resource_manager.h:1811/1829/2017-2018/2075-2076`）→ 只有当前请求自身候选 key 受（软）保护。
3. **排队零保护**：`demand_window_`（32 槽滑动窗，`commit_demand` 仅于 adopt 时入队）→ pending 未 admit 请求的匹配 prefix 在排队期间可被驱逐 → 准入即 root。
4. **root 同台胜出**：`materialization_planner.h:81-340` 中 root 恒 Feasible，reuse 不可行且预算内修不好时 root 胜出 → "prefix 匹配仍全量 prefill"的直接成因。

### 3.2 改动 1：硬保护 owner 集（规划器硬约束）

- **定义**（inspect 时，`resource_manager.h`）：由当前请求全部候选（含 root）推导匹配 key 集 `K = candidate_keys ∪ exact_resident_keys`；保护集 `P = {private 目录条目：其 endpoint / rewrite / long_anchors 任一 key ∈ K} ∪ {shared 目录条目：shortlist_key ∈ K}`。
- **硬约束**：压力目标空间（`materialization_planner.h` bounded 搜索 + `shared_capture_planner.h` 场景压力）中，任何 Evicted 或丢弃 P 中 owner checkpoint 的目标**不可行**（不是"更差"，是剪枝）。host 层开启（`host_kv_capacity_bytes > 0`）时 P 中 owner 允许 DeviceToHost 降级——迁移后复用仍成立（Retain + H2D transfer，KVCACHE.md §6/§7 机制不动）。
- **落点**：`PressureInputs` 新增 `protected_owners`（slot/owner 集合，`materialization_planner.h:85-94` 输入结构）；planner 目标扩展时跳过含保护 owner 的目标；capture planner 的 `direct_shared_victim` 选择排除 P。
- **保护范围**：owner 仅在其 key 被 **active（已 admit 未终结）或 pending（排队）** 请求匹配时受保护；请求终结/取消/超时后匹配关系消失 → owner 回到常规 retention 驱逐池（权重 Disposable=1 / RecentPrivate=4 / LiveSession=16 / SharedStable=0，`private_retention_weight`）→ 不会把页泄漏给无人引用的 prefix。

### 3.3 改动 2：root gating（选择硬规则）

- **M2 root gating 谓词（细化）**：仅当 `best_offered_reuse > 0`（`materialization_planner.h:72-80`，各候选 `summary().reusable_prompt_tokens` 的 max，**与物化可行性无关**）**且存在 feasible 或可经进展变为 feasible 的匹配候选**（等页释放 / 等锚点槽 / D2H 槽位——planner 诊断 `stop_reason` / `budget_exhausted` 可辨）时 → **root 候选禁止入选**；匹配候选全部不可行且压力预算内修不好 → plan 返回 nullopt → 引擎保持 TemporarilyBlocked 排队等待。
- **结构性死亡放行 root（M2 核心）**：匹配候选结构性死亡（prefix 物理副本已合法丢失：无设备页、host 层关闭、无可驱逐 owner，预算内无法恢复）→ **允许 root 全量 prefill**——此时 root 不会多丢任何东西（prefix 已不可恢复），基线行为同样是 root prefill（慢但成功）；旧版"永不选 root"过强，会死等至 `pending_timeout`（serve 默认 600s）→ `QueueTimeout` 失败，属对基线的破坏性行为变更（F-2，§11.4）。
- 无压力快速路径（无 expandable root → `seal_identity(FinalScheduleIntent)`）**同样加 M2 守卫**（root 可在该快速路径胜出）：存在 feasible/可进展的匹配候选时禁止快速路径（必须走完整压力搜索）；仅当无匹配候选（`best_offered_reuse == 0`）或匹配候选结构性死亡时，快速路径才允许 root 胜出。
- 该规则是目标 2 的直接落点："只要每次输入的 message 列表经过 chat-template 渲染成的结果与 prefix 是匹配的，那么无论什么情况，都不应该将 prefix 丢掉"（M2 细化后：匹配但物理已丢失 = "匹配"已失效，root 是基线行为，不算丢 prefix）。

### 3.4 改动 3：排队需求保护

- 请求 submit 时（engine submit 路径，`:982` 既有 `request_admission_check()` 点旁）计算其 prefix 需求 key（`program.plan_request` 是纯 host 侧准备、无事务；或新增更轻的 `extract_prefix_demand(prompt)` 只返回 shortlist keys），经新 API `resources_.note_pending_demand(request_id, shortlist_keys)` 登记进 demand window。
- 生命周期：排队 → admit（既有 `commit_demand` 接管）/ 取消 / 超时 / 终结（出窗，M3 增 owner 字段后支持定点出窗）。**M3：`note_pending_demand` 必须是纯窗口插入**，不得触发 `commit_demand` 的 explicit_credit 清除侧效应（shared Catalogued 且 explicit_credit 的条目，若其 shortlist_key 不在最新窗口记录 `exact_resident_keys` 且 `demand_epoch_ ≥ credit_expiry_epoch` → 清 credit，`resource_manager.h:2457-2480`；纯插入跳过该逻辑，避免误清他请求的匹配 credit）。pending 请求的匹配 owner 同样进硬保护集 `P`（§3.2）。
- 容量核算：active ≤ 8（`kMaximumConcurrency`，`types.h:20`）+ pending ≤ 16（serve 默认 `max_pending_requests`，`serve_options.h:33`）= 24 ≤ 32（`kDemandWindowCapacity`，`resource_manager.h:1423`）。**M3 运维约束：`max_pending_requests + active` 不得超过 32**（超出时最旧 pending 需求被窗口静默丢弃 → 保护降级；须文档化）。**M3 结构备注**：`PrefixDemandRecord`（`resource_manager.h:122-126`：candidate_keys / exact_resident_keys / selected_source_key）**无 owner 字段** → 取消/超时/终结时定点出窗不可能；**首选：增 owner（request id）字段（最小改动）**；若不加，则采用 lazy expiry（保护随窗口滑动自然过期，放宽 I5——须文档化，§11.3 #7）。

### 3.5 改动 4：decode 轮完成后重准入

- `worker_loop`（`engine_core.h:2104-2228`）每轮本就重读 `admission_check_pending_` 并评估 `should_attempt_admission(have_pending, ...)`（`:2133-2139`，gate 含 `previous_unit_was_decode`），但触发点不含 decode 轮完成（`run_decode_round` 尾部 `:2050-2066` = budget commit + append_output + `model_state=DecodeReady` + 统计发布，无 setter）。在 `run_decode_round` 之后补一次 `request_admission_check()`（一行；既有 gate 已能放行）：TemporarilyBlocked 请求在每次 decode 轮完成后立即重准入，消除最长 `pending_timeout_ms`（serve 默认 **600000ms**）的盲区（decode 步完成可能归还增长段/锚点槽，正是阻塞解除的时机）。

### 3.6 与支柱一的交互

硬保护只"禁止驱逐匹配 prefix"，不自身新增资源需求 → 不与池枯竭死锁（前向进展完整论证见 §5）。

---

## 4. 支柱三：StateImage 槽解冻（目标 3 / 问题 2 根因）

### 4.1 根因与已排除杠杆（不重提）

- 根因链见 §1.1 行 3：新锚点发布依赖新空槽，而首锚点被价值保护永不驱逐 → 纯私有 capture 永久 infeasible → 静默 skip（`program_impl.h:7823`）→ 复用前沿冻结。
- 已排除：增大 `device_state_slots`（任何有限池都被首锚点保护 + 无驱逐触发解冻）；改锚点数（同上）；共享路径（`openai_common.cpp:163-176` `allow_engine_automatic_shared_prefixes=false`；`matching_reuse_domains < 2` 不触发共享压力替换，`:585-590`）。
- 结论：唯一解是**新锚点发布不依赖新空槽**——槽池随锚点数增长，而非随代数和增长。

### 4.2 三机制（prefix-v1 方向，prefix-v2 重落地）

**机制 A — root run 预留 1 个 capture 目的槽**

- `request_plan_impl.h`：`root_active.state_slots = 2`（活跃 1 + capture 目的 1；基线 `:420` 为 `1U`）。理由：冷会话首轮是 root（无匹配候选，root gating 不拦），若 root run 发不出锚点（无目的槽），该会话永不进入复用轨道 → 每轮 root 全量。prefix-v1 `c0a3990e` 验证方向。激活期 `reserved_states[1] → sequence.reserved_state` 绑定与 `reserve_state_entitlement`（`program_impl.h:9474-9575` 激活管道、`:9823/9871`）在基线已存在，本机制仅把 root 的 `state_slots` 从 1 提到 2，槽在激活期即计入 active entitlement，无新增 peak——capture feasibility 公式只需 OR 一个目的分支。
- 预算纪律（针对 prefix-v1 缺陷类 1）：预留槽在计划期计入 `optional_resources` **恰好一次**；adopt 时标记实际占用、abort 时释放，两条路径都有单测（§4.3 #1）。
- **M7 状态槽冲突**：状态槽池紧张时（如池 = 2、两个并发 root run），第二个 root run 激活期 `reserve_state_entitlement`（`program_impl.h:9823/9871`）失败 → 定义为**请求级拒绝**（与目标 1 语义一致：申请失败 = 正常逻辑情况），**不得阻塞引擎 / 不得 fail_all**；单测（双 root run 并发、池 = 2，第二个被拒绝且第一个不受影响）。

**机制 B — 轮转锚点图像回收为 fork 目的**

- run 采纳 source@reuse_base 并需 fork 新状态图像（ConsumeToActive / 新锚点）时，优先复用**被本轮轮转掉的锚点**的 StateImage 槽（轮转锚点图像内容已不被任何候选需要）→ 不耗新槽。
- 落点：`program_impl.h` `prepare_active_capture`（`:8041`）目的选择顺序 = prepared → **recycled（轮转锚点图像）** → reserved_state（机制 A 槽）→ 新槽；`install_private_capture(..., recycles_replacement_image)`（`:7974`）回收时**不释放** victim 的 checkpoint 引用（引用迁移到新图像，引用数恒 ≥ 1，`checkpoint_references` 恒 ≥ 1 不变量）；abort 路径归还预留槽并恢复被回收图像的引用。
- prefix-v1 `651f2310` / `143f46b2` 验证方向。

**机制 C — 释放被取代的源锚点（窄 API）**

- 触发：run 采纳 source@reuse_base（Retain 或 ConsumeToActive）且发布了自己更深的锚点（frontier 更大的 LongAnchor）→ source 锚点@reuse_base 被严格支配（凡匹配旧锚点者必匹配新锚点）→ 释放它归还槽位，匹配前沿前进。
- 窄 API（resource manager 侧）：`std::optional<ContinuationSummary> drop_superseded_anchor(uint32_t continuation_index, uint32_t generation, CheckpointRef anchor) noexcept`。校验（全部通过才变更状态）：`!has_context_transaction()` ∧ slot < capacity ∧ 条目 `CatalogState::Catalogued` ∧ generation 匹配 ∧ `anchor.kind == LongAnchor` ∧ 锚点在该条目在册 ∧ `state_store->checkpoint_references(anchor.state) == 1`（无他引，`:209-211` 既有原语）。成功 → generation 自增（0 跳过）+ `advance_resource_revision` + `advance_revision` + `rebuild_prefix_index` + 返回刷新 summary；任一失败 → 返回 nullopt、**绝不抛、状态零变更**。
- 运行期包装 `release_superseded_source_anchor(program, lane)`：释放 lane retained source 的**最深** LongAnchor + `assign_continuation_summary` + `migrate_observations` + `advance_revision` + `rebuild_prefix_index`。
- skip 分支（capture 不可行，`program_impl.h:7822-7826` 既有路径 `!pressure && !physically_feasible`）：**先执行机制 C 释放，再重评估一次**；仍不可行照旧 skip → 把"永久 infeasible"转为"释放后可行"（prefix-v1 `143f46b2` 核心，本方案重实现）。
- 崩溃恢复降级：被释放锚点若是崩溃恢复可用 checkpoint，恢复降级 root——可接受（prefix-v1 PREFILL.md §6.1 结论，本方案沿用；只动 LongAnchor，不碰 endpoint）。

### 4.3 prefix-v1 缺陷史 → 防御不变量与测试

prefix-v1 15 笔提交（tip→base：`b806b2f6` → `143f46b2`，12 文件 +799/-68），其中 10+ 笔为 fix/diag。每类缺陷对应本方案不变量 + 回归测试（`tests/test_resource_manager.cpp` 的 FakeProgram 扩展）：

| # | prefix-v1 缺陷类（commit） | 本方案不变量 | 测试 |
|---|---|---|---|
| 1 | optional-resource 预算 overflow（`b806b2f6`/`1643cce4`/`74c53a81`/`5645ec6c`） | 每个状态图像/槽在 plan/adopt/commit/abort 四阶段恰好计一次；adopt 时实际 optional_resources == 计划预算 ± 回收释放量 | 串行 capture × N 轮，预算恒不溢/不欠 |
| 2 | underflow 诊断（`1e5b6c7c`/`8fce6fdc`/`96f3ca86`） | 诊断字段声明与调用点签名一致；调用点输出维度信息 | 编译期签名检查 + 诊断输出断言 |
| 3 | retain-fork 选择（`8a2d4eb9`） | 未绑定 long-anchor adopt 先试 Retain fork，不可行回退 ConsumeToActive（conservative_anchor_first）；选择唯一且确定 | 未绑定锚点两态，断言选择 |
| 4 | same-session 源消费（`ff4473a4`） | 同会话 long-anchor 物化允许 ConsumeToActive（源被新 run 消费，不再永久占用） | 同会话两轮复用，源条目状态迁移正确 |
| 5 | 轮转图像回收（`651f2310`） | 回收后 victim checkpoint 引用迁到新图像（引用数恒 ≥1）；abort 路径恢复引用 | 回收 + abort 混合场景 |
| 6 | superseded 锚点释放（`143f46b2`） | 释放校验失败（generation 不匹配 / 引用 >1 / 事务中）→ nullopt、零状态变更；skip 分支释放后最多重评估一次 | 各校验分支逐一命中 |
| 7 | root capture 槽（`c0a3990e`） | root run 计划恒含 1 个 capture 目的槽（state_slots=2）；abort 归还；M7 槽冲突 = 请求级拒绝 | 冷会话首轮成功发布锚点；双 root run 并发池=2 |
| 8 | skipped capture 计数（`6b9a535e`） | Skipped 结果计数可经请求诊断观测 | capture 跳过场景统计断言 |

---

## 5. 交互与前向进展（无死锁）论证

设：页池容量固定；active 请求 A（匹配，其 owner O ∈ P 受硬保护）；pending 请求 B（匹配 O，需要页）。

1. O 占用的页不可驱逐（支柱二）→ B 的物化可能不可行 → B TemporarilyBlocked，**不转 root**（root gating，M2 谓词）。
2. 每个 active 请求必然终结：max_tokens 完成（OutputLimit）或池枯竭 stall（ContextCapacity，支柱一）。max_tokens 是协议层上限（OpenAI/Anthropic 均有默认值兜底），不存在"无限生成"的请求；用户显式传超大 max_tokens 时，请求最终 stall 终结，同样收敛。
3. 终结时：lane 释放 + 增长段/锚点槽归还页池（`release_sequence_growth_entitlement` / `release_sequence_kv`；机制 B/C 进一步归还锚点槽）。
4. decode 轮完成触发 `request_admission_check()`（§3.5）→ B 重准入 → 第 3 步归还的页使 B 物化可行 → B 以 Retain 准入，prefix 不丢失。
5. 唯一"不进展"候选态：所有 active 请求都是大 max_tokens 匹配请求且池未满 → 池满的瞬间 stall 即触发终结 → 仍收敛。∎

结论：**硬保护 × 动态枯竭 × 轮级重准入**闭环，无需新增 EngineOptions、无需用户干预；用户视角的"上下文窗口到极限"就是 §2.4 的 ContextCapacity 正常终结。

---

## 6. 分阶段实施与回归验证

| 阶段 | 内容 | 验证 |
|---|---|---|
| **P0**（支柱一，独立可先行） | `logical_kv_store.h`（闸门 + `KvGrowth`）、`request_plan_impl.h`（entitlement=pages(prompt)）、`program.h/program_impl.h`（probe/stall，M1 前置与 M5 断言）、`engine_core.h`（stall 链 + decode 轮重准入，§3.5 一并做） | Docker `ninfer-local-build:latest` 全量编译 + ctest（仓库挂 `/src`、`build/` 复用、Ninja——AGENTS.md 约定，本机无 GPU）；GPU（4090 容器 `ninfer-4090-kaso-dev`，bind `/ninfer-4090-kaso`，端口 1234；**跑前须先请用户关闭常驻推理容器**）：① serve + `max_tokens=100000`（超池）→ 请求以 `finish_reason="length"` / `"model_context_window_exceeded"` 正常终结、引擎不停摆；② A 占大池 → B stall 终结 → 引擎持续；③ OpenAI/Anthropic 双协议 finish_reason 一致性；④ stall 专用 settle 断言（M5，含 fork_pending） |
| **P1**（支柱二） | `resource_manager.h`（硬保护集 + `note_pending_demand` + owner 字段，M3）、`materialization_planner.h` / `shared_capture_planner.h`（硬约束 + root gating，M2 谓词）、`engine_core.h`（submit 登记） | ctest：`test_resource_manager.cpp` 新增（匹配 prefix 压力下不可驱逐 / root gating 含 M2 结构性死亡放行 / pending 保护 / 终结后保护释放）；GPU：PREFIX-ISSUS §3.5 排查决策表场景——有匹配（feasible/可进展）时 `prefix_reuse_path` 恒非 Root，`best_reuse_prompt_tokens` 单调不降 |
| **P2**（支柱三） | `program_impl.h`（回收/释放/目的顺序）、`request_plan_impl.h`（root `state_slots=2`，M7）、`resource_manager.h`（`drop_superseded_anchor`） | ctest：§4.3 八类不变量表逐项；GPU：25 轮匿名场景——reuse 前沿单调前进（不再冻结），单轮重 prefill 长度 → 0，`active_captures_skipped ≈ 0`，崩溃恢复场景（机制 C 降级 root 可接受） |

---

## 7. 风险与不变量清单

- **I1 单调性**：KV 物化 target 单调不减（`target < page_count` throw 保留——违反即真实 bug）。
- **I2 原子性**：增长预留原子批（全成或全不成），失败无半态。
- **I3 唯一闸门**："池不足"在系统内仅 `materialize_to_tokens` 一个来源；其他路径的池耗尽是 bug（E2：防御预检失败走 fail_all 暴露，不静默）。
- **I4 目录一致性**：stall 终结必走 finish（Catalogued），不得走 abort（否则目标 2 断裂，M5 断言）；用户取消语义不变（仍 abort）。
- **I5 硬保护范围**：owner 仅在匹配 active/pending 请求时受保护；终结即释放，无页泄漏（M3：lazy expiry 模式下，保护随 32 槽窗口滑动自然过期，`resource_manager.h:2457-2480`）。
- **I6 选择确定性**：root gating（M2 谓词）/ 保护目标剪枝 / fork 与目的槽选择规则全部确定（无随机、无时序依赖），ctest 可复现。
- **I7 诊断完备**：stall / skip / 保护阻塞三类新路径均有 RuntimeStats 计数（复用既有分组 + 新增 capacity_stall 计数）+ request_log 输出（prefix-v1"诊断签名"教训，§4.3 #1/#2）。
- **I8 协议一致**：ContextCapacity → OpenAI `"length"` / Anthropic `"model_context_window_exceeded"` 映射不变，无新协议字段。

风险表：

| 风险 | 缓解 |
|---|---|
| R1 探测与轮内检查时序竞态 | E2：轮内预检为防御不变量，违反即 fail_all 暴露（刻意设计：暴露 bug 不静默，已文档化） |
| R2 硬保护导致长期等待 | §5 前向进展论证；M2：结构性死亡放行 root，避免"匹配候选已死但 root 永禁"；兜底为既有 `QueueTimeout`（`pending_timeout_ms`，serve 默认 600000ms） |
| R3 机制 C 释放削弱崩溃恢复 | 只释放 LongAnchor（不碰 endpoint/rewrite）；降级 root 可接受（§4.2） |
| R4 root `state_slots=2` 挤压他请求状态槽 | 池总容量不变；root run 是短暂态（首个 chunk 后即 settle capture 决策，槽即释放或转为锚点）；M7：槽冲突 = 请求级拒绝，不致引擎失败 |
| R5（M1）带 capture_pending 的 lane 被标 stall → settle ownership 校验 throw `"terminal-pending request has invalid ownership"` → fail_all | `mark_capacity_stalled` 前置增 `!capture_pending`；探测/stall 早于该轮首次 `build_round_membership`（`engine_core.h:2152`） |
| R6（M5）stall 分支 `program.finish` 非 Consumed → 自动 abort 回退 `clear_catalog_entry` 丢失该请求 endpoint 目录条目 → 下轮 prefix 匹配断裂 | stall 专用 settle 断言：非 Consumed → throw `std::logic_error` 暴露，不 fallback abort；单测覆盖 fork_pending 场景 |
| R7（M2）root gating 过强：匹配候选为幽灵匹配（key 匹配但物理副本已合法丢失）→ root 永禁 → `QueueTimeout`（serve 默认 600s）替代基线可成功的 root prefill | 谓词细化：仅在匹配候选 feasible 或可进展时禁 root；结构性死亡放行 root |
| R8（M3）`max_pending_requests + active > 32` → 最旧 pending 需求被静默丢弃（保护降级）；或 `note_pending_demand` 误清他请求 explicit_credit | 纯窗口插入（跳过 credit 侧效应）+ `PrefixDemandRecord` 增 owner 字段定点出窗 + 运维约束文档化（≤32） |

---

## 8. 参数与默认值变更

- **不新增 EngineOptions、不新增协议字段**。
- `KvCapacityPolicy`（explicit/automatic）语义不变：决定**页池大小**，不再决定每请求预留量；动态页下每请求计划预留恒为 `pages(prompt)`。默认 `explicit_tokens=2048` / `kDefaultKvCapacityHeadroomBytes=1GiB` 不变。
- `ContextCacheOptions`：host 两层（`host_state_slots` / `host_kv_capacity_bytes`）保持 opt-in（默认 0）——支柱二硬保护在 host 关时自动退化为"排队等待"，host 开时升级为"D2H 迁移 + 复用"，**无需改参数**。
- `device_state_slots` 默认值不变；机制 A 是计划期预算语义变化，不是池容量变化。
- **运维约束（M3）**：`max_pending_requests + active` ≤ 32（`kDemandWindowCapacity`，`resource_manager.h:1423`）；serve 默认 16 + 8 = 24 满足。

---

## 9. prefix-v1 对照表（采纳 / 修正 / 放弃）

| prefix-v1 项（commit） | 本方案处理 | 理由 |
|---|---|---|
| `52a6b787` 动态 entitlement（计划期 = prompt） | **采纳，重设计** | 方向正确；失败语义由 fail_all 改为请求级 ContextCapacity 正常终结（目标 1 明言"失败属于正常的逻辑情况"） |
| `c0a3990e` root 预留 capture 槽 | **采纳** | 冷会话进入复用轨道的唯一解；预算纪律强化（§4.3 #1）；M7 槽冲突 = 请求级拒绝 |
| `651f2310` 轮转锚点图像回收 | **采纳，重实现** | 池随锚点数而非代数和增长；补引用迁移不变量（#5） |
| `143f46b2` superseded 锚点释放 | **采纳，收窄为 noexcept 七项校验 API** | 校验集比 prefix-v1 完整（generation / checkpoint_references / 事务），绝不抛（#6） |
| `ff4473a4` 同会话 ConsumeToActive | **采纳** | 解决源永久占用；确定选择规则（#4） |
| `8a2d4eb9` conservative Retain-fork | **采纳，并入选择规则** | 确定唯一选择（#3） |
| `6b9a535e` Skipped 计数 | **采纳** | 可观测性前提（I7） |
| 池耗尽 `runtime_error` → 引擎 fail_all | **放弃** | 与目标 1 直接冲突；改请求级正常终结（§2.4） |
| "边改边修"的 diff 式实现（12 文件 +799/-68，10+ 笔 fix/diag） | **放弃（方法论）** | 每个机制按 KVCACHE.md 全景 + prefix-v2 基线重新推导落点（§1.2 逐点）；缺陷史预转为不变量 + 测试（§4.3），先测后改 |
| root 无条件 gating（匹配即禁 root，含幽灵匹配） | **修正（M2）** | 前缀物理已丢时匹配已失效，root 不多丢任何东西；无条件禁 root 会 `QueueTimeout`（serve 默认 600s）替代基线可成功的 root prefill |

---

## 10. 证据索引

| 内容 | 位置 |
|---|---|
| 全跨预留 / 增长闸门 / 统计 / 归还 | `request_plan_impl.h:44-46,246-282`；`logical_kv_store.h:223-235,860-880,905-945,1414-1425（resize_entitlement）,1417-1469,1442（闸门）`；`program_impl.h:6723-6729,9206,9818-9819,9866-9867,10692-10701` |
| decode/MTP/DFlash 增长点与行协议 | `program_impl.h:8768-8806,11664,11695,11740,11859-11860,11889,11927,12076,12231`；`program.h:588-600,1030,1162`；错误串 `"ordinary batch row is not decode-ready"`（`:11695`）、`"KV materialization exceeds active entitlement"`（`logical_kv_store.h:1442`）、`"pending row has an invalid licensed extent"` |
| 预算 / 终结原因 / 协议映射 / 服务默认值 | `generation_budget.h:11-45`；`include/ninfer/types.h:555`（FinishReason）、`:20`（kMaximumConcurrency=8）；`contract/types.h:598-601`（RoundBudget）；`openai_chat_response.cpp:46-47`；`openai_responses_response.cpp:22`；`anthropic_messages_response.cpp:64-65`；`src/serve/serve_options.h:33-34`（max_pending_requests=16 / pending_timeout_ms=600000） |
| 引擎终结 / stall / 准入重检链 | `engine_core.h:1109-1185`（settle_terminal_requests，finish 调用 :1142，ownership 校验 :1128-1133 = is_model_finished && !capture_pending && TerminalPending）、`:1167-1191`（cancel，preview_terminal :1178 / abort :1179）、`:2073-2105`（fail_all_locked，catch :2185 → :2187，注释 :268 permanent）、`:2104-2228`（worker_loop，admission 评估 :2133-2139，build_round_membership :2152，run_decode_round 尾部 :2050-2066）；`request_admission_check()` 触发点 :245/875/884/890/982/1093/1230/1545/1687/1714/1732/1749/1756/1776/1793/1802 |
| 软保护现状 / demand window / planner 结构 | `materialization_planner.h:26,72-80（best_offered_reuse）,85-94（PressureInputs）,100-354（plan/快速路径/bounded A*；kTargetBudget=4096；search_budget_ns=min(5'000'000, incumbent.total/20)）`；`context_portfolio_value.h:55-101`；`resource_manager.h:122-126（PrefixDemandRecord）,1423（kDemandWindowCapacity=32）,2457-2480（commit_demand + credit 侧效应）,1811,1829,1967-2100（build_pressure_inputs）,2017-2018,2075-2076` |
| StateImage 冻结根因与已排除杠杆 | `program_impl.h:7822-7826`（`!pressure && !feasible` 静默 skip）、`:7420-7660`（capture 可行性 / HostSnapshot :7543-7550）、`:7974`（install_private_capture）、`:8041`（prepare_active_capture）、`:9474-9575`（state_slots=2 激活管道）、`:9823/9871`（reserve_state_entitlement）；`resource_manager.h:3036-3039`（活跃边整 run 保护）；`state_image_store.h:69-74（StateImageRole）,209-211（checkpoint_references）`；`program.h:409`（reserved_state）；`openai_common.cpp:163-176,585-590`；实测数据：prefix-v1 PREFILL.md §6（`.tmp-prefix-plan/prefill_v1.md`，reuse=70878 冻结、尾 28.6k 重 prefill、71199 tok 场景） |
| 页池原语 | `paged_kv_cache.h:17（kPagedKVPageSize=64）,226-228（can_resize_reservation noexcept）`；`paged_kv_cache.cpp:268`（resize_reservation 失败 throw `std::bad_alloc`） |
| prefix-v1 完整 diff / 设计文档 / 提交史 | `.tmp-prefix-plan/full.diff`（1241 行，12 文件 +799/-68）；`.tmp-prefix-plan/prefill_v1.md`（271 行）；15 笔提交 tip `b806b2f6` → base `4f0faf0c`（= prefix-v2 HEAD，merge-base 相同） |
| 三问题归因（issue 1/2/3） | `PREFIX-ISSUS.md` §1/§2/§3（v2，250 行） |
| 可行性审查与修正记录（v2 并入） | `PREFIX-PLAN-REVIEW.md`（34352B / 181 行，独立报告）；本文 §11（并入版） |
| 分支全景（KV 物理 / 算子 / 规划器 / 调度 / prefill） | `KVCACHE.md` §0-§13（984 行） |

---

## 11. 可行性审查与修正记录（v2 并入）

> 来源：`PREFIX-PLAN-REVIEW.md`（181 行，独立审查报告）。方法：对方案逐条断言对照 prefix-v2 基线代码逐行核验（本章行号均为实测），并核对线程模型（所有引擎入口均取 `execution_mutex_`，`worker_loop` 跨整个 try 持锁 → 页池/槽池计数器单线程访问，`probe_decode_capacity` 无竞态）与协议映射。

### 11.1 审查结论

**条件通过**：三支柱落点全部真实存在——增长原语 `resize_entitlement`（`logical_kv_store.h:1414-1425`）、stall 状态迁移 `mark_terminal_pending`（`resource_manager.h:928-932`）、锚点校验原语 `checkpoint_references`（`state_image_store.h:209-211`）均为既有管道；新 API 面小、协议零新增（ContextCapacity 映射齐备）。完成 7 项 M 修正（均已在本版落实，§11.4）后，可满足三目标，不引入引擎级破坏性变更或新失效模式；唯一"意外"破坏性行为变更（F-2：root gating 过强 → `QueueTimeout` 替代基线可成功的 root prefill）由 M2 消除。

### 11.2 三支柱断言核验总览

| 支柱 | 关键断言 | 代码证据 | 结论 |
|---|---|---|---|
| 1 | 增长闸门 / 增长原语 | `logical_kv_store.h:1442` 闸门（`target > entitlement`）；`:1414-1425` `resize_entitlement` 已存在（`resize_reservation` 封装） | ✅ |
| 1 | 池耗尽 = bad_alloc 异常、可无异常预检 | `paged_kv_cache.h:226-228` `can_resize_reservation` noexcept；`paged_kv_cache.cpp:268` `resize_reservation` 失败 throw `std::bad_alloc`；`create_active` nullopt 仅静态容量 | ✅（旧文小误 #4） |
| 1 | 计划期基线与 limit_reason | `request_plan_impl.h:246-282`：`effective_limit_reason = requested <= capacity_output ? OutputLimit : ContextCapacity`（基线已预计算） | ✅（小误 #3/#6） |
| 1 | 协议映射 | `types.h:555`；`openai_chat_response.cpp:46` / `openai_responses_response.cpp:22` / `anthropic_messages_response.cpp:64` | ✅（小误 #1/#2） |
| 1 | stall 终结链 | `engine_core.h:1109-1185` settle（ownership 校验 `:1128-1133`）；`:2073-2105` fail_all_locked（catch `:2185-2187`）；`resource_manager.h:928-932` `mark_terminal_pending` | ✅（两缺口 → M1/M5） |
| 2 | 容量核算 | `kMaximumConcurrency=8`（`types.h:20`）+ serve 默认 `max_pending_requests=16`（`serve_options.h:33`）→ 24 ≤ 32 | ✅（溢出语义 → M3） |
| 2 | root gating 落点 | `materialization_planner.h:72-80` `best_offered_reuse`（= 各候选 `reusable_prompt_tokens` 的 max，**与物化可行性无关**）；`:100-354` plan()（快速路径 `seal_identity` + bounded A*，root 可在快速路径胜出） | ⚠️ 谓词过强（F-2）→ M2 |
| 2 | decode 轮重准入盲区 | `run_decode_round` 尾部（`engine_core.h:2050-2066`）后无 `request_admission_check` 触发点，盲区时长 = `pending_timeout_ms`（serve 默认 600000ms） | ✅（§3.5 一行修复） |
| 3 | 槽池 / 校验原语 | `state_image_store.h:69-74` StateImageRole；`:209-211` `checkpoint_references`；`program.h:409` `reserved_state` | ✅ 齐备 |
| 3 | `state_slots=2` 管道 | `program_impl.h:9474-9575` 激活期已处理 `reserved_state` 绑定（reserved_states[0]→active、[1]→sequence.reserved_state）；`:9823/9871` `reserve_state_entitlement` | ✅（冲突语义 → M7） |
| 3 | capture 可行性 / skip | `program_impl.h:7420-7660`（`physically_feasible = physical_peak_fits`；HostSnapshot `:7543-7550`）；`:7822-7826` skip = `!pressure && !physically_feasible` | ✅ |

### 11.3 方案小误修正记录（7 项，均已在本版修正）

| # | 位置 | 原文 | 修正后（实测） |
|---|---|---|---|
| 1 | §1.2 / §2.4 / §10 | `types.h:552`（FinishReason） | `types.h:555` |
| 2 | §2.5 / §10 | `program_impl.h:11688`（错误串） | `program_impl.h:11695`（`"ordinary batch row is not decode-ready"`） |
| 3 | §2.1 | DFlash 基线 entitlement = `pages_for_tokens(prompt_tokens)` | = `pages_for_tokens(reserved_context_tokens)`（`request_plan_impl.h:246-282`） |
| 4 | §2.1 | `create_active` nullopt = 通用请求级拒绝通道 | 仅覆盖**静态**（地址空间）容量；**运行期**池耗尽 = `std::bad_alloc` 异常（`paged_kv_cache.cpp:268`），经 `can_resize_reservation` 预检 |
| 5 | §1.1 / §3.5 / §7 | `pending_timeout_ms` 默认 30000ms | serve 默认 **600000ms**（`src/serve/serve_options.h:33-34`），30000ms 为部署配置值 |
| 6 | §2.2 | limit_reason 预计算规则与页池容量比较 | 保持基线语义（M6）：`requested <= capacity_output ? OutputLimit : ContextCapacity`，池维度全由 stall 覆盖 |
| 7 | §3.4 | "`demand_window_` 结构零改动"（无条件表述） | 条件成立（M3）：`PrefixDemandRecord`（`resource_manager.h:122-126`）无 owner 字段 → 增 owner 或文档化 lazy expiry；+ 运维约束 ≤32 |

### 11.4 M1–M7 修正记录（均已在本版落实）

| # | 修正 | 要求 | 落位 |
|---|---|---|---|
| M1 | stall 前置与时序 | `mark_capacity_stalled` 前置增 `!capture_pending`（违反 → settle ownership 校验 throw `"terminal-pending request has invalid ownership"` → fail_all，最致命风险）；探测/stall 早于该轮首次 `build_round_membership`（`engine_core.h:2152`）；stall lane 排除出后续所有 membership（零预算 lane 进 membership 必 throw） | §2.5 步骤 2/3 |
| M2 | root gating 谓词细化 | 仅在匹配候选 **feasible 或可经进展变得 feasible**（planner 诊断 `stop_reason`/`budget_exhausted` 可辨）时禁 root；匹配候选结构性死亡（prefix 物理已丢）→ 放行 root（基线行为同为 root prefill，成功） | §3.3 |
| M3 | 排队需求登记 | `note_pending_demand` 纯窗口插入（不触发 explicit_credit 清除侧效应，`resource_manager.h:2457-2480`）；`PrefixDemandRecord` 增 owner 字段（最小改动）或文档化 lazy expiry；运维约束 `max_pending_requests + active ≤ 32` | §3.4 / §7-I5 / §8 |
| M4 | E3 重构 | 新 entitlement = pages(prompt) 后，prefill 只从激活期已预留页（`tail_reservation_` `logical_kv_store.h:223-235` 持有，他者不可取）物化 → prefill 点池枯竭**结构性自消**；prefill 路径增长失败保留为防御 throw；**P0 不为 prefill lane 加 stall 信号通道**（`advance_prefill` 仅 PrefillProgress/异常通道，engine `catch(...)` 无法区分容量与 bug） | §2.5-E3 |
| M5 | stall 专用 settle | `program.finish` 非 Consumed → throw `std::logic_error`（暴露缺陷），**不走自动 abort 回退**（回退 `clear_catalog_entry` 丢该请求自己的 endpoint 目录条目 → 违反目标 2；stalled 请求带 fork_pending 且 `abort_fork` 失败时触发）；单测覆盖 fork_pending 场景 | §2.5 步骤 4 |
| M6 | limit_reason 基线 | 保持基线预计算语义，不引入 limit_reason 与页池容量纠缠的新规则；池维度全由 stall 覆盖（E1 覆盖规则不变） | §2.2 |
| M7 | 状态槽冲突语义 | root `state_slots=2` 激活期 `reserve_state_entitlement`（`program_impl.h:9823/9871`）失败（如池=2、双 root run 并发）→ **请求级拒绝**，不得阻塞引擎 / 不得 fail_all | §4.2 机制 A |

### 11.5 行为变更面（审查结论）

- **有意（4 项）**：① 池耗尽引擎停摆（fail_all）→ 请求级 `ContextCapacity` 正常终结（目标 1 明言）；② root gating 等待（M2 细化后：仅 feasible/可进展时拦截，结构性死亡放行）；③ entitlement 收缩（prompt 全跨 → 仅 prompt）→ 准入更松、并发页利用更满；④ root `state_slots=2` 临时占一状态槽（M7：冲突 = 请求级拒绝）。
- **副作用（2 项，语义正确）**：⑤ E1 reason 覆盖（池先枯竭 → ContextCapacity，Anthropic 协议要求区分 max_tokens 与上下文窗口）；⑥ E4 统计口径（`device_pages` 由 `entitlement - mapped` 改为按 `mapped` 计，动态页下原口径系统性高估）。
- **意外（1 项，已消除）**：F-2 root gating 过强 → `QueueTimeout`（serve 默认 600s）替代基线可成功的 root prefill —— M2 消除。

### 11.6 审查证据索引（补充）

| 内容 | 位置 |
|---|---|
| fail_all 链 / ownership 校验 / 轮成员构建 | `engine_core.h:2073-2105`（fail_all_locked）、`:2185-2187`（catch → fail_all）、`:1128-1133`（settle ownership 校验）、`:2152`（build_round_membership）、`:2050-2066`（run_decode_round 尾部） |
| stall 原语 / demand window / 压力输入 | `resource_manager.h:928-932`（mark_terminal_pending）、`:122-126`（PrefixDemandRecord）、`:1423`（kDemandWindowCapacity=32）、`:2457-2480`（commit_demand + credit 侧效应）、`:1967-2100`（build_pressure_inputs） |
| finish / terminal commit / state_slots=2 管道 | `program_impl.h:9149-9219`（finish，lifecycle != Finishable → 空）、`:12158-12236`（terminal commit，`:12231` lifecycle/pending 迁移）、`:9474-9575`（激活 reserved_state 绑定）、`:9823/9871`（reserve_state_entitlement）、`:7420-7660`（capture 可行性）、`:7822-7826`（skip 分支） |
| prefill 链（M4 依据） | `engine_core.h:1506-1550`（resolve_prefill_progress）、`:1554-1601`（run_prefill_step，异常 → catch → fail_all） |
| 计划期基线 / 页池 / 服务默认值 | `request_plan_impl.h:246-282`；`paged_kv_cache.h:226-228` / `paged_kv_cache.cpp:268`；`serve_options.h:33-34` |
| planner 结构 / 线程模型 | `materialization_planner.h:72-80,100-354`；`engine_core.h:251/283/302/317/361/395`（入口锁）、`:2226-2227`（auto-save 锁注释） |
| 独立审查报告 | `PREFIX-PLAN-REVIEW.md`（34352B / 181 行） |
