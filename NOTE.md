# NOTE.md — PREFIX-PLAN v2 实施工作日志（持久记忆）

> 用途：所有中间勘察结论/设计定案/实施状态落盘于此，防上下文压缩丢失；上下文被压缩后**反复重读本文件**恢复状态。
> 权威规格：`PREFIX-PLAN.md`（v2，376 行/56948B）；审查报告：`PREFIX-PLAN-REVIEW.md`（181 行）；分支全景：`KVCACHE.md`（984 行）。
> 仓库：`C:\Workspace\codes\ninfer-4090-kaso`，分支 prefix-v2，HEAD 4f0faf0c。
> 工作树约定：git status 只有未跟踪的文档（KVCACHE.md/PREFIX-ISSUS.md/PREFIX-PLAN.md/PREFIX-PLAN-REVIEW.md/.tmp-prefix-plan/），无已跟踪文件被改动。

## 0. 任务与约束

目标：按 PREFIX-PLAN.md v2 完成三支柱实施。
- P0 = 支柱一动态 KV 页：entitlement=pages(prompt)、增长闸门 KvGrowth{Ok,PoolExhausted,InvalidTarget}、probe_decode_capacity+mark_capacity_stalled、stall 专用 settle 断言 Consumed（M5）、E4 统计按 mapped、§3.5 decode 轮尾补 request_admission_check。
- P1 = 支柱二 prefix 硬保护（M2 root gating 谓词、owner 集 P 压力约束、note_pending_demand owner 字段、重准入）。
- P2 = 支柱三 StateImage 槽解冻（机制 A root state_slots=2+M7、机制 B 锚点图像回收、机制 C drop_superseded_anchor 七项校验）。
- 每阶段：Docker `ninfer-local-build:latest`（仓库挂 /src、build/ 复用、Ninja）编译 + ctest；GPU 验证用 4090 容器 `ninfer-4090-kaso-dev`（bind /ninfer-4090-kaso，端口 1234），**跑前先请用户关常驻推理容器**。
- 用户约束（原话）：改动最小、不引入新 bug、prefix-v1 仅作缺陷参照不照搬；"记得实时更新TODO列表"（todo_write 全量替换）；"别用edit了，换种方式，edit很容易退化"（改用 write 整体重写；但小范围 edit 在后续轮次正常使用——若再退化立即切 write）；**"你可以把中间过程全部落盘至NOTE.md文档，避免上下文压缩导致信息丢失，然后反复重读文件"**（本文件来源）。

## 1. P0 编码定案（设计冻结，实施照此）

### 1.1 logical_kv_store.h（src/targets/qwen3_6/impl/runtime/）
- 命名空间 `ninfer::targets::qwen3_6::detail`；文件头 :15 起前向声明区（`class KVActiveSnapshotReservation;` = :20 附近）→ **在此后加 `enum class KvGrowth : std::uint8_t { Ok, PoolExhausted, InvalidTarget };`**。
- `materialize_to_tokens`（:1437-1469）改造：
  - 签名 `void` → `[[nodiscard]] KvGrowth`；
  - `target < address.page_count` → 仍 `throw std::invalid_argument("KV materialization exceeds active entitlement")`（I1 单调性=bug）；
  - `target > page_capacity_` → 返回 `KvGrowth::InvalidTarget`（原 throw 改状态）；
  - `target == page_count` → 返回 Ok；
  - 新增增长段：`needed = target - page_count; if (needed > address.reservation.pages()) { if (!pages_->physical_pool().can_resize_reservation(address.reservation, needed)) return KvGrowth::PoolExhausted; pages_->physical_pool().resize_reservation(address.reservation, needed); }`
  - 其余（begin/count/added span/predecessor/pages_->materialize/publish try-catch 回滚/retain_active_reference/page_count=target）不动，尾部 `return KvGrowth::Ok;`
- 新增 const 探针 `[[nodiscard]] bool can_materialize_to_tokens(KVAddressSpaceHandle, std::uint32_t tokens) const noexcept`（内联 valid/active/reservation 检查 → target 边界 → needed≤headroom ? true : can_resize_reservation），插在 materialize_to_tokens 之后、commit_frontier 之前。
- **关键结构事实**：`Address.reservation` 是 `DeviceKVPageReservation`（**非 optional**，:1739）；`require_active` 已保证 reservation.valid()；`entitlement(address) = page_count + (reservation.valid() ? reservation.pages() : 0)`（:1822-1825）；`pages_for_tokens = tokens==0?0 : 1+(tokens-1)/64`（:1750-1752）。
- 现状基线（改造前实测）：:1441-1442 闸门 `target < page_count || target > entitlement` → throw；:1443-1446 target>page_capacity_ → throw；:1448+ 增长与 publish 回滚。
- `resize_entitlement` :1423-1430（entitlement<page_count||>page_capacity_ → invalid_argument，否则 resize_reservation(reservation, entitlement-page_count)）；`release_growth_entitlement` :1432-1435（resize 到 0）。

### 1.2 页池原语（src/core/paged_kv_cache.h / .cpp）——语义已彻底核实
- `DeviceKVPageReservation{owner_, pages_}`；`pages()` = **未物化头寸（headroom）**：materialize 成功后 `allocated_pages_ += count; reserved_pages_ -= count; reservation.pages_ -= count`（.cpp:343-346）；dematerialize 反向归还。
- `reserve(pages)` :253（noexcept，pages>available→nullopt）；`available_pages() = capacity - allocated - reserved`。
- `can_resize_reservation(reservation, new) const noexcept` :259-266：`!belongs_to || reservation.pages_ > reserved_pages_` → false；否则 `used = allocated + (reserved - reservation.pages_) + new ≤ capacity`。**这就是增长路径的无异常预检**（grow = new - 旧 headroom 消耗空闲页，与 reserve 同一会计口径）。
- `resize_reservation` :268-273：失败 `throw std::bad_alloc()`；成功 `reserved_pages_ = reserved - old + new; reservation.pages_ = new`。
- `materialize(reservation, count, predecessor)` :275+：`count > reservation.pages_` → throw invalid_argument("Paged KV materialization exceeds reserved capacity")；`count > capacity - allocated` → throw logic_error("Paged KV reservation invariant was violated")（仅在会计不变量被破坏时发生——can_resize 预检后结构性不可能）。
- 结论：**can_resize 通过后 resize+materialize 零异常路径**（物理页由池不变量保证足够）。

### 1.3 调用方更新（!=Ok → throw logic_error 暴露 bug，E2 语义；生产环境 decode 先经 probe，prefill 页已激活期预留结构性安全）
- `program_impl.h` `materialize_sequence_kv` :10755-10769（text :10763 / backend :10765-10767）：两分支各加 `!= KvGrowth::Ok` → `throw std::logic_error("KV ... growth failed outside its reservation")`。
- `program_impl.h:1097`（predictor/MTP 预测器 materialize，text_kv_addresses）：防御性 `!= Ok` → throw。
- `program_impl.h:10925`（`addresses.materialize_to_tokens(*allocation, 1, device.stream)`，capture 行）：防御性 throw。
- `session_snapshot_impl.h:866`：防御性 throw。
- 测试 `tests/targets/qwen3_6/test_context_store.cpp` 13 处调用（:187/200/322/347/380/402/435/449/498/512/526/533 等）忽略返回值 → `[[nodiscard]]` 下须处理：成功断言 `== KvGrowth::Ok`；**须逐一检查是否有断言 throw（原 entitlement 越界用例）并更新**（基线越界→throw，新语义→PoolExhausted 或 Ok（池够时））。

### 1.4 request_plan_impl.h（src/targets/qwen3_6/impl/runtime/）entitlement 动态化（§2.2）
- 基线 :246-282 实测：`capacity_output = capacity - prompt_tokens + 1`；`effective_output_tokens = min(requested, capacity_output)`；`effective_limit_reason = requested <= capacity_output ? OutputLimit : ContextCapacity`（**保持不动，M6**）；`reserved_context_tokens = prompt + (effective==0 ? 0 : effective-1)`；`text_kv_page_entitlement = pages_for_tokens(reserved)`；MTP backend = `pages_for_tokens(min(capacity, reserved+draft_window-1))`；DFlash 基线 = `pages_for_tokens(reserved_context_tokens)`。
- 改为：**text = `pages_for_tokens(prompt_tokens)`**；**MTP backend = `pages_for_tokens(min(capacity, prompt_tokens + draft_window - 1))`**；**DFlash = `pages_for_tokens(prompt_tokens)`**；删除/不再使用 reserved_context_tokens 的预留语义（effective 字段仅留作 limit_reason/预算，不进预留量）。
- `root_active.state_slots = 1U`（:420，P2 机制 A 改 2）。
- prompt-fit guard（池级）改请求级正常拒绝（经 can_resize_reservation 预检不抛异常）。

### 1.5 Program 新 API（program.h 声明 + program_impl.h 实现，ProgramImplCore）
- `KvGrowth probe_decode_capacity(std::uint32_t lane) const`（纯查询，const）：
  - lane ≥ max_concurrency / lifecycle != Active / !sequence / !sequence->kv → `InvalidTarget`；
  - **ordinary**（speculative_backend==None）：text 目标 = `frontier + 1`；
  - **MTP**（实测公式 :11940-11949）：`max_by_budget = remaining > 1 ? remaining - 1 : 0`；`extent = min{sequence.mtp_draft_count, draft_window, max_by_budget, capacity - frontier - 1}`；text = `frontier + extent + 1`；backend = `min(capacity, frontier + extent + draft_window)`；
  - **DFlash**（草稿公式，恢复后须先重读 :12030-12120 核实）：extent = min{draft_window, budget, capacity-frontier-1}；text = frontier+extent+1；backend = frontier；
  - 判定：`text_kv_addresses->can_materialize_to_tokens(kv->text, text_target)` 且 backend（如有）同 → 否则 `PoolExhausted`；**不查 StateImage 槽**（decode 目的槽 lane 自持，:10329 区）。
  - `remaining` 取自 `requests[lane].budget`（optional<GenerationBudget>，GenerationBudget::remaining()）。
- `bool mark_capacity_stalled(std::uint32_t lane)`（纯 host 状态迁移，无 CUDA）：
  - 前置：`!has_context_transaction()`、`!pending_transaction_`、`requests[lane].lifecycle == Lifecycle::Active`、lane 有效；任一不满足 → 返回 false（engine 侧 throw 暴露）。
  - 迁移（镜像 terminal commit :12231）：`requests[lane].lifecycle = Lifecycle::Finishable; requests[lane].pending = {};`（+ 若 MTP：`sequence.mtp_draft_count = 0`）；不动 ledger/text_kv_valid/prefix_identity/frontier（stall 请求上一轮 commit 后 frontier 不变量已成立）。
  - **不**做 settle_state_fork/trim（probe 时刻上一轮 pending 已解析）。
- engine 侧 stall 链（engine_core.h，src/runtime/engine/）：
  - `RequestRecord`（request_record.h :105-224）加 `bool capacity_stalled = false;`（放 terminal_reason :187 附近）。
  - worker_loop（:2079-2204，execution_mutex_ 跨整个 try）插入点 = **第一次 cancel_active_requests(:2121) 后、第一次 build_round_membership(:2122) 前**：
    1. 遍历 DecodeReady 且 `!capture_pending` 的 slot（M1）：`program->probe_decode_capacity(lane)`；
    2. `PoolExhausted` 时：`request->budget->commit(budget->remaining())`（清零安全，generation_budget.h commit(>remaining) 才 abort）；
    3. `output.preview_terminal(FinishReason::ContextCapacity)`（镜像 cancel 的 preview 调用）；
    4. `program->mark_capacity_stalled(lane)` → false 则 `throw std::logic_error(...)`（暴露，E2 语义）；
    5. `resources_.mark_terminal_pending(LaneId{lane})`（resource_manager.h :927-930 Active→TerminalPending，复用为 note_capacity_stall，不新写 API）；
    6. `request->capacity_stalled = true; request->model_state = EngineRequestState::ModelFinished; request->terminal_reason = FinishReason::ContextCapacity;`
  - 效果：stall lane 的 state=ModelFinished → 不进入后续 build_round_membership 的 decode membership（零预算 lane 进 membership 必 throw "ordinary batch row is not decode-ready" program_impl.h:11695）→ 无 E2 触发。
  - 下一轮 settle_terminal_requests（:1113-1185）ownership 校验（:1128-1133：is_model_finished && !capture_pending && sequence && lane 匹配 && lane_state==TerminalPending，否则 throw logic_error("terminal-pending request has invalid ownership")）→ resources_.finish（:1142）→ **M5：capacity_stalled 请求走 strict finish——program.finish 非 Consumed → throw logic_error（绝不 fallback program.abort → clear_catalog_entry 丢该请求 endpoint 目录条目 → 违反目标2）**；非 stall 请求保持基线 fallback 行为。
  - finish 成功后 Catalogued → 记 retained_slot/retained_session_digest → complete_success(request, ContextCapacity)（:1158，reason 通用）→ remove_completed_slot（:1091-1094，内部 :1093 触发 request_admission_check 排队补位）。
  - Protocol 映射零新增：OpenAI "length"（openai_chat_response.cpp:46）、Anthropic "model_context_window_exceeded"（anthropic_messages_response.cpp:64）、FinishReason::ContextCapacity（include/ninfer/types.h:555）。
- §3.5：run_decode_round 尾部（decisions 循环+terminal 结算后、publish 前）补一行 `request_admission_check()`（既有 gate should_attempt_admission :233-237 含 previous_unit_was_decode 可放行）。

### 1.6 E4 统计（program_impl.h :6718-6730 add_kv 闭包）
- `device_pages += entitlement - mapped;`（:6728）→ `device_pages += mapped;`（host 分支不变）。

### 1.7 边缘语义
- E1：预计算 limit_reason=OutputLimit 但池先枯竭 → stall 的 terminal_reason 恒 ContextCapacity（preview_terminal 用 stall 自身 reason，覆盖基线预计算）。
- E2：行内（decode_raw 前）防御预检失败 = bug → throw 走 fail_all 暴露（刻意不静默）。
- E3：prefill 只从激活期已预留页（tail_reservation_ logical_kv_store.h:223-235）物化 → 池枯竭结构性不可能；prefill 路径保留防御 throw；P0 不做 prefill stall 通道（M4）。
- M7（P2）：root state_slots=2 槽冲突 = 请求级拒绝，不得 fail_all。

### 1.8 关键不变量/错误串（写单测时引用）
- "ordinary batch row is not decode-ready"（program_impl.h:11695）；"MTP batch row is not decode-ready"（:11833 区，检查 mtp_draft_count>draft_window）；"terminal-pending request has invalid ownership"（engine_core.h:1128-1133 区）；"KV materialization exceeds active entitlement"（logical_kv_store.h:1442，仅 target<page_count 分支保留）；"Paged KV materialization exceeds reserved capacity" / "Paged KV reservation invariant was violated"（paged_kv_cache.cpp materialize）；"KV entitlement is smaller than mapped pages"（resize_entitlement）。
- fail_all_locked（engine_core.h:2073-2105）：permanent 引擎停摆（failed_=true、pending 清空、program->fail_all_cleanup）——任何未受控 logic_error 都会触发，故前置条件必须严。
- GenerationBudget（generation_budget.h:11-45）：构造 (effective_tokens, limit_reason)∉{OutputLimit,ContextCapacity} → std::abort()；commit(tokens>remaining) → std::abort()。
- ProgramImplCore::finish（program_impl.h:9149-9219）前置：has_context_transaction||pending_transaction_||!valid_sequence → 空；**lifecycle != Finishable → 空**；!publish_continuation → Released（Consumed 不 catalog）；catalog 路径 fork_pending→abort_fork+release；role 须 ActiveMutable→freeze 或 CheckpointImmutable；release_sequence_growth_entitlement :9206；lifecycle=Empty。
- cancel_active_requests :1170-1196（capture_pending 跳过 :1174；resources_.abort 丢 continuation 不 catalog → stall 必须走 finish）。

## 2. 已核实但未读的代码区（恢复后按需重读）
- engine_core.h：:2009-2069 run_decode_round 本体尾部（补 request_admission_check 的精确插入点）；:2106-2204 worker_loop 全文（插入点 :2121-2122 已定）。
- resource_manager.h（src/runtime/engine/）：mark_terminal_pending :927-930 精确签名、finish :932-1000（加 strict 参数/重载的落点）、FinishDisposition 枚举。
- program_impl.h：:1090-1100（predictor materialize 上下文）、:10920-10930（capture 行）、:12030-12120（DFlash growth 公式核实）、:12158-12240（terminal commit 迁移，:12231 lifecycle/pending 迁移）、:10329（StateImage 槽 lane 自持）。
- request_plan_impl.h :240-290 与 :415-425 原文。
- program.h :1240-1270（新 API 声明插入区）。
- tests/targets/qwen3_6/test_context_store.cpp 全文（更新 13 处调用 + 查找 throw 断言用例）。

## 3. 实施进度（随做随改）
- [ ] P0-a logical_kv_store.h：KvGrowth 枚举 + materialize_to_tokens 改造 + can_materialize_to_tokens  ← **下一步**
- [ ] P0-b 调用方更新（program_impl.h ×3、session_snapshot_impl.h ×1、test_context_store.cpp）
- [ ] P0-c request_plan_impl.h entitlement 动态化
- [ ] P0-d program.h/program_impl.h 新 API（probe_decode_capacity / mark_capacity_stalled）
- [ ] P0-e engine_core.h stall 链 + request_record.h capacity_stalled + resource_manager.h strict finish（M5）
- [ ] P0-f decode 轮尾补 request_admission_check
- [ ] P0-g E4 统计改 mapped
- [ ] P0-h 单测（KvGrowth 耗尽/I1/can_materialize 一致性；stall settle 保 Catalogued 含 fork_pending）
- [ ] P0-i Docker ninfer-local-build 编译 + ctest
- [ ] P1 / P2（P0 通过后再展开）

## 4. 环境备忘
- 本机无 GPU；编译/ctest：`docker run --rm -v <repo>:/src -w /src ninfer-local-build:latest`（build/ 复用，Ninja；具体命令沿用仓库内已有脚本/此前用法）。
- 4090：容器 ninfer-4090-kaso-dev（bind /ninfer-4090-kaso，端口 1234）；跑 serve/GPU 测试前**先请用户关闭其常驻推理容器**。
- DSH file policy = danger-full-access；approval prompts 已禁用（不得设 sandbox_permissions）。

## 4. P0-a/b 完成记录 + P0-d 最终定案（编码前必读）

### 4.1 P0-a 完成（logical_kv_store.h → 93252B/1908 行，CRLF）
- R1: detail 前向声明区（`class KVActiveSnapshotReservation;` 后）加 `enum class KvGrowth : std::uint8_t { Ok, PoolExhausted, InvalidTarget };`（namespace ninfer::targets::qwen3_6::detail，KvGrowth 现 :27）
- R2: materialize_to_tokens :1437 起改 `[[nodiscard]] KvGrowth`：target<page_count → throw invalid_argument("KV materialization exceeds active entitlement")（I1 原串保留）；target>page_capacity_ → KvGrowth::InvalidTarget【基线无此分支：移除 entitlement 守卫后必须显式补，否则 memberships span 越界写穿】；target==page_count → Ok；needed=target-page_count > reservation.pages() 时 can_resize_reservation(reservation,needed)==false → PoolExhausted，否则 resize_reservation 原子批
- R3: 尾部 `address.page_count = target; return KvGrowth::Ok;`
- R4: commit_frontier 前插入 `[[nodiscard]] bool can_materialize_to_tokens(handle,tokens) const noexcept`（valid/active/row/reservation.valid 检查→target 边界→needed<=headroom→can_resize_reservation）
- 实测修正：基线 materialize_to_tokens 无 page_capacity 检查（此前误记 :1443-1446 有该 throw）；publish try/catch 回滚 dematerialize 未动、异常仍上抛（E2）

### 4.2 P0-b 完成（调用方适配 !=Ok→throw）
- apply_p0b.ps1 v2（v1 多行 here-string 因续行 52 空格对齐脆弱 0 命中失败；v2 = 按行 -eq 精确匹配 + 数组拼接 + R1 正则容忍续行）全部成功：
- src/targets/qwen3_6/impl/runtime/program_impl.h 677484B：materialize_sequence_kv 内 text 调用→`!= KvGrowth::Ok` throw logic_error("text KV growth failed during sequence materialization")；backend 调用→"backend KV growth failed during sequence materialization"；causal 行（:1097 区）→"causal score KV materialization failed"；capture 行（:10925 区）→"capture KV materialization failed"
- src/targets/qwen3_6/impl/runtime/session_snapshot_impl.h 51954B：:866 行→"session snapshot KV materialization failed"
- tests/targets/qwen3_6/test_context_store.cpp 37924B：12 处调用改 `expect(<call> == store::KvGrowth::Ok, "KV materialization returns Ok");`（测试 12 场景全 target≤entitlement→行为不变；构建无 -Werror 全 CMake grep 零命中）
- 生产 materialize_to_tokens 调用方仅 4 处（program_impl.h materialize_sequence_kv/:1097/:10925、session_snapshot_impl.h:866）+ 测试 12 处

### 4.3 P0-c 待执行（request_plan_impl.h entitlement，行号 post-P0-a/b 实测）
- :253-260 保留不动（capacity_output=capacity-prompt_tokens+1；effective_output_tokens=min(requested,capacity_output)；effective_limit_reason=requested<=capacity_output?OutputLimit:ContextCapacity = M6 基线语义）
- 删 `const std::uint32_t reserved_context_tokens = base->summary.prompt_tokens + (base->summary.effective_output_tokens == 0 ? 0U : base->summary.effective_output_tokens - 1U);`（4 行）
- `base->text_kv_page_entitlement = pages_for_tokens(reserved_context_tokens);`→`pages_for_tokens(base->summary.prompt_tokens)`
- MTP backend `pages_for_tokens(mtp_tokens)` 的 mtp_tokens=min(capacity, (uint64)reserved_context_tokens+draft_window-1)→min(capacity, (uint64)base->summary.prompt_tokens + draft_window - 1)
- DFlash backend `pages_for_tokens(reserved_context_tokens)`→`pages_for_tokens(base->summary.prompt_tokens)`
- :284-289 root_active{active_lanes=1,state_slots=1U,main_kv_pages=text entitlement,backend_kv_pages=backend} 不动

### 4.4 P0-d 最终定案（编码落点，全部已实测）
- 公共枚举：export runtime.h（namespace ninfer::targets::qwen3_6，:14 起；`template <class Variant> class Program` 于 :858-859 前）加 `enum class SequenceGrowth : std::uint8_t { Ok, PoolExhausted, InvalidTarget };`（与 detail::KvGrowth 不同名防遮蔽）
- public 声明（runtime.h，has_context_transaction() 附近）：`[[nodiscard]] SequenceGrowth probe_decode_capacity(SequenceHandle<Variant> sequence, std::uint32_t remaining) const;` + `void mark_capacity_stalled(SequenceHandle<Variant> sequence);`
- api_impl.h 透传：`template <> SequenceGrowth Program<Variant>::probe_decode_capacity(SequenceHandle<Variant> sequence, std::uint32_t remaining) const { return impl_->probe_decode_capacity(sequence, remaining); }`（impl_ = ProgramImplCore*；decode/finish 同模式 :503-532）
- core 声明（program.h，类尾 `};` anchor 区、eHZ 前）+ 实现（program_impl.h）：
  - probe（const，纯查询不 throw 语义，前置失败→InvalidTarget）：handle→`SequenceState& state = ...; if (!state.valid) return InvalidTarget; lane=state.lane`→active_sequence(lane)（:6478-6497：lane>=max_concurrency→out_of_range；active_continuations[lane]→index；role!=Active→logic_error("active lane has no continuation binding")；返回 continuation_states[index]）；!state.kv→InvalidTarget；frontier=state.execution_frontier>=capacity→InvalidTarget；公式：ordinary（无 backend）text=frontier+1；MTP：max_by_budget=remaining>1?remaining-1:0，extent=min{state.mtp_draft_count, draft_window, max_by_budget, capacity-frontier-1}，text=frontier+extent+1，backend=min(capacity, frontier+extent+draft_window)；DFlash：extent=min{draft_window, max_by_budget, capacity-frontier-1}，text=frontier+extent+1，backend=frontier（不增长）。text: can_materialize_to_tokens(kv->text,text) false→PoolExhausted；backend 同；否则 Ok。返回 public SequenceGrowth（core 内部 detail::KvGrowth 由 can_materialize 直接判 bool，public 方法直接返回 SequenceGrowth 即可，无需映射层）
  - mark_capacity_stalled(handle)：前置 !has_context_transaction() && !pending_transaction_（否则 throw logic_error("capacity stall overlaps an open context transaction")）；state.valid + kv 检查；lifecycle!=Lifecycle::Active→throw logic_error("capacity stall requires an active decode lane")；迁移：request.lifecycle=Lifecycle::Finishable; request.pending={}; state.mtp_draft_count=0;（probe 在轮首 progress 之后，pending 已解析，无需 settle_state_fork/trim）
- request_record.h：`std::optional<FinishReason> terminal_reason;` 后加 `bool capacity_stalled = false;`
- resource_manager.h finish 加尾参 `bool allow_abort_fallback = true`；`result.status != ConsumeStatus::Consumed` 分支内首行：`if (!allow_abort_fallback) { throw std::logic_error("capacity-stalled terminal settlement did not consume the sequence"); }`（M5：禁 abort fallback→clear_catalog_entry 丢 prefix；基线调用不受影响）
- engine_core.h settle_terminal_requests：`resources_.finish(*instance_.program, *request->lane, *request->sequence)`→加第 4 参 `request->capacity_stalled`
- engine_core.h worker_loop 插入点 = `cancel_active_requests(cancelled_at_boundary, boundary);` 之后、`RoundMembership membership =` 之前（锚 :2110-2122 区）。stall 循环：
  ```cpp
  for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
      const auto& request = slots_[lane];
      if (request == nullptr || !request->is_decode_ready() || request->capture_pending ||
          !request->sequence || !request->lane || request->lane->value != lane) { continue; }
      const std::uint32_t remaining = request->budget ? request->budget->remaining() : 0U;
      if (instance_.program->probe_decode_capacity(*request->sequence, remaining) !=
          targets::qwen3_6::SequenceGrowth::PoolExhausted) { continue; }
      if (request->budget) { request->budget->commit(request->budget->remaining()); }
      (void)request->output.preview_terminal(FinishReason::ContextCapacity);
      instance_.program->mark_capacity_stalled(*request->sequence);
      resources_.mark_terminal_pending(LaneId{lane});
      request->terminal_reason = FinishReason::ContextCapacity;
      request->capacity_stalled = true;
      request->model_state = EngineRequestState::ModelFinished;
  }
  ```
  （引擎 namespace ninfer::runtime 内 `targets::qwen3_6::SequenceGrowth` 经 ninfer:: 外层解析；GenerationBudget::commit(==remaining) 不触发 abort()；preview_terminal 镜像 cancel 路径；stall lane 下轮 settle：ownership 校验全过→strict finish→Catalogued→complete_success(ContextCapacity)→remove_completed_slot 触发排队补位）
- §3.5：worker_loop decode 分支 `run_decode_round(membership, cancelled_at_unit_start); previous_unit_was_decode = true;` 后（continue 前）加 `request_admission_check();`（消除 30s/600s 重准入盲区）
- E4：program_impl.h add_kv 统计 `device_pages += entitlement - mapped;`→`device_pages += mapped;`（overflow guard `entitlement - mapped > max - device_pages` 改 `mapped > max - device_pages`；inconsistency guard 保留）

### 4.5 写 PS1 前须重读的小段（取精确行文本作匹配锚）
1. request_plan_impl.h :272-283（entitlement 3 行 + 4 行局部）
2. export runtime.h :840-865（Public API 声明区，has_context_transaction 前后）
3. api_impl.h :500-535（decode/finish 委托处）+ :405-415（has_context_transaction 委托）
4. program.h :1255-1268（声明插入点）
5. program_impl.h finish 前置区（has_context_transaction/pending_transaction_ 用法，:9149 区已压缩，重读）
6. engine_core.h :2117-2122（worker_loop 插入点）+ :1420-1430（decode commit 尾部→request_admission_check 落点确认：run_control_batch 与 run_decode_round 共用 commit 区？需确认 decode 轮尾=该函数 :1479 前的哪个位置；若 commit 区为共享函数则在函数尾 :1478 `}` 前加）
7. request_record.h :187-190（terminal_reason 锚）
8. resource_manager.h :927-945（mark_terminal_pending + finish 头 + 非 Consumed 分支）

### 4.6 编辑方式（用户三令）
禁止 edit 工具（连续退化）→ write PS1 至 .tmp-prefix-plan/ + pwsh -NoProfile -File 执行。范式：ReadAllText→CRLF 检测→按行精确匹配/正则（[regex]::Escape 计数必须==1）→WriteAllText UTF8 no-BOM→关键串 grep 验证 + bytes/lines 输出。

### 4.7 进度
P0-a ✅ P0-b ✅ P0-c ✅ P0-d ✅ P0-e(§3.5 重准入，随 d) ✅ P0-f(E4，随 d) ✅ P0-g 新单测 ✅ P0-h Docker 编译+ctest ✅ P1 ⬜ P2 ⬜

## P0 编码进度（goal-757fe8a1）
- P0-a ✅ logical_kv_store.h：KvGrowth 枚举 + materialize_to_tokens 闸门（resize_reservation 增长，bad_alloc→PoolExhausted）+ can_materialize_to_tokens（93252B/1908 行）
- P0-b ✅ 调用方 !=Ok→throw：program_impl.h 4 处 + session_snapshot_impl.h 1 处；test_context_store.cpp 12 处 expect(...==store::KvGrowth::Ok)
- P0-c ✅ request_plan_impl.h：entitlement=pages_for_tokens(prompt_tokens)；MTP backend=pages_for_tokens(min(capacity,prompt+draft_window-1))；DFlash=pages(prompt)；删 reserved_context_tokens 局部（75275B/1306 行）
- P0-d ✅ 新 API+stall 链 7 文件：runtime.h（SequenceGrowth 枚举+probe_decode_capacity/mark_capacity_stalled 声明）、api_impl.h 委托、program.h core 声明、program_impl.h（probe 三式 + mark_capacity_stalled 迁移 + E4 统计 device_pages+=mapped）、request_record.h（capacity_stalled 字段）、resource_manager.h（finish 尾参 allow_abort_fallback，M5 strict）、engine_core.h（worker_loop stall 循环 + settle 第4参 + §3.5 decode 尾 request_admission_check）；F6B/F7C 锚点问题经 patch_p0d/patch_f6b/patch_f7c.ps1 修复
- P0-g ✅ test_kv_growth 单测插入 test_context_store.cpp（41376B/681 行）+ main 注册
- 教训：PS1 行正则不用 $ 尾锚（CRLF）；LineSubst 须 [regexoptions]::Multiline；含引号文本用 @'...'@ here-string；锚不唯一时加邻行双行锚
- P0-h ✅ Docker ninfer-local-build:latest 编译+ctest：3 轮修复（runtime.h SequenceGrowth/template 位置、program.h public 可见性、test_context_store.cpp 12 处多余分号）；ninja 101/101 exit 0；ctest 100% passed 0/104（24 passed + 80 GPU-skipped）
