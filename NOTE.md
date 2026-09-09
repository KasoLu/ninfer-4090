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

## 6. prefix_real_test 第三轮失败诊断 + E4 修复（本轮）

### 6.1 诊断（4090 诊断输出）
- 错误：`materialized sequence does not match its active entitlement: state_slots 2 vs 2; main_kv 2 vs 1; backend_kv 2 vs 1`
- 检查点：program_impl.h `start_request`（:7175 区，`start_sequence` 后立即比对 `resident_resources(sequence)` vs `details.demand.active_entitlement`）
- **根因 = 我 P0-f E4 改动的双重计数 bug**：`resident_resources` 的 add_kv 闭包先在循环里 `++device_pages` 每个独占 device 页，然后 active 分支又 `device_pages += mapped`（我改的）→ 全独占序列 actual = 2×mapped。基线是 `+= entitlement - mapped`（全独占时恰好 = entitlement）
- 失败请求 = 任意 prompt≤64 的 root 请求（mapped=1, entitlement=1 → actual=2 > 1）；4090 上 qwen3.8 groupwise 路径首个请求即触发
- backend 同因：backend_materialized = min(capacity, prompt + (initial_mtp_extent==0?0:initial_mtp_extent-1))，initial_mtp_extent = min{draft_window, effective_output>1?effective_output-2:0, capacity-prompt>0?capacity-prompt-1:0}（program_impl.h:4489）≤ draft_window → backend mapped ≤ expected 恒成立，纯粹被双重计数放大
- state_slots 2 vs 2 匹配（P2-A 正确）；host 两侧 0

### 6.2 修复（program_impl.h resident_resources add_kv 尾段）
- **删除 `device_pages += mapped;`**（保留 `entitlement < mapped` 不变量检查 + overflow guard `mapped > max - device_pages`）；顺带修复 P0-f 引入的 `mapped >` 顶格缩进
- 语义（= PLAN E4 本意，PREFIX-PLAN.md:72/122/363）：owner 统计 = 循环已计的独占 mapped device 页；**不加任何 bulk 项**——未物化 reservation headroom 是池级会计（reserved_pages_），动态页下计入 owner 口径系统性高估；共享页（address_references>1）在 owner 释放后仍存活，不属于其转移效应
- 修复后 start_request 检查：root 全独占 → actual == entitlement 相等 ✓；reuse（共享前缀）→ actual ≤ entitlement（放宽的 `>` 检查允许）✓；backend mapped ≤ expected ✓
- `resident_resources` 其余消费者（request_plan_impl.h:814 source、5878 delta.removed 等）均为 cataloged/共享（inactive）路径，语义不变

### 6.3 状态
- 本地 Docker 增量构建 + ctest 回归进行中；之后 commit+push、4090 pull+rebuild、用户重跑 G1/G2
- 4090 常驻推理容器现为 `quizzical_robinson`（romantic_leavitt 停掉后自动重启的新名，同镜像 0908-villina-prefix，端口 1234，GPU 全占）；**用户自行操作停/起**
## 7. 系统性审计（用户："你再整体检查一遍，是否还存在这种潜在的问题"）→ P2-A 容量缺陷 + 修复 3a6b7d06

### 7.1 审计结论（逐项，全部代码级核实）
- ① P2-A state_slots × device 容量：**发现唯一真缺陷**（见 7.2），已修复
- ② probe_decode_capacity 三式 vs decode 实际增长：ordinary（program_impl.h:11860 `materialize(frontier+1, 0)` = probe `frontier+1`）、MTP（:12019-12020 `materialize(frontier+extent+1, min(capacity, frontier+extent+draft_window))`，extent=min{mtp_draft_count, draft_window, max_by_budget, capacity-frontier-1}，max_by_budget=remaining>1?remaining-1:0 —— 与 probe :9184-9219 逐字段一致；remaining 来自同一 GenerationBudget，probe 与 decode 同一 worker 迭代无交错）、DFlash（:12206 `materialize(frontier+extent+1U, frontier)`，extent=min{draft_window, max_by_budget, capacity-frontier-1} —— 一致）→ probe Ok ⇒ decode 必 Ok，无 E2 误触发
- ③ resident_resources(SharedPrefixState) shared 版（:6749+）：add_kv 只计 address_references==1 页，无 bulk 项 → 无双重计数
- ④ 全局统计 device_main_kv_occupied_pages = program.physical_usage()（resource_manager.h:1122 ← program_impl.h:9559 ← physical_occupancy :6815-6818 = pool.allocated_pages()+reserved_pages()）：旧设计 allocated+reserved ≡ entitlement 足迹，新设计 ≡ mapped+headroom，两式数值相同 → 测试基线（127 页、>0&&<=8）仍有效
- ⑤ 压力/capture effect：resident_resources 修复后对 active 少计 headroom（reservation 是池级会计）→ 所有消费方（effect.removed/reservation）偏保守，准入更严不会更松 → 无未定义行为
- ⑥ resize_sequence_kv_entitlement（PrivateEndpoint/rewrite 复用路径 :9936/:9984）：append-only 保证 pages(new_prompt) ≥ pages(base)=trim 后 page_count，MTP/DFlash backend 同单调 → resize_entitlement 前置 entitlement≥page_count 恒成立
- ⑦ 非 root 计划/实际状态槽一致性：converted=min(active, preC)，P2-A 的 +1 位移（S:1→2，count 同步 +1）保持 WSI `count==S` 与 start 检查 `actual==expected` 的相等关系不变 → P2-A 未引入新不一致，只在容量不足时失败（即 7.2 缺陷）
- host 检查 actual.host != expected.host：h2d 恢复后 host 副本收敛语义为基线（基线全字段严格相等已验证过 4090），P0/P2 未改 host KV 会计
- Retain+DeviceOnly 的 `++conversions.state_slots`（request_plan_impl.h:1113-1120，!state_fork_required 时 +1）语义存疑（Retain 下 source 保留副本，credit 似乎多计），但为基线逻辑且 P2-A 的 +1 位移不改变其一致性 → 不修，若 scenario（source-pressure-protection）GPU 跑挂再查

### 7.2 P2-A 容量缺陷（真 bug，commit 3a6b7d06 修复）
- 缺陷：request_plan_impl.h:370（修复前）`root_active.state_slots = 2U;` 无条件 → root_demand.reservation_added.device.state_slots=2 → prepare_materialization（program_impl.h:4811-4876）按 `state_count = reservation_added.device.state_slots` 循环 `state_store->reserve_destination()`（state_image_store.h:723-731 要求空闲 **Device** 槽）→ device_state_slots=1 的引擎上根请求**永远无法准入**（死锁在首个 engine.generate）
- 受害引擎：tests/targets/qwen3_6_27b/test_engine_prefix_real.cpp `host_restore_engine_options`（:45-46 device=1/host=2）与 `shared_replacement_engine_options`（:65-66 device=1/host=4）—— E4 双重计数 bug 修复后 prefix_real_test 推进到这两个引擎必挂
- 修复：`root_active.state_slots = (state_store != nullptr && state_store->device_capacity() >= 2U) ? 2U : 1U;`（容量自适应；device_capacity()=state_image_store.h:137 池总槽数，引擎内恒定）→ 单槽引擎回退基线行为（终结时 active image 原地 freeze 即 capture destination，单槽内无争用，P2-A 保证平凡成立）；≥2 槽引擎 P2-A 完全生效
- 单槽引擎的 fallback 安全性：Both 分裂只加 host 副本不占 device 槽（reserve_logical_destination）；h2d restore 走 take_device_slot 直取；ConsumeToActive/HostOnly/Retain 各路径的 count/S +1 位移分析（7.1⑦）确认无二次不一致
- 验证：本地 Docker ninja 32/32 + ctest 104/104；push a066cb15..3a6b7d06；4090 pull 3a6b7d06 重建中
- 测试 fixture 不改（device=1 正是验证 host 路径的用意；改 fixture 会削弱覆盖）

### 7.3 状态
- prefix-v2 提交链：4f0faf0c → 2c26c74f → 809537ff → bbb41e3b → a066cb15 → 3a6b7d06（HEAD，4090 已同步）
- 待用户：停 quizzical_robinson 后重跑 G1/G2；本轮修复预期使 prefix_real_test 能推进过 host_restore/shared_replacement 两个单槽引擎

## 8. P1 硬保护 UAF 根因确诊 + M3 定点出窗修复（83f60ca9 之后）

### 8.1 根因：build_pressure_inputs 的 protected_owner_ids 是 lambda 局部 → 返回 span 悬空（UAF）
- 4090 trace（NINFER_ADMISSION_TRACE=1，容器 66e132d9aa6a，日志 /ninfer-4090-kaso/admission_trace.log）显示：
  planner root 的 protected= 列出大数值 {1948717728, 28763, 1948718048, ...} 且有重复；
  而真实集合在 builder 内有去重（if (!already) push）→ 重复值只可能是**已释放堆内存的垃圾**。
- 机制：resource_manager.h 的 build_pressure_inputs lambda 体内声明 `std::vector<PlanningOwnerId> protected_owner_ids;`
  （原 :2143 区），返回 `PressureInputs{.protected_owners = protected_owner_ids}` 的 span 指向 lambda 栈帧；
  该 lambda 作为 PressureInputsFn 传入 planner_.plan()（:2258），在 lambda **返回之后**才被调用（materialization_planner.h:249
  `const PressureInputs pressure = pressure_inputs();`）→ 之后对 pressure.protected_owners 的每次读取都是 UAF。
  trace 的 size()（6/8）是返回前拷贝的真实值，值内容是释放后被复用的堆垃圾。
- 误剪：`po == outcome.owner` 比较中垃圾值 0/1 与真实 positional owner ID 巧合相等 →
  `PRUNED (hard protection; evicted owner=0)` 是**假阳性** → no selection + idle → engine_core.h:1893 抛
  "isolated-feasible request is blocked in an idle Engine"。整个 P1 硬保护自引入起一直在读悬垂内存。
- 对照：capture builder（:789 区）同名 lambda 局部变量**安全**——其 Input 的 span 在 lambda 体内即被
  capture_planner_.plan() 消费（lambda 返回前），不悬空。
- 附带发现：demand window 无定点出窗（note_pending_demand 只进不出，靠 32 槽环形挤出；旧场景的陈旧记录
  会持续保护旧条目）——规格（PREFIX-PLAN.md §3.4 生命周期"排队 → admit/取消/超时/终结（出窗）"）要求但实现缺失。

### 8.2 修复（.tmp-prefix-plan/patch_uaf_fix.ps1，已验证落盘）
1. resource_manager.h：`protected_owner_ids` 声明从 lambda 体提升到函数作用域（:1994，与其他 builder 向量并列）
   → span 生命周期覆盖整个 plan() 调用。capture builder 的 :789 不动（本就安全）。
2. resource_manager.h（note_pending_demand 后，~:2540）新增：
   `void clear_pending_demand(std::uint64_t owner) noexcept` —— 删除 demand_window_ 中
   `owner == 该请求id && !selected_source_key && exact_resident_keys.empty()` 的记录（= 纯 pending 记录）。
   安全前提（已核实）：request id 从 1 开始（engine_core.h:2254 `next_request_id_ = 1`）；committed 记录 owner 恒为 0
   （inspect 路径从不设置，commit_demand 在 resource_manager.h:3160 的发布路径调用）→ 永不误删 committed。
3. engine_core.h on_waiting_removed（:1598-1601）加 `resources_.clear_pending_demand(request->id);`
   —— 单一汇合点覆盖全部出队路径（admit :1848/:1932 区、cancel/expired :1214-1215、error :1627、
   准入期取消 :1760、Aborted :1807）；线程/锁纪律与既有 note（submit 侧）/commit（worker 发布路径）一致，未引入新锁。

### 8.3 状态
- 本地 docker ninfer-local-build:latest build+ctest：进行中（后台 pwsh-14，期望 32-35/35 + 104/104）。
- 下一步：commit+push → 4090 pull+ninja → 用户停 GPU 占用后重跑
  `export NINFER_ADMISSION_TRACE=1 && ctest --test-dir build --output-on-failure -R ninfer_qwen3_6_27b_prefix_real_test`
  → 读新 trace：若 protected= 全为小 positional 值且仍 PRUNED → 真·保护/容量死锁（下一层：head 豁免或 P2-A 双槽贪婪）；
  若无 PRUNED 且通过 → UAF 即根因，收尾（考虑摘除 trace 门控或保留）。
## 9. part 3（P1 硬保护作用域）根因与修复（commit e341e68f）

### 9.1 背景：UAF+M3 修复后本地 ctest 仍失败 1 例
- `ninfer_resource_manager_test`（tests/test_resource_manager.cpp:2525，test `test_retained_source_is_protected_until_terminal` :2505-2533）
- 断言：`FAIL retained source protection: terminal release did not return retained source to pressure policy`
- 测试流：make_manager(2,3)；seed（digest 9，session1，LiveSession）start+finish（目录化）；fork（digest 9，session2 → Retain）start 后 abort；`inspect(77)` 期望有 choice → 实际 TB
- 真实 trace（带 NINFER_ADMISSION_TRACE=1）：`planner root: status=0 owners=1 outcomes=0:1 protected=0` → `PRUNED (hard protection; evicted owner=0)` → `inspect: blocked (no selection; candidates=1)` → FAIL
- protected={0} 是**真实**集合（非 UAF 垃圾）

### 9.2 根因（part 3，代码级确诊）
- `demand_window_` 同时含 pending 记录（note_pending_demand）与 committed 记录（commit_demand 于发布路径 push，owner 恒 0）；硬保护推导（resource_manager.h 原 :2142-2151）用 `demand_mask_for(checkpoint_key, provisional)` = **全部** window 记录 + provisional
- `demand_matches`（resource_manager.h:1472-1480）= candidate_keys / exact_resident_keys / selected_source_key 任一精确相等的 OR 语义
- fork 的 committed 记录采纳时 `exact_resident_keys` = seed 条目 key（fork 精确匹配 seed 条目）→ seed checkpoint key 匹配 fork 记录 → fork 终结（abort）后 seed owner 仍被**永久**硬保护（直至环形挤出）→ 违背规格 §7-I5「owner 仅在匹配 active/pending 请求时受保护；终结即释放」
- 4090 映射：每个终结的 reuse 请求都留下 committed 记录钉住其匹配的旧 prefix → 保护集累积 → root 候选全被剪枝 → IDLE-BLOCK
- 测试 fake key（tests/test_resource_manager.cpp:223-238）：`prefix_shortlist_key(frontier) = (shortlist_digest, frontier)`，`make_base(digest)` 设 shortlist_digest=digest、opportunities 空 → 无需 digest 碰撞，exact_resident_keys 即足以触发

### 9.3 修复（commit e341e68f，3 文件 +46/-3）
1. `PrefixDemandRecord.pending_origin`（resource_manager.h:129）：note_pending_demand 置 true（:2556）；committed 记录默认 false
2. `protection_mask_for(key, provisional)`（resource_manager.h:1532）：window 仅匹配 `pending_origin` 记录 + provisional 位
3. `MaterializationCheckpointPolicy.protection_mask`（materialization_planner.h:30，结构体尾加字段保 designated initializer 合法）；materialization 两 builder 填值（私有 :2087 / 共享 :2147）
4. 硬保护推导（resource_manager.h:2157）：`cp.demand_mask == 0` → `cp.protection_mask == 0`
5. **capture builder 未动**（其 protected 集仍基于 committed demand，resource_manager.h:788-800）——保守点，记为已知项
- 语义：硬保护 = 当前 inspect 请求 provisional + 在队 pending；committed 记录仅留软 credit（符合 I5「终结即释放」与单测 terminal release 期望）

### 9.4 PS1 踩坑记录（patch_part3.ps1 → repair_r4.ps1 → fix_r4_order.ps1）
- R5 尾锚 `*entry.handle, entry.summary.checkpoint.ref)),` 出现 2 处（capture :733 + shared builder :2122）→ Trim 匹配 + 取最后一个 + 正向校验 L[i-4] 含 provisional_demand
- R6 实际缩进 16sp（capture 侧孪生为 20sp）；初版 12sp 锚 0 匹配
- R4 24sp 锚命中 capture 侧 `append_private_checkpoint`（私有 builder 尾部实为 28sp）→ `.protection_mask` 误插进 `CapturePlanner::CheckpointPolicy`（无此字段，必编译错）→ repair_r4 修复
- `List.Insert(m+2, x)` 后 `List.Insert(m+1, y)` 会把 y 夹进 x 与后续行之间——连续两行插入须先小索引后大索引
- 校验正则 `\.protection_mask =` 会前缀匹配 `cp.protection_mask ==` 行——用 `TrimStart().StartsWith('.protection_mask =')` 精确计数
- 教训沉淀：锚点「唯一性」须先扫描全文件（含 capture 侧孪生构造器），缩进用 pwsh 实测而非目测

### 9.5 验证与交接
- 本地 docker（ninfer-local-build:latest）：ninja 33/33 + ctest **100% 104/104**（ninfer_resource_manager_test Passed）
- commit e341e68f "fix(P1): scope hard protection to pending demand (UAF + M3 window exit + committed-record exclusion)"；首提漏 engine_core.h（M3 on_waiting_removed 钩子）已 amend 补入；push `83f60ca9..e341e68f → origin/prefix-v2`
- 4090：pull ff OK + ninja 33/33（devel:0905 容器 66e132d9aa6a）
- 用户重跑（需 GPU 空闲；容器内 root 执行）：`export NINFER_QWEN3_8_27B_WEIGHTS=/models/qwen3_8_27b.ninfer && export NINFER_ADMISSION_TRACE=1 && ctest --test-dir /ninfer-4090-kaso/build --output-on-failure -R ninfer_qwen3_6_27b_prefix_real_test 2>&1 | tee /ninfer-4090-kaso/admission_trace.log`
- 判读：通过 = 三层根因（UAF+M3+part3）全部确认，随后决定 trace 门控去留；仍 PRUNED/IDLE-BLOCK = 真容量死锁，下一层 = head 豁免或 P2-A 双槽贪婪 capacity-aware
- 提交链：4f0faf0c→2c26c74f→809537ff→bbb41e3b→a066cb15→3a6b7d06→83f60ca9→e341e68f


## 10. M3 出窗缺陷（4090 死锁二查真因，part 4）

### 10.1 trace 重读（e341e68f 二进制，UAF 修复后）
- UAF 确认修复：protected 从垃圾大数值变为真实小 ID `2 3 4 5 6 7`；`PRUNED (hard protection; evicted owner=2)` 是真保护（非假阳性）
- 9× inspect_admission 全 `source=0 reuse_tokens=0`（head=纯 root，短名单不匹配任何条目）；`candidates=1`（无 source 候选）
- 死锁链不变：head TB + active 集合空 → engine_core.h try_admit_one 抛 "isolated-feasible request is blocked in an idle Engine"

### 10.2 根因（代码级确诊）
- `ensure_base_plan`（engine_core.h，P1 埋点）对每个排队请求 `note_pending_demand(request->id, {key,1}, domain)`：window 记录 `pending_origin=true`，`candidate_keys=[请求自身 prefix key]`
- **正常 admit 路径 `admit_planned_request` 成功分支只做 `erase_pending`，从不调 `on_waiting_removed`/`clear_pending_demand`**（原 :1814-1816；on_waiting_removed 钩子只在排队期 cancel/expire/error 路径调用）
- → 正常完成的请求其 pending 记录永不擦除 → **自保护**（记录 key == 自己 retained 条目的 checkpoint key）→ 引擎 idle 后新 root 请求须驱逐这些 owner → 全部被硬保护剪枝 → TB + idle → 抛错
- 与 trace 精确吻合：protected=2..7 = 6 个已完成请求的 stale 记录自保护（owner 0/1 的条目 checkpoint key 与其 stale 记录不匹配故未保护）
- active 请求保护由 active-edge 排除独立承担（builder 跳过 `private_has_active_edge`/`shared_active_edge_count!=0` 条目，不进 owner 集）→ 本修复不影响 active 保护；单测 test_retained_source_is_protected_until_terminal 靠 active-edge 而非 window，不受影响

### 10.3 修复（.tmp-prefix-plan/patch_m3_admit_clear.ps1，已落盘）
- engine_core.h：`admit_planned_request` 的 erase_pending 成功分支后加 `resources_.clear_pending_demand(request->id);` + 4 行注释（admit=离开排队=出窗）
- tests/test_resource_manager.cpp：新增 `test_pending_demand_window_exit_releases_protection`（note→protection_mask≠0；clear→protection_mask==0，provisional 用默认空 PrefixDemandRecord）+ main 注册
- 生命周期全路径审计：排队期 cancel/expire/error（on_waiting_removed 清 ✓）、admit 成功（新清 ✓）、reserve Stale（留队不动 ✓）、reserve Aborted（on_waiting_removed ✓）、fail_all（引擎停摆，window 随引擎销毁）
- 不做的改动：note_pending_demand 的每轮重注是刻意的刷新机制（head 防 ring 挤出），未改一次性注册（register-once 会让 head 记录被 ring 挤出后失去保护，反而退化）

### 10.4 状态与判读
- 提交链：...→83f60ca9→e341e68f→**1300f493**（fix(P1-M3): release queued hard-protection record on admission，2 文件 +25：engine_core.h、tests/test_resource_manager.cpp；NOTE.md 保持 untracked）
- 本地 docker ninfer-local-build:latest：ninja 33/33 + ctest **100% 104/104**（#11 ninfer_resource_manager_test 含新 test 通过；#25 prefix_real 本地无权重 skip）
- push `e341e68f..1300f493 → origin/prefix-v2`（KasoLu/ninfer-4090）；4090 `git pull --ff-only` OK + `docker run --rm -v C:\Data\ninfer\ninfer-4090-kaso:/ninfer-4090-kaso -w /ninfer-4090-kaso/build ninfer-4090-kaso-devel:0905 ninja` 33/33
- 待用户重跑（devel 容器 root，GPU 须空闲）：`export NINFER_QWEN3_8_27B_WEIGHTS=/models/qwen3_8_27b.ninfer && export NINFER_ADMISSION_TRACE=1 && ctest --test-dir /ninfer-4090-kaso/build --output-on-failure -R ninfer_qwen3_6_27b_prefix_real_test 2>&1 | tee /ninfer-4090-kaso/admission_trace.log`
- 判读：通过 = 四层根因（UAF+M3 出窗时机+part3 committed 排除+admit 出窗）全部确认 → 决定 NINFER_ADMISSION_TRACE 门控（83f60ca9）去留；仍 PRUNED = 真容量死锁 → 下一层 head 豁免或 P2-A 双槽贪婪 capacity-aware（root state_slots=2 以总容量为键、未计 pinned 占用）

## 11. prefix 实跑全量 re-prefill 根因分析（2026-09-09，4090 trace 实跑 + 代码级）

### 11.1 现象（admission_serve_trace.log + log.txt，1300f493 二进制，serve 4 轮对话）
- 引擎无死锁（M3 修复生效：无 IDLE-BLOCK/PRUNED，owners=0，4 请求全部 admit+done）
- 但每轮 prefix_reuse_path=root、prefix_cache_hit_tokens=0、best_reuse_prompt_tokens=0 → 全量 re-prefill
- admission trace 每轮仅 1 行 inspect（root 候选，source=0）→ **reuse 候选从未生成**
- 客户端确实在回传完整历史：prompt 9185→9371→9758→23849（= 上轮 prompt+output+新内容，算术吻合）
- 日志 request 字段：protocol=openai_chat_completions、enable_thinking=true、preserve_thinking=true、has_tool_history=true、media_items=0

### 11.2 引擎侧逐项排除（代码级）
- retained root 正常目录化：finish()（program_impl.h:9293+）publish_continuation 时 endpoint_valid=true（:9346）+ freeze 活性图 → 条目持 endpoint 检查点（populate_continuation_summary :7323-7355，endpoint 仅在 endpoint_valid 时设置）→ rebuild_prefix_index（resource_manager.h:1750-1770）为 endpoint/rewrite/long_anchors 建索引项；空闲 occupancy=1 槽佐证条目存活
- 候选生成（resource_manager.h:336-344）：`base.prefix_shortlist_key(index.key.frontier)` 必须 **==** 存储的 index.key，否则静默 continue → source=0 唯一解释 = **key 失配**（索引非空已证）
- 摘要对称性（prefix_identity.cpp）：`append_digest(token, token_type, positions, rewrite_frontiers, vision)`；纯文本 prompt 路径 assign_text_positions（frontend.cpp:355-372）token_types 全 0、positions=index、rope_delta=0；生成路径 append_generated（:464-483）= (token, 0, {index+rope_delta}) —— **本场景（无媒体）两者输入完全一致** → 失配只能来自 token 序列本身不同
- identity_tag（program_impl.h:7285-7288：speculative_backend|proposal_head<<8|kv_dtype<<16）两端同源
- 结论：引擎复用路径完好；失配 = **传入 prompt 的 token 序列与上轮上下文（prompt+生成 output，含 thinking）在 token 级不完全相同**

### 11.3 根因（客户端回传保真度）
- 多轮 chat-completions 复用要求：新 prompt 的 token 前缀 == 上轮上下文 token（摘要+精确双重校验，prefix_matches 为权威）。生成 token 含 thinking 全文（enable_thinking=true 时每轮产出）；协议虽双向支持 reasoning_content（openai_chat_response.cpp:196-197 输出、openai_chat_request.cpp:414-434 解析回传），但**客户端若未在下一轮 assistant 消息中原样带回 reasoning_content（或回传文本与原始生成 token 有转义/边界差异）→ 摘要在首个 assistant 消息处发散 → 全前缀失配 → 无候选**
- 次级真缺陷（配置级，复用一旦生效就会咬人）：本部署 device 槽总量=2（device 1+shared 1），P2-A（3a6b7d06）新 root 恒需 2 槽（capacity>=2 → state_slots=2）→ 1(retained)+2(root)>2 永不可共存 → 每次 root fallback 必驱逐 retained owner（日志 private_owners_evicted=1/轮）→ 此配置下 retained prefix 结构性无用
- Responses API（/v1/responses + parent_response_id）是引擎设计的多轮复用通道：session_key=response_id → LiveSession+session 索引 → PrivateTurnClosure/rewrite 路径可容忍非逐 token 相同的回传（仅重编发散后缀）；chat completions 无 session 通道（frontend.cpp:1532 update_session_index=false）→ 只能精确匹配

### 11.4 建议（按优先级）
1. 客户端侧：agent loop 每轮保存响应 reasoning_content，下一轮 assistant 消息原样带回（含工具结果等全部历史逐字回传）
2. 决定性 A/B：同客户端同流程加 enable_thinking=false 跑一轮——若 prefix 复用生效=thinking 丢失实锤；若仍 0=模板边界/转义差异，再做 token 级 diff
3. 或改用 Responses API（parent_response_id）走 session 通道
4. 配置缓解（次级缺陷）：device_state_slots 提到 2-3（total>=3）让 retained 与新 root 可共存；P2-A 占用感知（按 pinned 占用降档 state_slots）列为后续引擎改进

## 12. 客户端假设被用户推翻 + v1/v2 全量静态对比 + KEY-MISS 埋点

### 12.1 用户裁决（推翻 §11.3 客户端根因）
- 用户原话："客户端肯定是没问题的，相同的客户端，已经在prefix-v1分支上验证过，可以直接排除客户端侧的原因。问题肯定出在推理端。"
- 同客户端在 prefix-v1 上 prefix 复用正常 → §11.3"客户端回传保真度"根因作废；根因在 v2 推理端（引擎/运行时行为差异）。

### 12.2 v1/v2 全量静态对比（merge-base(prefix-v1, prefix-v2)=4f0faf0c；v1 tip=b806b2f6；diff=13 文件 +626/-541）
- 结论：**key/digest/frontier 全路径代码 v1/v2 结构相同或逐字节相同**，静态找不到能致 key 失配的代码差异：
  - prefix_identity.{h,cpp}：无 diff（摘要函数完全一致）
  - append_generated 调用点：每分支 4 主位 + 2 MTP-spec 位，一一对应、同参 (tokens, sequence.rope_delta)；MTP 投机提交主体（ledger.insert + append_generated(span, rope_delta) + execution_frontier=base_E+committed + mtp_draft_count 更新）**逐字节相同**
  - ledger.push_back 仅 2 处（两分支同）；reuse 回滚 ledger.resize(base)+prefix_digests.truncate(base)：v1 3 处（resident/rewrite/long-anchor-consume），v2 2 处（long-anchor consume 分支被删，与 v2 强制 Retain 对应，非缺陷）
  - execution_frontier 更新点结构相同（prefill =end；ordinary =base_E+committed；settle =pending.base_E+produced / prompt_tokens）；resolve_non_speculative_pending 不变量 prefix_digests.size()==ledger_frontier 两侧相同
  - prefix_shortlist_key（api_impl.h:111-122）逐字节相同：frontier>prefix_digests.size() → nullopt → 候选循环静默 continue（=serve trace 无 source 行的机制）
  - checkpoint_summary/populate_continuation_summary（stored key = sequence.prefix_digests.at(execution_frontier) + identity_tag；tag = speculative_backend | proposal_head<<8 | kv_dtype<<16）逐字节相同
  - resource_manager.h 候选循环：v2 仅删 v1 特有机制（long-anchor conservative_anchor_first retain→consume 回退、release_superseded_source_anchor 取代通道），对 endpoint 候选无影响；rebuild_prefix_index/valid_prefix_index_entry 无 diff
  - request_plan_impl.h diff = P0-c 注释删减 + P2-A state_slots 容量自适应 + inspect_lane long-anchor 强制 Retain + state_exclusive_to_sequence 过滤（均候选可行性侧，与 key 匹配无关）
  - 引擎层其余差异 = P0 stall 链/P1 硬保护/M2 root gating/M3 窗口（不影响 key 生成）
- 推论：差异只能在运行时行为/输入（resident 侧 token 记账 vs 客户端回放的长度或序列、或 retained 条目根本没入索引）→ 上埋点实测裁决
- 保留的次级配置缺陷（§11.3）：本部署 device 槽总量=2，P2-A 新 root 恒需 2 槽 → retained+root 不可共存，每轮 root fallback 驱逐 retained owner（private_owners_evicted=1/轮）——即使 key 修好，此配置下 reuse 候选也须靠 ConsumeToActive 就地接管才可行

### 12.3 KEY-MISS 埋点（resource_manager.h inspect 候选循环，3 处，NINFER_ADMISSION_TRACE=1 门控）
- T1（rebuild_prefix_index 后）：`inspect: index entries=N base_prompt=M | slot=.. shared=.. kind=.. front=.. ord=.. [tag=..]`（全量索引项 dump）
- T2（valid_prefix_index_entry 失败）：`index-invalid slot=.. shared=.. kind=.. front=.. owner=.. rev=.. occupied=..`
- T3（key 门失败）：`key-miss slot=.. shared=.. kind=.. front=.. ord=.. incoming=0/1 [tag=.. in_tag=..]`
- 判读表：entries=0 → retained root 从未入索引（publish_continuation/prompt.identity.reusable/settle 路径问题）；key-miss incoming=0 → 存储 frontier 超出 incoming base digest 表覆盖 → resident 侧 token 记账变长（MTP 提交或 settle 多计）；incoming=1 同 tag 不同 digest → token/位置序列漂移；tag≠in_tag → identity_tag 记账差异
- 编译踩坑：base.summary 是方法 summary() 非成员；测试 FakeShortlistKey 无 identity_tag 字段 → tag 打印用 if constexpr (requires { x.identity_tag; }) 泛型保护（C++20）
- PS1 踩坑：here-string 是单字符串，$t[$k] 取的是字符 → 首跑把 resource_manager.h 拆成逐字符行（已 git checkout 恢复）；正确：$tlines = $t -split "`r?`n" 后逐行 Insert
- 状态：本地 docker build+ctest 验证中 → commit+push → 4090 同步 → 用户 serve 重跑（NINFER_ADMISSION_TRACE=1）贴 trace → 按判读表定位真因

### 12.4 状态更新（KEY-MISS 埋点已发布，等用户 serve trace）
- 本地 docker：ninja 33/33 + ctest 100% 104/104（首轮 compile 失败=base.summary 误用成员+FakeShortlistKey 无 identity_tag，已修：summary() 方法 + if constexpr (requires { x.identity_tag; }) 泛型保护）
- commit **ca208f76** "chore(debug): trace prefix-index key-miss in admission inspection"（1 文件 +64/-2：resource_manager.h；NOTE.md 保持 tracked-but-uncommitted）；push `1300f493..ca208f76 → origin/prefix-v2`
- 4090：`py -3 scripts/remote_4090.py run "cd C:\Data\ninfer\ninfer-4090-kaso && git pull --ff-only" inf` → ff OK；`docker run --rm -v C:\Data\ninfer\ninfer-4090-kaso:/ninfer-4090-kaso -w /ninfer-4090-kaso/build ninfer-4090-kaso-devel:0905 ninja` → 33/33（容器 66e132d9aa6a）
- 用户重跑（devel 容器 root，GPU 须空闲）：`NINFER_ADMISSION_TRACE=1 ./build/apps/ninfer-serve /models/qwen3_8_27b.ninfer --preserve-thinking --chat-template v22_4 --spec mtp --draft-tokens 3 --lm-head-draft --kv-dtype rk8v4 --max-context 200000 --request-log-jsonl logs/log.txt` 跑 2-3 轮对话，贴 logs/ 下 trace 行（grep `admission-trace`）
- 判读：§12.3 判读表（entries=0 / index-invalid / key-miss incoming=0|1 / tag 对比）→ 定位后修复+本地 build+ctest+4090 同步
- 提交链：…→3a6b7d06→83f60ca9→e341e68f→1300f493→**ca208f76**
## 13. LCP 分歧扫描埋点（commit 0766142a）+ 4090 同步 + 交接

### 13.1 本轮动作
- 补丁 .tmp-prefix-plan/patch_divergence_trace.ps1（A runtime.h 声明 / B api_impl.h 委托 / C program.h core 声明 / D program_impl.h 实现 @valid_capture_offer 前 / E resource_manager.h key-miss 分支调用）+ patch_divergence_E.ps1（E 单独重做）
- D 实现 = ProgramImplCore::debug_trace_prefix_divergence(stored, base) const：env 门控 NINFER_ADMISSION_TRACE；打印 stored_size/incoming_size/stored_frontier/ledger_frontier/ledger_tokens/rope_delta/in_tag/endpoint_valid/anchors + 每锚点 front/ord；线性扫 1..min(overlap) 首处分歧；first==0 打 "chains identical through N"；否则打 first + 两侧 digest 词 + stored 侧 [first-4..first] 每 token tok/tt/p0/p1/p2（tt=prefix_identity.token_types，p=position_axis 0/1/2）
- E 落点：resource_manager.h key-miss 分支（`if (!incoming || *incoming != index.key)`）内 env-gate 闭括号后、continue 前（403 行后），`if (!index.shared) { key_probe=catalog_[slot]; if (key_probe.handle) program.debug_trace_prefix_divergence(*key_probe.handle, base); }`
- 编译炸点：tests/test_resource_manager.cpp FakeProgram 无此方法 → 加 no-op stub（`void debug_trace_prefix_divergence(const FakeContinuationHandle&, const FakeRequestBasePlan&) const {}`，isolated_request_feasible 之后）
- 本地 docker：ninja 32/32 exit 0 + ctest 100% 104/104
- commit 0766142a "chore(debug): prefix digest divergence trace on key-miss"（6 文件 +112/-1）；push ca208f76..0766142a → origin/prefix-v2
- 4090：git pull --ff-only 报 "Cannot fast-forward to multiple branches"（upstream 歧义，仓库状态实际 behind 1 干净）→ 改 `git merge --ff-only origin/prefix-v2` 成功；devel 容器 ninja 43/43 exit 0

### 13.2 判读表（读 divergence 行）
- first ≥ stored prompt 长度（≈2.6万 类值）且 st@ 行 tok 合理 → 客户端回显 re-tokenize 漂移 → 但用户已排除客户端（v1 同客户端复用正常）→ 该形态实际指向 v2 缺锚点（状态槽预算 device1+shared1=2，root 占满 → 锚点 capture 结构性不可行，automatic_private_anchors 恒 0）使 endpoint-only 匹配过严；v1 靠锚点兜底
- first 处 st@ 行 tok 异常（0/草稿/巨大值）→ resident 侧 token 记账 bug（MTP 提交 span 错位类）
- "chains identical through N" → stored 链与 incoming 完全同前缀但 stored frontier 超出 incoming 表（客户端回显截短/驻留侧多记账）
- anchors=N（本配置预期 0）
- 若 divergence 行根本没出现 → key-miss 分支未走到该条目（索引/valid_prefix_index_entry 门）→ 回查 T1/T2 行

### 13.3 交接用户（GPU 须先停常驻推理容器）
- serve：`build/apps/ninfer-serve /models/qwen3_8_27b.ninfer --preserve-thinking --chat-template v22_4 --spec mtp --draft-tokens 3 --lm-head-draft --kv-dtype rk8v4 --max-context 200000 --request-log-jsonl logs/log.txt`，env `NINFER_ADMISSION_TRACE=1`，跑 2-3 轮对话
- 贴回：stderr（含 [admission-trace] divergence: 行）+ logs/log.txt

### 13.4 踩坑
- 主 PS1 E 步误断言 continue 前两个 `}`（实际 key-miss 结构 = env-gate 单 `}` + continue + key-miss `}`）→ hits 空 → 拆出 patch_divergence_E.ps1 用"anchor 后向前扫 `}`+continue+`}` 三连"定位
- remote_4090 `git pull --ff-only` 歧义 → 一律 `git merge --ff-only origin/prefix-v2`

### 13.5 提交链
- …→1300f493→ca208f76(key-miss trace)→**0766142a**(divergence trace) = 本地 HEAD = 4090 二进制


## 14. 会话污染 → HANDOFF.md 转写（2026-09-09 末）

- 本轮会话因推理产物中的模板控制 token（数字 ID 198/248068/248069/271 簇）多次异常停止，用户裁决转写交接：当前完整状态（环境/提交链/分歧数据表/根因形态/修复候选/代码锚/下一步/踩坑）已落盘 **HANDOFF.md**（覆盖 P0 时代旧版），为该时点唯一权威交接。
- 核心结论快照：分歧非 v2 记账 bug（token/位置/token_type 口径两侧一致、v1/v2 digest 代码无 diff），而是存储链含原始生成段控制 token、客户端回放为解析后文本重序列化 → 首控制 token 处必分叉；本部署锚点通道同时死亡（total 状态槽=2 池满，captures/anchors 恒 0）。
- 首选修复 = endpoint 存储 key frontier 前移至最后 prompt 边界（上轮 prompt_tokens，digest 表天然覆盖）；备选 = P2-A 第二槽容量感知恢复锚点通道。下轮会话从 HANDOFF.md §6 继续。
- 会话卫生：回复/文档不复现聊天模板控制序列，token 一律用数字 ID。


## 15. 根因定案 + 修复设计：Prompt 边界 Endpoint（2026-09-09 末，接手 HANDOFF.md §6）

### 15.1 根因定案（0766142a divergence trace 解读）
- 分歧点 = 生成段内首个模板控制 token 簇（st@ token ID 13/198/248069/271 簇）；round2 req：stored 首分歧 58743 / stored 全长 58822，incoming 60328
- 存储链 = 原始生成 token 流（含生成段控制 token）；客户端回放 = 解析后文本重序列化 → 在生成段首个控制 token 处必分叉；两侧记账均无错（token/位置/token_type 口径一致，v1/v2 digest 代码字节相同）
- 用户裁决（原文）："客户端肯定是没问题的，相同的客户端，已经在prefix-v1分支上验证过，可以直接排除客户端侧的原因。问题肯定出在推理端。" → 修复 = 让 resident 侧存储 key 落到 prompt 边界（两侧 token 相同段）
- anchors=0 通道第二根因（本会话新确认）：inspect_capture（program_impl.h:7526-7760）把 capture 目的地按"新池槽"计费（:7671-7678 `device_destination_available = recycles_private_state || (device_occupied() - replaced_shared) < device_capacity()`；DeviceFork → added.device.state_slots=1），从不计入 sequence 自身 P2-A 预预留槽 → root 占满 2 池槽期间，带标志 capture → HostSnapshot（host 池=0）→ physically_feasible=false → reserve_active_capture skip_capture。执行侧已支持预留槽（prepare_active_capture :8242-8245 `else if (sequence.reserved_state) { destination = *reserved_state; reset(); }`），判定侧未计 ⇒ 本部署锚点通道结构性死亡，不修（最小改动原则），主修 endpoint 通道
- 关键事实：PLAIN capture group（无 rewrite/long_anchor/shared 标志）走 inspect_capture :7564-7569 早退 → 恒 physically_feasible=true，不计费不判 placement → plain 组无需改 inspect_capture

### 15.2 修复设计："Prompt 边界 Endpoint"
- 概念：endpoint 存储 key+image 前移至上一轮 prompt 边界（prev prompt_tokens）。prompt 段两侧 token 恒同 → incoming prefix_shortlist_key(prompt_boundary) 命中。prompt 边界状态 image = prefill 末（group.frontier == prompt_tokens）的普通 capture → root 的 P2-A 预预留槽（2 槽池：root 自持 active+reserved=2，plain capture 用预留槽不占新池）；finish 的 fork-abort 使 state.read = 冻结的 prompt 边界 image（生成尾 image 被释放），tail_hidden view → prompt 边界 hidden 槽（恰为 BeforeSuffix MTP 桥输入，program_impl.h:11672-11697）
- 下一轮：ConsumeToActive 将该 image 移入新 lane（retained 1 + 新 root 1 = 2/2 无驱逐），只 re-prefill base..prompt_tokens；其自身 prefill 末 capture 延续链。复用损失 = 生成段 + 新消息（≈110-1800 / 58k+ tokens）
- 刻意放弃：tail-frontier 双索引（尾 image 在 finish 被释放；客户端恒在生成段内分歧）；inspect_capture 预留槽修复（§4.4-2 锚点通道，仅 auto-anchors 必须生效时才需要 —— 缓办）；root 槽容量感知（2 槽池靠 ConsumeToActive 即可运转）；DFlash 支持（dflash_context_frontier==base 精确检查堵死 early-base 复用 → DFlash 保持 legacy endpoint）
- 组选择（要点）：2 槽引擎上带标志 capture（auto-anchors）判 infeasible 被 skip，从不占预留槽、从不 throw → 无需移除其他组；若 auto-anchor 恰落 prompt_tokens 边界与本组合并 → 清除其 rewrite/long_anchor 标志转 plain（否则带标志组被判 infeasible 跳过，endpoint 失效）

### 15.3 变更集（5 处，无新 Program API → test fakes 不动）
- D1 program.h:429（SequenceState，`endpoint_valid` 后）加 `std::uint32_t endpoint_frontier = 0;`（endpoint image 所在 frontier；0 = legacy execution_frontier）；program_impl.h 七个 `endpoint_valid = false` 点同步清零：2857 / 4689 / 6735 / 10111 / 10752 / 10822 / 10887（finish 的 :9429 `= true` 不动）
- D2 request_plan_impl.h plan_request（shared_candidates sort 结束 :375 后、capture_backing :376 前，已处于 publish_continuation + allow_prefix_reuse + reusable + context_cache.enabled 门内）：`state_store != nullptr && state_store->device_capacity() == 2U && speculative_backend != SpeculativeBackend::DFlash` → 在 base->capture_groups 里 find-or-create frontier=prompt_tokens 的 plain 组（append 尾部保持 (frontier,input_order) 排序，prompt_tokens=最大 frontier）；若与既有带标志组合并 → `rewrite.reset(); long_anchor=false;`（shared 标志在另一向量 shared_candidates，无交叉）
- D3 program_impl.h publish_active_capture（`++prefill.next_capture;` 后，~:8566）：`if (transaction.group.frontier == prefill.prompt_tokens) { sequence.endpoint_frontier = prefill.prompt_tokens; }`（plain 组不触发 populate_continuation_summary；finish() :9435 直调 populate_continuation_summary 读到冻结 image + 新 frontier ✓）
- D4 program_impl.h populate_continuation_summary（:7410-7417 段）：endpoint frontier = `(endpoint_frontier != 0 && endpoint_frontier < execution_frontier) ? endpoint_frontier : execution_frontier`；endpoint_work = 前式 ? `runtime::make_prefill_work(0, frontier, sequence.rebuild_work.vision_items, sequence.rebuild_work.vision_patches, prefill_chunk)` : `sequence.rebuild_work`（validated_rebuild_work 仅要求 work.tokens == frontier；vision 量全在 prompt 内 ✓；make_prefill_work 签名 (prefix_tokens, suffix_tokens, items, patches, chunk)，src/runtime/contract/types.h:229）；image 恒 = sequence.state.read
- D5 request_plan_impl.h :497-503（inspect_lane SessionEndpoint 分支）：throw 条件改 `selected.frontier == 0 || (selected.frontier != source->execution_frontier && (source->endpoint_frontier == 0 || selected.frontier != source->endpoint_frontier))`
- 无需改：MTP 门（:546-560，mtp_kv_valid=execution_frontier ≥ base-1、tail_hidden_valid 恒 true、view 指 hidden@base-1 正确）；selected_state（PrivateEndpoint → state.read 不变）；materialization PrivateEndpoint（:10016-10109，text_kv_valid/mtp_kv_valid 均 ≥ base）；selected_state_requires_fork（本部署无 alias 引用 → ConsumeToActive 免 fork）；inspect_capture（plain 组走早退）
- 预期：prompt 边界条目 key-miss 消失；prefix_cache_hit_tokens ≈ 上轮 prompt_tokens；re-prefill = 生成段 + 新内容；captures≥1/轮；无 private_owners_evicted
- 降级路径：单槽引擎（capacity≠2 → 组不加 → legacy endpoint）；DFlash → legacy；运行时 capture offer 被跳过（program transaction 占用）→ endpoint_frontier 恒 0 → execution_frontier legacy，不 crash
- 容量 ==2U（非 >=2U）：3+ 槽引擎上 plain 尾组需与带标志组争免费池槽，reserve_destination 竞争可能 throw（未验证的失败模式）→ 严格限定 ==2U 匹配本部署；≥3 槽推广留 follow-up
- 风险：test_engine_prefix_real battery 中 device_capacity≥2 的引擎将新增该 plain 组，reuse 量断言可能偏移；本部署组合（同会话多轮 + 2 槽池 + MTP + preserve_thinking）是 battery 盲区 → 验证后补场景

### 15.4 执行记录
- 2026-09-10 实施：edit 工具本会话失效（E_BAD_SHAPE）→ 全部改用 PS1：.tmp-prefix-plan/patch_prompt_boundary_endpoint.ps1（锚点计数断言、自底向上行操作）+ fix_alignment.ps1；program.h 首次失败后曾需 git checkout 还原再重跑
- git diff 复核（HEAD 0766142a 之上未提交）：program.h +1（endpoint_frontier 字段 @428）；program_impl.h D1b×7（2858/4691/6738/10129/10771/10842/10908）+ D4（:7414-7429 endpoint_frontier 三元 + make_prefill_work(0, frontier, vision_items, vision_patches, prefill_chunk) 分支）+ D3（:8579-8581 publish 后 `if (transaction.group.frontier == prefill.prompt_tokens) sequence.endpoint_frontier = prefill.prompt_tokens;`）；request_plan_impl.h D2（:374-395，gate `state_store && device_capacity() == 2U && backend != DFlash`，find-or-create frontier=prompt_tokens 的 plain 组并清 rewrite/long_anchor）+ D5（:520-527 throw 条件接受 source->endpoint_frontier）
- 本地 docker build+ctest（job pwsh-22）：ninja 32/32、ctest 104/104 全绿（GPU 用例本地跳过）
- 池公式定案（layouts_impl.h:119-121）：StateImage Device 槽数 = max_concurrency + device_state_slots → 部署=2（1+1，shared 另计使总 device 池=2）、pressure_resume=4（2+2）、private_checkpoint_pressure=4、host_restore=2（1+1，host=2）、shared_replacement=2（host=4）、rewrite-branch=3、base=5、concurrent=24
- 终端 plain 组目的地安全性证明（==2U gate 下 throw 不可达）：池=2 且 host=0 → 内部带标志 capture 判 infeasible（occupied==capacity、DeviceFork peak+1 无 host 可落）→ 引擎 skip_capture，预留槽完好；池=2 且 host>0 → 内部 capture → HostSnapshot（仅 logical+host 槽，不占 device 槽）→ 预留槽完好；池≥3 引擎被 ==2U gate 排除。prepare_active_capture 目的地序：recycles → sequence.reserved_state（P2-B :8256-8259）→ reserve_destination()（池≥3 且有预留槽被内部 capture 占用时 free=C-2≥1）
- 部署 captures=0 与 pressure_resume 测试通过的矛盾由此消解：两池均=2 时带标志 capture 一律 skip（部署 MTP ResponseReplay 同此），pressure_resume 池=4 时 rewrite 经预留槽捕获成功（reused_pages=120 = generation_begin 7680 边界）
- 降级路径：运行时 offer 被跳过（program transaction 占用）→ endpoint_frontier 恒 0 → D4 三元回落 execution_frontier（legacy endpoint），不 crash
- battery 影响面（==2U gate）：仅 host_restore / shared_replacement 两引擎新增 plain 组（池=2、host>0 → 内部 HostSnapshot 路径安全）；endpoint 前移至 prompt 边界可能偏移 reuse 量断言 → 4090 ctest 验证
- 后续：commit+push → 4090 merge --ff-only + ninja → 用户停常驻 GPU 容器后：4090 ctest battery + serve 重跑（NINFER_ADMISSION_TRACE=1，2-3 轮；判据：prompt 边界 key-miss 消失、prefix_cache_hit_tokens ≈ 上轮 prompt_tokens、captures≥1、无 private_owners_evicted）；仍 miss 则上 §4.4-2（inspect_capture 预留槽计费）
