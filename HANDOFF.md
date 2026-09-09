# HANDOFF.md — PREFIX-PLAN v2 实施交接（ninfer-4090-kaso / prefix-v2）

> 写于本会话切换前。目标会话从本文 + `PREFIX-PLAN.md` v2 + `NOTE.md` 恢复。
> 仓库：`C:\Workspace\codes\ninfer-4090-kaso`，分支 prefix-v2（HEAD 4f0faf0c，前两条 5a09e419 / d24fdb90）。
> **工作树已含 P0 全部未提交改动（11 文件，未编译）**。

## 1. 任务与三目标（用户原话，任务依据）

- 原始任务："对当前分支中，涉及到kvcache和prefix以及prefill相关的逻辑，进行一个全面完善的梳理，将结果落盘至KVCACHE.md中"（已完成：KVCACHE.md 984行）。
- 派生："结合KVCACHE.md分析我实际使用中的三个问题，落盘PREFIX-ISSUS.md"（已完成 v2）。
- 设计三目标（PREFIX-PLAN.md §0）："1.max_tokens只是最大值的约束，不做强制性的page占用，动态申请，申请失败=上下文窗口到极限=正常逻辑 2.渲染结果与prefix匹配时无论如何不得丢prefix（除非一开始就不匹配才全量prefill） 3.所有page按需分配，prefix稳定必复用，禁止prefix匹配时触发全量prefill"。
- 实施任务："现在，按照PREFIX-PLAN.md，开始执行"；可行性审查已完成并合并（"将本次分析，合并至PREFIX-PLAN.md文档中，对列出的问题给予修正"）。
- 持续约束：**改动最小、不引入新 bug**；prefix-v1 分支（= prefix-v2 + 15 commits，tip b806b2f6）**仅作方向参考，不能照搬**（§9 对照表：fail_all 明确放弃）；"记得实时更新TODO和GOAL"。
- 用户三令：**禁止使用 edit 工具**（本会话连续退化），文件修改一律 `write` 落 PS1 到 `.tmp-prefix-plan\` + `pwsh -NoProfile -File` 执行。

## 2. 磁盘权威文档（全部在仓库根）

| 文件 | 状态 |
|---|---|
| `KVCACHE.md` | 分支全景 §0-§13（984 行），已完成 |
| `PREFIX-ISSUS.md` | 三问题归因 v2（250 行），已完成 |
| `PREFIX-PLAN.md` v2 | **执行规格**（376 行/56948B，§0-§11；已并入 7 处小误修正 + M1-M7 设计修正） |
| `PREFIX-PLAN-REVIEW.md` | 可行性审查报告（181 行/34352B；判定=条件通过，M1-M7 + F-2） |
| `NOTE.md` | 本会话持久工作日志（28829B：任务/定案/代码事实/进度/环境备忘） |
| `.tmp-prefix-plan\` | apply_p0a/p0b/p0c/p0d/p0g.ps1 + patch_p0d/patch_f6b/patch_f7c.ps1（**已执行完毕的锚点脚本，不可重跑**——锚点已消耗，Subst 会 count=0） |

## 3. P0 改动清单（全部已落盘，全部未编译）

### 3.1 支柱一：动态 KV 页（已实现）

1. **`src/targets/qwen3_6/impl/runtime/logical_kv_store.h`**（93252B/1908 行，CRLF）
   - detail 命名空间加 `enum class KvGrowth : std::uint8_t { Ok, PoolExhausted, InvalidTarget };`（:27 区）。
   - `materialize_to_tokens(handle, tokens, stream)` 由 `void` 改 `[[nodiscard]] KvGrowth`：
     - `target < page_count` → `throw std::invalid_argument("KV materialization exceeds active entitlement")`（I1 单调性，原串保留）；
     - `target > page_capacity_` → `KvGrowth::InvalidTarget`（**基线无此分支**——新代码移除 entitlement 守卫后必须显式补，否则 memberships span 越界写穿）；
     - `target == page_count` → Ok；
     - `needed = target - page_count > reservation.pages()` 时：`can_resize_reservation(reservation, needed)==false` → `PoolExhausted`（无异常），否则 `resize_reservation` 原子批后继续物化。
     - publish try/catch 回滚 dematerialize 结构未动，物化异常仍上抛（E2：其他路径池耗尽=bug→fail_all 暴露）。
   - 新增 `[[nodiscard]] bool can_materialize_to_tokens(handle, tokens) const noexcept`（valid/active/row/reservation.valid → target 边界 → needed<=headroom → can_resize_reservation）。
   - 语义依据：`entitlement(address) = page_count + reservation.pages()`（:1822 区）；页池 `can_resize_reservation(reservation,new) const noexcept`（`src/core/paged_kv_cache.h:226-228`/`.cpp`，used=allocated+(reserved-reservation.pages)+new≤capacity）= 无异常预检；`resize_reservation` 失败 throw bad_alloc。
2. **调用方 `!= KvGrowth::Ok` → throw**（5 处，错误串即调试索引）：
   - `program_impl.h` `materialize_sequence_kv`（:10755 区）：text → `logic_error("text KV growth failed during sequence materialization")`；backend → `("backend KV growth failed during sequence materialization")`。decode/prefill 路径的 PoolExhausted 结构性不该发生（decode 有轮首 probe+stall；prefill 只从激活期已预留页物化，M4），发生=bug。
   - `program_impl.h` causal score 行 → `"causal score KV materialization failed"`；capture 行 → `"capture KV materialization failed"`。
   - `session_snapshot_impl.h` → `"session snapshot KV materialization failed"`。
   - `tests/targets/qwen3_6/test_context_store.cpp` 原 12 处调用全改 `expect(... == store::KvGrowth::Ok, "KV materialization returns Ok");`（12 场景全 target≤entitlement，行为不变；构建无 -Werror）。
3. **`request_plan_impl.h`**（75275B/1306 行）：删 `reserved_context_tokens` 局部；`text_kv_page_entitlement = pages_for_tokens(base->summary.prompt_tokens)`；MTP backend = `pages_for_tokens(min(capacity, prompt_tokens + draft_window - 1))`；DFlash backend = `pages_for_tokens(prompt_tokens)`。`capacity_output/effective_output_tokens/effective_limit_reason` 保持基线语义不动（M6：`requested<=capacity_output ? OutputLimit : ContextCapacity`）。`root_active.state_slots=1U` 不动（P2 机制 A 才改 2）。
4. **新 API（7 文件）**：
   - `src/targets/qwen3_6/export/ninfer/targets/qwen3_6/runtime.h`：`class Program` 前加 `enum class SequenceGrowth : std::uint8_t { Ok, PoolExhausted, InvalidTarget };`（与 detail::KvGrowth 不同名防遮蔽）；public 声明 `[[nodiscard]] SequenceGrowth probe_decode_capacity(SequenceHandle<Variant> sequence, std::uint32_t remaining) const;` + `void mark_capacity_stalled(SequenceHandle<Variant> sequence);`（`has_context_transaction()` 声明后）。
   - `api_impl.h`：两个 `template <>` 委托透传 `impl_->`。
   - `program.h`：core 声明（裸 `SequenceHandle`）。
   - `program_impl.h` 实现（`ProgramImplCore::finish` 前插入）：
     - `probe_decode_capacity`（const 纯查询，不 throw）：`has_context_transaction()||pending_transaction_||!valid_sequence` → InvalidTarget；`lane = ContractAccess::lane(sequence).value`；`state = active_sequence(lane)`；`!state.kv` / `frontier>=capacity` → InvalidTarget。目标公式（镜像 decode 增长，冻结）：ordinary `text=frontier+1`；MTP `extent=min{state.mtp_draft_count, draft_window, remaining>1?remaining-1:0, capacity-frontier-1U}`、`text=frontier+extent+1`、`backend=min(capacity, frontier+extent+draft_window)`；DFlash `extent=min{draft_window, max_by_budget, capacity-frontier-1U}`、`text=frontier+extent+1`、`backend=frontier`（不增长）。`can_materialize_to_tokens` false → PoolExhausted，否则 Ok。
     - `mark_capacity_stalled`：前置 `!has_context_transaction()&&!pending_transaction_`（否则 `logic_error("capacity stall overlaps an open context transaction")`）；`!valid_sequence`/`!state.kv` throw；`lifecycle != Lifecycle::Active` → `logic_error("capacity stall requires an active decode lane")`。迁移（镜像 terminal commit `program_impl.h:12231` 基线）：`request.lifecycle = Lifecycle::Finishable; request.pending = {}; state.mtp_draft_count = 0;`（probe 在轮首 progress 之后，无需 settle_state_fork/trim）。
   - `src/runtime/engine/request_record.h`：`terminal_reason` 后加 `bool capacity_stalled = false;`。
   - `src/runtime/engine/resource_manager.h`：`finish(program, lane, sequence)` 签名加尾参 `bool allow_abort_fallback = true`；`result.status != ConsumeStatus::Consumed` 分支内、`program.abort` fallback **之前**：`if (!allow_abort_fallback) { throw std::logic_error("capacity-stalled terminal settlement did not consume the sequence"); }`（**M5**：stall 请求禁 abort fallback——abort 会 `clear_catalog_entry` 丢该请求自己的 endpoint 目录条目→下轮 prefix 匹配断裂→违反目标2）。
   - `src/runtime/engine/engine_core.h`：
     - settle：`resources_.finish(*instance_.program, *request->lane, *request->sequence, request->capacity_stalled);`（第 4 参）。
     - **stall 循环**（worker_loop，`cancel_active_requests(cancelled_at_boundary, boundary);` 后、首次 `build_round_membership` 前——M1：必须早于 membership 构建，stall lane 被清预算后结构性排除，零预算 lane 进 decode membership 必 throw `"ordinary batch row is not decode-ready"`）：
       ```cpp
       // 对 lane 0..max_concurrency_：
       //   stalled = slots_[lane]；null / !is_decode_ready() / capture_pending / !sequence / !lane / lane->value!=lane → continue
       //   remaining = stalled->budget ? stalled->budget->remaining() : 0;
       //   if (instance_.program->probe_decode_capacity(*stalled->sequence, remaining)
       //       != targets::qwen3_6::SequenceGrowth::PoolExhausted) continue;
       //   if (stalled->budget) { stalled->budget->commit(remaining); }            // 清零（commit(remaining) 安全）
       //   stalled->output.preview_terminal(FinishReason::ContextCapacity);        // 引擎侧唯一 reason 来源
       //   instance_.program->mark_capacity_stalled(*stalled->sequence);
       //   resources_.mark_terminal_pending(LaneId{static_cast<std::int32_t>(lane)}); // note_capacity_stall 复用
       //   stalled->terminal_reason = FinishReason::ContextCapacity;
       //   stalled->capacity_stalled = true;
       //   stalled->model_state = EngineRequestState::ModelFinished;
       ```
       下一轮 `settle_terminal_requests`（:1109 区）自然交付：ownership 校验（`is_model_finished && !capture_pending && lane 匹配 && TerminalPending`，违反 → `logic_error("terminal-pending request has invalid ownership")`→fail_all，**这正是 stall 循环必须检查 `!capture_pending` 的原因**）→ `resources_.finish`（strict）→ program.finish 应 Consumed（lifecycle 已置 Finishable；`!publish_continuation`→Released 合法；Catalogued→记 retained_slot/retained_session_digest）→ `complete_success`（对任意 reason 通用，协议映射已齐备：`FinishReason::ContextCapacity` 在 `include/ninfer/types.h:555`；OpenAI→"length"（`src/serve/openai_chat_response.cpp:46`）、Anthropic→"model_context_window_exceeded"（`anthropic_messages_response.cpp:64`））→ `remove_completed_slot` 内部触发 `request_admission_check` 排队补位。
     - **§3.5**：worker_loop decode 分支 `run_decode_round(membership, cancelled_at_unit_start);` 后（20 空格缩进）`request_admission_check(); // PREFIX-PLAN P0 (S3): re-arm admission after a decode round (30s blind spot)`（消除 TemporarilyBlocked 者最长等 pending_timeout 的盲区；既有 gate `should_attempt_admission` 含 `previous_unit_was_decode` 已能放行）。
5. **E4 统计**：`program_impl.h` add_kv lambda `device_pages += entitlement - mapped;` → `device_pages += mapped;`（guard `entitlement - mapped >` → `mapped >`；`entitlement < mapped` 不一致 guard 保留，串 `"resident active KV entitlement is inconsistent"`）。
6. **P0-g 单测**：`tests/targets/qwen3_6/test_context_store.cpp`（41376B/681 行）新增 `test_kv_growth(ninfer::DeviceContext& device)`（独立 fixture 同 `test_kv_store` 模式）：`grow=create_active(2,0)`→`materialize_to_tokens(*grow,65)==Ok`→`mapped==2`；`materialize_to_tokens(*grow,32)` 期望 `std::invalid_argument`（I1）；filler 循环 `create_active(1,0)`（catch bad_alloc）耗尽 `available_pages()`；`materialize_to_tokens(*grow,130)==store::KvGrowth::PoolExhausted`（**无异常**）；`!can_materialize_to_tokens(*grow,130)`、`can_materialize_to_tokens(*grow,65)==true`；`mapped` 仍 `==2`。main 内 `test_kv_store(device);` 后注册 `test_kv_growth(device);`。

### 3.2 P0 未做（有意）

- E3 prefill stall 通道不做（新 entitlement 后 prefill 只从激活期已预留页物化，枯竭结构性自消，M4）；prefill 增长失败保留防御 throw。
- M7（root state_slots=2 槽冲突=请求级拒绝）、P1 支柱二、P2 支柱三 均未动。

## 4. 下一步（目标会话从这里继续）

1. **P0-h 编译验证**（用户上次说"暂停一下"，未启动）：
   ```
   docker run --rm -v C:\Workspace\codes\ninfer-4090-kaso:/src -w /src/build ninfer-local-build:latest ninja
   docker run --rm -v C:\Workspace\codes\ninfer-4090-kaso:/src -w /src/build ninfer-local-build:latest ctest --output-on-failure
   ```
   （AGENTS.md:482 约定：本机无 NVIDIA 显卡，nvcc/ctest 一律在 `ninfer-local-build:latest`（nvidia/cuda 13.1.2 基座，`/usr/local/cuda/bin` 在 PATH，**无默认工作目录**，必须显式 `-w`）；`/src/build/build.ninja` 已存在可复用 Ninja。）
   - 编译失败大概率点：core 里 `SequenceGrowth`（公共枚举，core 命名空间嵌套在 qwen3_6 下，未限定可见）/ `ContractAccess::lane` 用法 / stall 循环里 `targets::qwen3_6::SequenceGrowth` 解析（ninfer::runtime 内 unqualified 经 ninfer 外层可解析）/ `budget->remaining()` 存在性（GenerationBudget 有 remaining()，`commit(>remaining)` 会 `std::abort()`——`budget->commit(remaining)` 清零安全）。
   - 通过后：GPU 4090 容器 `ninfer-4090-kaso-dev`（bind `/ninfer-4090-kaso`，端口 1234）验证 **前必须先请用户关闭其常驻推理容器**。验证项（PREFIX-PLAN.md §6 P0）：① serve + `max_tokens=100000`（超池）→ 请求以 `finish_reason="length"`/`"model_context_window_exceeded"` 正常终结、引擎不停摆；② A 占大池 → B stall 终结 → 引擎持续；③ OpenAI/Anthropic 双协议 finish_reason 一致性。
2. **P1 支柱二**（PREFIX-PLAN.md §3，全部未动）：压力目标空间禁 Evicted 保护 owner 集 P（host 开时允许 D2H 降级，落点 `resource_manager.h build_pressure_inputs :1967-2100` + `materialization_planner.h` owner policy）；**M2 root gating 谓词细化**（仅匹配候选 feasible 或"可被进展变 feasible"时禁选 root；结构性死亡=前缀物理已丢 → 放行 root，用 planner diagnostics `observe_planner_diagnostics resource_manager.h:2482-2497` 的 budget_exhausted/stop_reason 可辨；快速路径 seal_identity 同样加守卫——root 可在快速路径胜出）；**M3 note_pending_demand**（复用 kDemandWindowCapacity=32 窗口 `resource_manager.h:1423`/`commit_demand :2457-2480`；`PrefixDemandRecord :122-126` 无 owner 字段→增 owner 或文档化 lazy expiry；**必须纯窗口插入，不得触发 explicit_credit 清除侧效应**（只看 back() 记录会误清其他请求的匹配 credit）；运维约束 pending+active≤32，serve 默认 max_pending_requests=16、pending_timeout_ms=600000（`src/serve/serve_options.h:33-34`，kMaximumConcurrency=8））。
3. **P2 支柱三**（§4，全部未动）：机制 A root `state_slots=2`（`request_plan_impl.h root_active :284-289 区` 改 1U→2U + placement 加 reserved_state 目的分支，目的序 prepared→recycled→reserved_state→new；M7：`reserve_state_entitlement program_impl.h:9823/9871` 失败=请求级拒绝，不得 fail_all）；机制 B 轮转锚点图像回收为 fork 目的（`install_private_capture program_impl.h:7974`/`prepare_active_capture :8041`）；机制 C `drop_superseded_anchor` noexcept 七项校验（含 `checkpoint_references(anchor.state)==1`，`state_image_store.h:209-211` 原语齐备）+ skip 分支（`program_impl.h:7822-7826 if (!pressure && !assessment.physically_feasible) → skip`）先释放再重评估一次。
4. 每完成一步：更新 NOTE.md 进度 + 实时更新 TODO/GOAL + Docker 编译/ctest 回归。

## 5. 环境硬约束

- 本机（Windows）无 GPU；**DSH file policy=danger-full-access；approval prompts 已禁用——不得设置 sandbox_permissions，拒绝即终局**。
- 文件修改：**edit 工具禁用**（用户三令）。范式=`write` 落 PS1 到 `.tmp-prefix-plan\` + `pwsh -NoProfile -File` 执行。PS1 教训（血泪）：CRLF 文件行正则**不用 `$` 尾锚**；`[regex]::Matches/Match` 须显式 `[regexoptions]::Multiline` 否则 `^` 只匹配串首；生成含引号/反引号续行的脚本片段用 `@'...'@` here-string（双引号拼接会被转义打断 ParserError；单引号字面量内嵌单引号须双写 `''`）；锚不唯一时加邻行组成双行锚；所有替换先全量校验 count==1 再统一写盘（fail-fast 不写=无损）。
- 4090 GPU 容器跑测试前**必须先请用户关闭其常驻推理容器**。
- 错误串索引（调试用，全部为新增）：`"capacity stall overlaps an open context transaction"` / `"capacity stall requires an active decode lane"` / `"capacity-stalled terminal settlement did not consume the sequence"` / `"text KV growth failed during sequence materialization"` / `"backend KV growth failed during sequence materialization"` / `"causal score KV materialization failed"` / `"capture KV materialization failed"` / `"session snapshot KV materialization failed"`；基线既有：`"KV materialization exceeds active entitlement"`（I1）/ `"ordinary batch row is not decode-ready"` / `"terminal-pending request has invalid ownership"` / `"Program could neither retain nor discard terminal sequence"`。
- 关键不变量：I1 物化单调；I3 唯一闸门（"池不足"仅 `materialize_to_tokens` 一源，其他路径池耗尽=bug→fail_all 暴露）；stall 请求必须走 finish 而非 cancel（cancel 的 `resources_.abort` 丢 continuation 不 catalog）；worker_loop `catch → fail_all_locked`（`engine_core.h:2073-2105`）= 引擎**永久**停摆——任何未受控 throw 都会触发，stall 链所有前置检查（!capture_pending、lifecycle==Active、事务无重叠）都是防这条命的。

## 6. 目标/TODO 状态

- GOAL：goal-757fe8a1-3e21-4a39-95d4-561bad506f05（按 PREFIX-PLAN.md v2 实施三支柱；active，rounds ~9/40，目标会话用 update_goal resume 重新武装）。
- TODO（全量替换制）：P0-a/b/c/d/f/g ✅；P0-h（Docker 编译+ctest）⬜ in_progress；P1（支柱二+GPU 验证）⬜；P2（支柱三+GPU 验证）⬜。
