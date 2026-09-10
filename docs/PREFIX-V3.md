# PREFIX-V3.md — 前缀复用最小改动重做方案（设计规划，不含代码改动）

> 定位：本文是**重做（redo）设计规划**，不含代码改动。它基于 prefix-v2 已验证全链路（`d181dfd0`，本地/origin/4090 三处一致）与两个真实客户端（旧客户端 / dsh-tui）实跑实证，做回顾性重设计，目标是**用比 v2 小一个数量级的改动面达到同等效果，且更稳健**。
> 参考材料：`PREFIX-PLAN.md`（v2 设计，三支柱 + 语义 S1/S2/S3 + 目标原话）、`HANDOFF.md`（v2 复用失效确诊）、`NOTE.md`（持久工作日志）、prefix-v2 分支完整提交链（§0.2）、`DOING.md`（V3 执行过程日志）。
> 基线声明：V3 本体落在 **prefix-v3 分支 @`4f0faf0c`**（= `feat(frontend): add the registered v22_4 chat template`；与 prefix-v2 的 merge-base 即自身 HEAD，是 **v2 全部提交（三支柱 + 稳定化 + P1 修复）之前的新基线**，**已含 v1 前缀复用机制**：endpoint_valid / prepare_consumed_source / PrivateEndpoint 复用路径 / state_image_store / 引擎 prefix index / v22_4 模板均在基线存在）。prefix-v2 @`d181dfd0` 为**冻结参考锚点**（仅对照，不回移、不照搬）。本文 §1/§3 机制描述以 prefix-v3 基线为准（附录 A.2 行号）；附录 A.1 行号属 d181dfd0 树，仅供 v2 对照。
> 一句话结论：前缀复用真正生效的机制只有两个——**key 重锚到 prompt 边界** 与 **复用不毁源**；最小实现约 60–90 行生产代码、2–3 个文件，引擎侧（`resource_manager.h`）核心路径**零改动**（多流加固的可选优化项至多 1 个通知点，见 §1.5）。

---

## 0. 背景：v2 走了什么，证明了什么

### 0.1 原始缺陷与确诊（细节见 HANDOFF §4）

- **现象**：4090 真实 serve（`device_state_slots=1 + shared=1 → total=2` 槽池 + MTP + preserve_thinking + v22_4 模板）下每轮都全量 re-prefill（`prefix_reuse_path=root, prefix_cache_hit_tokens=0`）；同一客户端在 prefix-v1 上复用正常（用户已裁决排除客户端侧）。
- **确诊**（`0766142a` divergence 扫描）：存储端点的 key 锚在**执行前沿**（execution_frontier，生成段末尾）。存储 digest 链把**原始生成 token 流**原样记录——包括模型生成途中吐出的模板控制 token 簇（如 `13, 198, 248069, 271, 97625`，数字代指角色/边界特殊 token）；而客户端回放对话时是**从解析后文本经 chat 模板重新序列化**，生成段在第一个控制 token 簇处必然产生不同 token → digest 链在该点分叉 → 每轮 key-miss。每轮 divergence 点精确落在生成段前部第一个控制簇（req2: 58743；req3: 60574；…）；**prompt 段两侧逐 token 全匹配**，identity_tag 两侧一致（327937）。
- **叠加**：兜底的 anchor/turn-closure 通道在此 2 槽池下结构性死亡（运行中 root 占 1 槽 + retained root 占 1 槽 = 池满，`anchors=0` 恒成立）→ 两条复用通道全灭 → 100% 全量。
- **推断**（HANDOFF 标注未核实）：v1 命中大概率走 key 锚在 prompt 侧边界、不受生成段重序列化影响的通道（anchor/turn-closure）。

### 0.2 v2 实际修复链（剔除 trace/tripwire；均已提交并经 4090 实跑验证）

| commit | 机制 | 治了什么（实跑暴露的故障） |
|---|---|---|
| `9b727428` **D3** | 端点 key 重锚到最后一个 prompt 边界；prompt 边界 plain 捕获钉进 P2-A 预置槽；finish() fork-abort 本应使冻结 prompt 边界镜像成为 endpoint；`endpoint_frontier` 贯穿 populate 与 planner | （纸面设计）原始缺陷 |
| `b1a5441f` **D6** | plain prompt 边界组变成"零需求可预约私有捕获"（可行 iff 有预置槽，不装索引项，destination=预置槽） | plain 组被 `inspect_capture` 判成"空捕获"（`publishes_private=false`），引擎+program 双双在 prepare 前 SKIP 掉 offer，D3 从未触发 |
| `931e7315` **D7** | plain prompt 边界组在 `inspect_lane` transfer 中保留 | 边界组的预置槽/destination 信息活不过 transfer |
| `57f9e87b` **方案 B** | `publishes_checkpoint` 标志贯穿 CaptureAssessment/ActiveCaptureTransaction/ActiveCaptureRecord；state-only 发布跳过 populate 守卫与引擎 settle 校验/合并 | 边界发布是 state-only（endpoint 在 finish() 才成形）→ publish 时 continuation summary 为空 → program 侧 populate 守卫与引擎 settle 校验都拒 → `private continuation has no checkpoint` |
| `bf3e9796` **D7c** | `prepare_consumed_source` drop 分支对 PrivateEndpoint 跳过主图 drop | drop 分支（`execution_frontier > reuse_base`）把**复用载体本身**（被消费的 endpoint 主图）销毁 → `materialization source has no resident state` → 引擎 fail-stop → 后续全 503 |

> **更正（基线实测）**：另有 5 个支柱稳定化提交（`2c26c74f` entitlement 允许 actual<expected、`809537ff` P2-C stale assessment、`bbb41e3b` 池耗尽测试、`a066cb15` 活跃映射页双计数、`3a6b7d06` P2-A 容量自适应）与 2 个 P1 死锁链提交（`e341e68f` UAF+M3+committed 排除、`1300f493` admit 成功释放 pending 记录）——这些治的是 v2 自身的真 bug，**与"复用 key 锚在哪"无关**；但它们经 `git merge-base --is-ancestor` 实测**全部 OUT 于 prefix-v3 基线**（v2 的全部提交都不在基线中，只有 v1 机制在基线中，见 §0.5）。V3 若复现其故障需逐个移植单点修复，不在本文"可精简"范围内。

### 0.3 机制事后复盘：实际生效的机制远比设计简单

- 所有真实 run（两客户端、20+ 轮）`finish: fork=0` **恒成立**：第一次 MTP decode commit 就触发 `settle_state_fork` 提交 fork 并释放冻结源 → **"冻结 prompt 边界镜像"设计从未真正成形**。
- 实际生效的 endpoint = **执行图 + `endpoint_frontier` 字段**：执行图的 KV 天然含 0..prompt_tokens 的完整有效前缀（它自己 prefill 的 prompt 段）；`prompt..exec` 尾是陈旧的但无害——消费时由既有 truncate 路径清掉（`trim_sequence_kv(base)` + 后端 destructive truncate），plan 把 ConsumeToActive 建模为 move（槽位传给 lane），记账自洽（实跑 `materialize: src_dev_slots=1` 恒通过 entitlement 检查）。
- **推论**：设计前提"endpoint 图必须是冻结的 prompt 边界镜像"**是多余的**。D6/D7/方案 B 都是为让冻结镜像物化跑通而存在的"机制配套"改动——若一开始走"前沿重锚 + 执行图作 endpoint"（约 30 行），这三个提交根本不会存在。当时每个都治了真实发生的故障（不是改错），多花的 3 个提交 = 忠实实现了一个设计上规定、事后证明不必要的物化机制。

### 0.4 V3 的命题

重头实现前缀复用：达到**与当前 `d181dfd0` 相同的效果**（round1 root；round2+ 每轮命中上一轮 prompt 边界；上下文压缩事件只付一次全量 prefill；两个真实客户端的契约行为可复现），改动面**比 v2 小一个数量级**，且**更稳健**（全配置通用、承重约定减半）。

### 0.5 V1/V2 参考策略与基线事实（V3 重做同步）

- **参考、不照搬**：prefix-v1 / prefix-v2 只作机制与教训的参考锚点（哪条通道实际生效、哪些故障真实发生、修法语义是什么）；代码**不直接照搬**——v2 代码携带 V3 基线中不存在的整套三支柱机制（边界 CaptureGroup / D6 零需求预约 / 方案 B `publishes_checkpoint` / P2-A 预置槽）。V3 落点在 prefix-v3 基线上逐处重新推导（附录 A.2）；机制与 v2 同构处（M1 重锚、M2 不毁源）也按基线现状独立实现，而非 diff 搬运。
- **基线事实（git 实测）**：prefix-v3 HEAD = `4f0faf0c` = `merge-base(prefix-v3, prefix-v2)`，即早于 v2 全部提交的新基线；v2 三支柱提交（`9b727428`/`b1a5441f`/`931e7315`/`57f9e87b`/`bf3e9796`）与 7 个稳定化/P1 提交经 `git merge-base --is-ancestor` 验证**全部 OUT** 于该基线；v1 前缀复用机制（endpoint_valid / prepare_consumed_source / PrivateEndpoint 路径 / state_image_store / 引擎 prefix index / v22_4 模板）已 **IN** 于该基线。
- **继承风险清单**（v2 已修真 bug、V3 基线不含；若 V3 测试/实跑复现对应故障，移植对应单点修复并在 DOING.md 记录，不引入三支柱机制）：
  - `2c26c74f` entitlement 放宽（允许 actual<expected）——基线 `start_request`（`program_impl.h:7171-7177`）仍为严格 `actual != expected → throw materialized sequence does not match its active entitlement`，复用轮 move 记账可能踩中（R-V3-2 测试第 0 天即可捕获）。
  - `809537ff` P2-C stale assessment / `a066cb15` 活跃映射页双计数 / `3a6b7d06` P2-A 容量自适应——池/评估类，压力与复用规模增大时可能复现。
  - `e341e68f` UAF+MTP+committed 排除 / `1300f493` admit 成功释放 pending 记录——P1 死锁链仅在 `max_concurrency≥2` 暴露，正是 R-V3-6 并发测试的暴露面。

---

## 1. V3 最小方案

### 1.1 两个核心机制

**M1 key 重锚（frontier re-anchor）**：私有端点存储 key 的前沿恒为该 lane 的 **prompt 边界**（`prefill.prompt_tokens`），绝不落在生成段内部。digest 表在 prefill 时已天然含 prompt 段前缀 digest，key 只取已算好的链值，零额外成本。
- 端点**状态图复用执行图**（finish 冻结后 lane 的唯一驻留图），不新建镜像、不 fork、不占额外槽：其 KV 必含 0..frontier 完整前缀；frontier 之后的尾是陈旧的、可存在，消费时裁剪（M2 / S-V3-2）。
- **赋值落点（关键工程细节，已按 prefix-v3 基线核实并定案）**：`SequenceState`（`program.h:405-434`）**没有** `prompt_tokens` 成员——`prompt_tokens` 住在 `RequestControl::Prefill`（基线 `program.h:479`）。基线定案落法：
  - `SequenceState` 加 `std::uint32_t prompt_boundary = 0;`；**单点写入 = `start_sequence` 的 staging 校验处**（基线 `program_impl.h:9450-9457`：`staged=*request.prefill` 且 `materialization staging is incomplete` 检查之后、Root/preserving/consume 三分支之前）写 `sequence.prompt_boundary = staged.prompt_tokens;`——`start_sequence` 是所有已物化 sequence（Root/保留/消费三条路径）生效的唯一入口，单点覆盖全路径；
  - finish() 内 `endpoint_valid=true`（基线 `:9195`）之后、populate（`:9199`）之前写 `state.endpoint_frontier = state.prompt_boundary;`（**单赋值点**，钉在 populate 之前）。
- v2 是在 **publish 时**（`prefill` 在作用域内，d181 树 `program_impl.h:8665-8666`）双点赋值 `endpoint_frontier`；V3 无 mid-prefill 边界发布，改为 finish() 单点赋值 + start_sequence 的 prompt_boundary，更简单也更不易被打散。另：finish() 的 fork-abort 分支（基线 `:9173-9180`）**实现定案=保留不删**（偏离原设计"删除"）——该块是基线 v1 防御代码（非常规路径）：V3 树内 finish 时的 fork 来自 prefix COW/DFlash 既有机制；删除会使"零 token 生成轮 + pending fork"的端点永久不可消费（start_sequence 检查 `state.fork_pending` 会走 a2 受控中止）且 destination 泄漏；保留块在常规路径恒不触发，零成本。finish trace 记录 `fork_pending` 供实跑证伪。

**M2 复用不毁源（no-drop on reuse）**：`prepare_consumed_source` 清理"陈旧 endpoint"的 drop 分支必须跳过 **PrivateEndpoint** 路径的主图——被消费的 PrivateEndpoint lane 接管的就是那张图本身（selected state **就是** `source.state.read`）；`execution_frontier > reuse_base` 只是陈旧 KV/hidden 尾（由既有 truncate 清），不能作为销毁载体的依据。其余复用路径（TurnClosure/ResponseReplay/LongAnchor）drop 行为不变（那些路径 selected 图是 checkpoint 图、非主图）。

**为什么"执行图当 endpoint"安全（M1 选图的论证）**：
1. **前缀 KV 有效性**：执行图 0..frontier 段是该 lane 的真实 prefill KV，digest 链与客户端重序列化一致（实跑实证 prompt 段逐 token 两侧匹配）。
2. **尾无害**：消费时 `trim_sequence_kv(reuse_base)` + 后端 destructive truncate 清尾；plan 建模 move（槽位传 lane），`materialize` 驻留检查与 entitlement 检查在 v2 终态实跑全通过（v2 终态 endpoint 正是执行图）。
3. **MTP 兼容**：`mtp_kv_valid = backend_frontier_at(Mtp, reuse_base)` 重置到边界 + BeforeSuffix 桥重算——实跑所有复用轮（`consume-source → materialize → start-seq refs=0 fork_required=0 → prefill-end base>0`）证明全链跑通。

### 1.2 改动落点（对照基线 `d181dfd0`；新基线重做时同位置重推）

| 文件 | 位置 | 改动 | 规模 |
|---|---|---|---|
| `src/targets/qwen3_6/impl/runtime/program.h` | `SequenceState`（:405-434） | 加 `std::uint32_t endpoint_frontier = 0;` 与 `std::uint32_t prompt_boundary = 0;` | 2 行 |
| 同文件（prepare/prefill 起点） | 请求建立处 | `sequence.prompt_boundary = prefill.prompt_tokens;`（基线定案落点 = `start_sequence` staging 校验处，§1.1） | 1 行 |
| `src/targets/qwen3_6/impl/runtime/program_impl.h` `finish()`（现 :9607-9664） | `endpoint_valid=true` 后、populate（现 :9643）前 | `state.endpoint_frontier = state.prompt_boundary;`（单赋值点）；fork-abort 分支**保留**（基线 v1 防御，§1.1 定案理由）；加 finish trace | ~3 行 |
| 同文件 `populate_continuation_summary`（现 :7452-7521） | endpoint 分支 | frontier 取 `endpoint_frontier != 0 && endpoint_frontier < execution_frontier ? endpoint_frontier : execution_frontier`（现 :7464-7468 的 min 逻辑）；`endpoint_frontier < execution_frontier` 时 rebuild_work 取 `make_prefill_work(0, endpoint_frontier, …)`（现 :7474-7479） | ~5 行（现成逻辑照搬） |
| 同文件 `prepare_consumed_source`（现 :4693-4694） | drop 分支 | 条件追加 `&& details.reuse != ReusePath::PrivateEndpoint` | 1 行 |
| `src/targets/qwen3_6/impl/runtime/request_plan_impl.h` planner | PrivateEndpoint 的 reuse_base / MTP append_ready | 自然兼容（reuse_base 从 summary 前沿流入；`mtp_kv_valid >= reuse_base-1` 恒成立）→ 以测试确认，原则 0 行 | 0 行 |
| `src/runtime/engine/resource_manager.h` | **整文件** | **核心零改动**（V3 无新 capture 类型、无新 record 字段、无 settle 门控）；可选优化 (b)（§1.5）至多 1 个通知点/谓词修正 | 0 行（可选 0–10 行） |
| 同文件 `start_sequence` PrivateEndpoint 分支（基线 :9771-9827） | role/ownership 检查 | 选中图 borrowed/not-movable → 不再抛异常，本请求**受控收尾**（infeasible-SKIP：Aborted 终态、请求带错结束、引擎存活，见 §1.5 (a2) 基线现实）；选点时（inspect/verdict）被借用的图判 infeasible 按 root 规划（(a1) 是优雅回退 root 的主力） | ~15 行 |
| 引擎侧 consume 时索引失效（可选） | `rebuild_prefix_index`（现 :1864）新增 "consume" 触发 / T2 失效扫描谓词补 role 检查 | 缩小 borrowed 窗口（§1.5 修法 (b)）；(a) 在位时纯优化，可延后 | 0–10 行 |
| `tests/…` 前缀电池 | — | 加 1 个重序列化回归 scenario（§3 R-V3-2）+ 1 个单槽池变体 + 1 个多流并发 scenario（§3 R-V3-6） | 2–3 用例 |

> 表内行号属 d181dfd0 树（v2 对照）；**prefix-v3 基线对应锚点行号见附录 A.2**，实现时以 grep 复核为准。

生产代码合计 **~60–90 行、2–3 个文件**（含 §1.5 多流加固 (a) 双层降级；可选 (b) 另计 0–10 行）；v2 对应改动是 5 个文件 + 测试假件、数百行。

### 1.3 相对 v2 消失的东西（全是"冻结镜像物化"设计的配套机制）

| v2 机制 | V3 消失原因 |
|---|---|
| D6：plain prompt 边界组"零需求可预约私有捕获"评估分支（`inspect_capture`） | V3 不建边界捕获组 → 无边界 offer → 无 RESERVE |
| D7：边界组在 `inspect_lane` transfer 的保留 | 没有需要保留的组 |
| `publishes_checkpoint` 字段及其在 3 类型 + 3 个 record 构造点 + 引擎 settle 校验/合并门控的贯穿 | 根本没有 state-only 发布 → finish 的 populate 必有 endpoint，"no checkpoint" 守卫天然通过 |
| 边界处预置槽消费（DeviceFork 到 reserved slot） | 常规路径无边界 fork（单图；shared 引用场景的既有 fork 机制保留但与边界无关）；finish 的 fork-abort 分支**保留**（基线 v1 防御，非 v2 边界镜像 fork，§1.1） |
| 配置分叉（仅 2 槽池重锚；单槽/DFlash 回退 legacy → 原始 bug 静默复活） | **重锚无条件生效、不占槽、全池配置通用**——严格稳健性增量 |
| 承重约定（难破、无测试保护的隐性依赖）从 7 条降到 2 条（附录 B） | — |

### 1.4 行为等价声明

V3 与 `d181dfd0` 终态运行时行为逐项等价（都：endpoint=执行图、fork=0、key 锚 prompt 边界、消费为 move、压缩事件只付一次全量 prefill）。因此 v2 在两个真实客户端上的验证结论（dsh-tui 13 请求全符合契约：req1/req2 root 为 bootstrap/首段、req3–10 逐轮命中、req11 压缩后单次全量、req12/13 立即恢复命中；旧客户端"压缩后双全量"为其历史非单调重序列化的客户端侧问题）可直接作为 V3 的**行为预期**，仅新基线落地需再验证（§4）。

### 1.5 多流场景（并发复用）

**可行性结论：可行**——M1/M2 并发中性：复用单元是"已 finish 的 lane 的唯一驻留图"（冻结、refs=0/pins=0 → 干净 move），M2 守卫 per-transaction，流 A move 自己的图不碰流 B；跨流复用（C 命中 A 的历史）与同前缀多流（两客户端共享同一段对话前缀，共享一张图，每轮 move→trim→extend）都走同一机制。**但**：此前所有验证均为 `max_concurrency=1`（请求级串行，任何 inspect 时刻图恒为冻结态），并发路径从未被测试，其中 1 个继承竞态仅在 ≥2 流并发时暴露（下节"必须修"）。

**容量（配置问题，非方案缺陷）**

- 每条存活历史 = 1 个 device state 槽 + 全量 KV 驻留（rk8v4）：`total_device_state_slots` ≥ 需保活的并发流数；2 槽池实际只支撑 ~1 条流有效复用。
- 超额的流被压力机制驱逐（实跑已验证：驱逐后该流下轮 re-prefill——`host=0` 时为全量、配置了 host 槽时为 rebuild，优雅降级不报错）；`kv_capacity` 页全流共享，驻留随流数线性涨。
- **范围边界**：V3 的可复用单元只有**已完成**的端点；流 mid-prefill/mid-generation 期间的进度不可被其他流共享（那是 interior checkpoint/D6 机制，V3 裁掉，正交、可后叠，见 §5.2）。

**真实隐患：borrowed-image 窗口竞态（必须修）**

- 机制：A 第 N 轮 finish → 条目 E_A@F_A 入索引（→ 图 S_A，冻结）→ A 第 N+1 轮到达，ConsumeToActive 把 S_A move 成 ActiveMutable、被 A 的 lane 借走在途生成；此时索引中指向 S_A 的条目（至少未消费的旧边界条目）仍在（自清理是被动的：等下次 rebuild/扫描）→ 并发流 C（prompt 匹配 A 的旧边界 F_Ak）inspect 选中 E_Ak → `start_sequence` 的 role 检查（必须 CheckpointImmutable）抛 `resident endpoint StateImage is not movable` → A 在途第 N+1 轮陪葬 → 引擎 fail-stop（`engine_core.h:2099` `failed_=true`、`fail_all_locked`）→ 全量 503。
- 定性：**无数据损坏、只有可用性**——role 检查是 v2 既有的所有权守卫，正确阻止"偷走在途图"；问题是引擎"异常→fail-stop"设计把单请求冲突放大成全局宕机。`max_concurrency=1` 下窗口恒闭合（串行），此前验证对此全盲。
- 落地前需核实：T2 索引失效扫描的谓词是否含 `role==CheckpointImmutable` 检查（决定 (b) 是"修谓词"还是"加触发点"）。基线实测：`valid_prefix_index_entry`（`resource_manager.h:1726`）= Catalogued+handle+id/revision 匹配、**无 role/movable 检查**；引擎 inspect 私有候选已有 `entry.state==Catalogued && handle && !private_has_active_edge(slot)`（`resource_manager.h:343-346`）作第二层。

**修法（纳入 V3 本体）**

- **(a) 双层受控降级（必须，完整关闭 TOCTOU）**：
  - (a1) 选点时（inspect/verdict）：被借用的图 → 该端点判 infeasible → 引擎按 root 规划（零浪费、引擎侧零改动）。program 侧三处：`inspect_admission` 源 slot 非 Catalogued 时 return nullopt（不再 throw stale）；`inspect_lane` PrivateEndpoint 分支 prefix-match 后加借用/不可移动检查 → nullopt；`reserve_materialization` preflight=StalePolicyState → return Aborted（引擎 :448-452 转 Stale → 请求留队重 inspect 优雅重试），InvariantFailure 保留 throw；
  - (a2) `start_sequence`（基线 :9771-9827）最后防线：选中图 borrowed/not-movable 时**不抛异常**——**基线现实**：引擎在 commit（materializing_）之后**没有重排队通道**（Aborted → `complete_detached_cancelled`，`engine_core.h:1703-1713`），无法以 ~15 行做 root re-plan；故 (a2) 收尾 = **infeasible-SKIP**（请求带错结束、引擎存活不 fail-stop，S-V3-5 允许的两种收尾之一），覆盖 inspect 与 start 之间的 TOCTOU 罕见尾巴；**常规回退 root 由 (a1) 承担**（选点时判 infeasible → 引擎自然按 root 规划，客户端无感）。**实现落法**：匿名命名空间 `class SourceUnavailableException final : public std::runtime_error`（消息 `materialization source became unavailable before publication`）；start_sequence PrivateEndpoint 分支 6 处 throw（role 块 `resident endpoint StateImage is not movable`、text/MTP/DFlash coverage、rewrite `resident rewrite checkpoint has no complete KV allocation`/`resident rewrite StateImage is not movable`）改抛该异常；执行主循环 publish try/catch 前加 `catch (const SourceUnavailableException&)`：source slot 已 stale 则先清 `has_source`（避免 `complete_source_acknowledgement(false)` 对 stale slot 二次 throw）→ `cancel_pending` → `abort_transaction()`（自带 staging release+terminal+out.status=Aborted+acknowledgements）→ return；`prepare_consumed_source` 的 slot 检查（基线 `:4624-4629`）同样改受控中止（cancel_pending+has_source=false 后 return，清 claim 使 ack 重校验 no-op 化）；`!sequence.kv`（无 KV bundle）与 endpoint/rewrite 别名检查保留 throw（编程错误类）。
- **(b) consume 时主动失效（推荐优化，缩小窗口）**：lane 的图 冻结→激活 的瞬间立即失效索引中指向该图的条目——实现二选一：`rebuild_prefix_index` 加 `"consume"` 触发点（program→engine 一个通知点），或 T2 扫描谓词补 role 检查；引擎侧 ≤10 行。(a) 在位时此项纯优化，可延后。
- **(c) 并发回归测试（R-V3-6）**：见 §3；实跑配置见 §4 第 6 条。

---

## 2. 语义不变量（V3）

- **S-V3-1 key 只锚 prompt 边界**：私有端点存储 key 的前沿恒为 `prefill.prompt_tokens`；禁止把 key 锚在生成段内任意位置（这是 v2 原始缺陷的唯一根因）。
- **S-V3-2 端点图可以是执行图**：端点图的 KV 必含 0..frontier 完整前缀；frontier 之后的陈旧尾可存在、消费时裁剪；**不**要求端点图是"干净镜像"。
- **S-V3-3 复用不毁源**：被消费 PrivateEndpoint lane 的主图就是复用载体本身，任何源清理逻辑不得 drop/release 它；move 之后 `materialize` 驻留检查必须通过。
- **S-V3-4 客户端失效可观测**：同一会话（key = conversation_key，首条非 system 消息的摘要；会话累计 digest 每轮必变、不可作键）连续两轮 `prefix_reuse_path=root` → serve 打一条 warning（客户端历史非单调重序列化的金丝雀；引擎修不了这类问题，但要能一眼归因给客户端）；该轮为 root 且存在上轮记录时，另在上轮 KV 边界处打 21-token 窗口（root-diag，见 §6）。
- **S-V3-5 单请求冲突不炸全局**：并发下任何"选中图已被借用/not-movable"情形，必须以受控回退（(a1) 层 infeasible→root 规划）或 infeasible-SKIP（(a2) 层 Aborted 收尾）收尾；引擎不得因单请求的状态冲突 fail-stop（fail-stop 仅保留给真正的引擎级故障）。
- （v2 的 S1 上限不预留 / S3 按需分配不变，不在本文范围。）

---

## 3. 流程与安全规则（重做教训）

- **R-V3-1 最小优先**：埋点（key-miss + divergence 扫描）→ 确诊 → **先上 ~40 行最小修复到实机** → 只有实测出现真实缺口（exact-hit 陈旧 tail、长生成内存、required_kv 不一致等）才加冻结镜像这类重机制。禁止"先把纸面设计忠实实现、再验证最小假设"（v2 正是此顺序，多出 3 个提交）。
- **R-V3-2 盲区测试第 0 天就位**：修复提交之前，前缀电池必须先含一个重序列化回归 scenario：同会话 ≥2 轮 + 2 槽池 + MTP + preserve_thinking + **客户端式重序列化**（把 round1 响应解析后、经 chat 模板重序列化 round2 历史），断言 round2 `reuse_tokens = round1 prompt_tokens` 且 `path=private_endpoint`。这是能同时抓住 v2 五个 bug 的唯一测试；v2 期间电池全绿、实跑全灭是本案例最大的安全教训。
- **R-V3-3 实现期关键点位预埋埋点（opt-in、行为零影响）**：**在实施代码的同时**，把所有关键点位的全链 trace 一次性埋好 + prefill tripwire——目标是**测试时跑一次就能收集到全部关键点位日志、直接定位问题根因，避免反复重跑**。它是"每层实跑验证"模型的前提，也是快速定位客户端差异（如两个客户端）的工具。用户 2026-09-10/11 明确指示：v3 实现时关键点位提前埋好 trace。**点位清单、门控环境变量、tripwire 语义见 §6**（基线现状：全仓无任何 `NINFER_ADMISSION_TRACE`/`NINFER_PREFILL_TRIPWIRE` 埋点，V3 全部新埋；参考 v2 `6e7b508c`/`d181dfd0` 的做法但不照搬）。
- **R-V3-4 不变量写成测试而非注释**：S-V3-1/3 各配一个断言级测试（含钉住"`endpoint_frontier` 单赋值点在 populate 之前"、钉住"drop 守卫只豁免 PrivateEndpoint"）。
- **R-V3-5 冻结基线不动**：prefix-v2 分支（`d181dfd0`）保持 4090 部署的已验证基线；V3 落新分支、不回移；V3 任何改动走完整周期（本地 docker 绿 → 4090 同步编译 → 两客户端实跑各一轮）。
- **R-V3-6 多流并发测试与实跑**：前缀电池含并发 scenario——流 A 第 2 轮 in-flight ∥ 流 C（prompt = A 的 round1 边界）→ 断言 C 优雅走 root、无 500、无 fail-stop；再加双流交错长跑。§4 实跑配置用 `max_concurrency≥2`（`total_device_state_slots` 按流数相应调大）。

---

## 4. 验证计划（V3 落地后）

1. **本地**：`docker run --rm -v <repo>:/src -w /src/build ninfer-local-build:latest sh -c "ninja && ctest --output-on-failure"`（无 GPU，GPU 用例本地 skip；期望全绿）。
2. **4090**：`cd /d C:\Data\ninfer\ninfer-4090-kaso && git fetch origin <v3-branch> && git merge --ff-only origin/<v3-branch>` + `docker run --rm -v C:\Data\ninfer\ninfer-4090-kaso:/ninfer-4090-kaso -w /ninfer-4090-kaso/build ninfer-4090-kaso-devel:0905 ninja`（需用户先停常驻推理容器，nvidia-smi 显存 >20GB 即被占）。
3. **实跑**：`build\apps\ninfer-serve.exe /models/qwen3_8_27b.ninfer --preserve-thinking --chat-template v22_4 --spec mtp --draft-tokens 3 --lm-head-draft --kv-dtype rk8v4 --max-context 200000 --request-log-jsonl logs/log.txt`。
   - **多轮验收 run**：前缀只设 `NINFER_ADMISSION_TRACE=1`（tripwire 关）；真实客户端跑 2–3 轮（round1 长 prompt）+ 1 轮模拟上下文压缩事件 + 1 轮 exact-hit 重发（相同 prompt）。
   - **单轮冒烟 run**：`NINFER_ADMISSION_TRACE=1 NINFER_PREFILL_TRIPWIRE=1`（触发语义见 §6；round2 触发即设计内的 run 终点，看其前 trace 判复用是否生效：round2 `base>0` → 好，`base=0` → 坏）。
4. **验收**（逐项对照 `d181dfd0` 基线行为）：
   - round1：`path=root`；finish 后 `ep_frontier=prompt_tokens`、索引条目 `front=prompt_tokens` 入册；
   - round2/3：`inspect_admission reuse_tokens=上轮 prompt_tokens` → `consume-source reuse=1 state_read_valid=1` → `materialize src_dev_slots=1`（不抛）→ `start-seq refs=0 fork_required=0`（move）→ `prefill-end base>0` → `path=private_endpoint`；
   - 压缩轮：只付一次全量 prefill，下一轮立即命中；
   - 多轮 run（tripwire 关）：`prefill-end` trace 显示 round2+ `base>0`、`path=private_endpoint`；单轮冒烟 run（tripwire 开）：正确 run 不触发，触发即说明复用失效（看触发前全链 trace）；无 500/503。
5. **变体验证**：单槽池配置（验证 S-V3-1 通用性，即 1 槽配置下重锚不丢失）与 DFlash/无 MTP 配置（finish 无 fork 路径）各跑一轮。
6. **多流并发验证**：`max_concurrency=2`（`total_device_state_slots` 按流数调大）：流 A 长 prompt 2 轮 ∥ 流 C 命中 A 的 round1 边界 → C 优雅走 root（不 500/503）；双流交错 5+ 轮无 fail-stop；压力驱逐场景（流数 > 槽数）验证优雅降级。

---

## 5. 诚实边界与已知边缘

1. **V3 是从 `d181dfd0` 实际生效机制反推**的（每条主张都有实跑/代码证据），置信度高，但它**不是**一个被独立验证过的成品：必须在新基线走完整 §4 验证周期后才能视为有效。
2. **结构性限制不变**：2 槽池下复用粒度 = "上一整段 prompt"（prefill 中 interior checkpoint 恒 infeasible）。若要上下文压缩后还能挽救稳定前缀（如某 run 的 10883 token 公共前缀），仍需 `total_device_state_slots` 3–4 + interior checkpoint 机制——与本文正交，需要时再叠加。
3. **已知边缘（v2 终态与 V3 共有，"执行图当端点"的固有属性，非任一方案引入）**：
   - exact-hit 追加路径（客户端重发字节相同 prompt，`base==prompt_tokens`）的 tail_hidden 陈旧：BeforeSuffix 桥重算、安全；实跑无反例；
   - dangling 前缀索引条目（指向已释放图）依赖 digest/invalid 扫描被动自清——v2 实跑确认是预期行为，但无专门测试（V3 可顺手补一个）。
4. **不回移**：V3 不改 prefix-v2 任何文件；重做的价值在下一个基线（更干净改动面、承重约定减半、全配置覆盖），不在当前 4090 部署（`d181dfd0` 冻结不动）。
5. **多流并发的范围**：borrowed 图竞态的双层受控降级（§1.5 (a)）已纳入 V3 本体；consume 时主动失效（(b)）为可选优化；**在途前缀共享**（mid-prefill 进度被旁路流命中）仍为后续叠加项（interior checkpoint，与本文正交）。

---

## 6. Trace 日志设计（R-V3-3 落地规范）

**门控与格式**：
- `NINFER_ADMISSION_TRACE`：环境变量**非空**即启用全链 trace；统一 `stderr` 格式 `[admission-trace] <point>: key=value ...`，每行 `fflush`；关闭时零行为影响（不打印任何日志）。
- `NINFER_PREFILL_TRIPWIRE`：非空启用 prefill tripwire（语义见下）。
- S-V3-4 的 serve warning 走 serve 的 warning/请求日志通道（**实际落点 = `src/serve/generation_service.cpp`** `GenerationService::run()` 尾部：按 conversation_key（首条非 system 消息的摘要；原设计"session digest"是每轮必变的累计 ledger digest、warning 永不触发的 bug，已替换）跟踪 `last_prefix_path_`，连续两轮 root → stderr `[prefix-warn]` + `logger_->warn`），不属于 admission-trace 通道。例外：`root-diag`（21-token 窗口，见下表）常开、不受 `NINFER_ADMISSION_TRACE` 门控。

**点位表**（V3 全集；除 serve 层 `root-diag` 常开外，每点均受 `NINFER_ADMISSION_TRACE` 门控；行号 = prefix-v3 基线，附录 A.2）：

| point | 落点 | 输出内容 |
|---|---|---|
| `plan` | `plan_request`（`request_plan_impl.h:~300`） | `prompt=.. allow=.. reusable=.. cache=.. publish=.. backend=mtp|dflash|none` |
| `offer` | `wrap_prefill`（`program_impl.h:7131-7162`） | `lane=.. offer=.. base=.. cursor=..` |
| `inspect` | `inspect_admission`/`inspect_lane`（`program_impl.h:1225-1310` / `request_plan_impl.h:447+`） | `lane=.. reuse_tokens=.. path=..`；跳过原因：`source-stale-skip`（(a1)-1 源 slot 非 Catalogued）、`borrowed-stale-skip`（(a1)-2 图被借用/不可移动） |
| `verdict`/`RESERVE` | 引擎 `inspect`（`resource_manager.h:273-405`）；program `reserve_materialization`（`program_impl.h:4187+`）；引擎 `reserve_materialization`（`resource_manager.h:418-463`） | verdict：`lane=.. candidates=.. selected=none` 或 `dest=.. private=.. shared=.. mode=.. reuse_tokens=.. pub_slot=..`；program RESERVE：`result=aborted(cancelled)` / `result=aborted(stale-policy)`（(a1)-3）/ `lane=.. result=reserved`；引擎 RESERVE：`lane=.. result=stale(revision)` / `aborted`|`stale` / `reserved` |
| `prepare` | `prepare_consumed_source`（`program_impl.h:4611+`） | `src_slot=.. reuse=.. mode=consume` |
| `consume-source` | 主图 drop 分支处（`program_impl.h:4650`） | `reuse=.. exec=.. reuse_base=.. endpoint_valid=1 ep_frontier=.. state_read_valid=.. mtp_kv_valid=.. action=drop`；PrivateEndpoint 被 M2 守卫豁免时 `reuse=private_endpoint ... action=retain(M2-guard)` |
| `materialize` | 执行主循环（`program_impl.h:6303+`）+ `start_request`（:7171-7177 校验前） | `lane=.. has_source=.. reserved_states=..`；`entitlement lane=.. reuse=..`（v2 D7c 的 entitlement 炸点可观测） |
| `start-seq` | `start_sequence`（`program_impl.h:9447+`） | `lane=.. prompt=.. base=.. path=..`（prompt_boundary 写入后立即打）；(a2) 受控中止：`a2-abort lane=..` |
| `summary` | `populate_continuation_summary`（`program_impl.h:7264-7333`） | `endpoint_valid=1 endpoint_frontier=.. exec=.. stored=..`（stored=发布 key 前沿，S-V3-1 可观测） |
| `finish` | `finish()`（`program_impl.h:9117-9220`） | `lane=.. endpoint_frontier=.. exec=.. fork_pending=..`（fork 块保留决策的实跑证伪位） |
| `fork-settle` | `settle_state_fork`（`program_impl.h:10431-10445`） | `source_valid=.. dest_valid=..` |
| `publish` | `commit(PendingBatch)`（`program_impl.h:9090+`） | `lane=.. offer=.. frontier=..`（OFFER 正常发布）；`lane=.. inconsistent cursor=.. next=.. pending_offer=..`（INCONSISTENT 守卫触发前） |
| `index-rebuild` | `rebuild_prefix_index`（`resource_manager.h:1687`） | `entries=..`（重建后目录条目数） |
| `prefill-end` | `advance_prefill`（BeginSummary 起点 / prefill 结束） | `prompt=.. base=.. path=.. next_frontier=..`（每轮验收的直接证据） |
| `root-gate`/`root-gate-entry` | 引擎 `inspect()`（`resource_manager.h`，候选循环 + 判定收尾） | 对每个 prefix-index 条目记录门控判定：`stale-entry`（引擎侧条目失效/丢失：驱逐/退休/索引 bug）/ `key-mismatch`（内容与缓存分叉：客户端改写或 hash bug）/ `not-catalogued`（私有未发布或被借用、共享未发布，M3 受控窗口）/ `inspect-skip`（(a1) 主动让位）/ `candidate`（真正成为候选，带 `reuse`）。输出条件：① 判定 `selected=none` → `root-gate: lane=.. prompt=.. selected=none entries=N`（entries=0 时追加 `(no cached boundary)`）；② 选中候选 `reuse_tokens=0`（走 root）→ `root-gate: lane=.. prompt=.. selected=root entries=N (cached boundary lost or content diverged)`；两者后随最多 8 行 `root-gate-entry: frontier=.. shared=.. gate=.. reuse=..` |
| `root-diag` | serve `GenerationService::run()`（`src/serve/generation_service.cpp`；**常开，不受 `NINFER_ADMISSION_TRACE` 门控**） | 本轮 `path=root` 且该会话有上轮记录时（首轮 root 无边界可对比，不打）：`conversation=<key> lane=<id_slot> previous_boundary=<F> prompt_tokens=<n> before10=[t…] center=[t] after10=[t…]` —— F = 上轮 `prompt_tokens`（KV cache 末端，即"新消息和 kvcache 对不上的位置"锚点）；before10 = tokens[F-10..F-1]（F<10 时能凑几个打几个）；center = token[F]（边界后第一个 token，即理论上首个不匹配的 token）；after10 = tokens[F+1..F+10] 到 prompt 末尾截断（F ≥ prompt 长度时 center/after 为空）。打印渲染后 token id，解码成文本需离线用 artifact tokenizer，属预期；同时 `logger_->warn` |

**prefill tripwire（`NINFER_PREFILL_TRIPWIRE`）**：每次 prefill session 起点（`staged.cursor == staged.base`，即新 prefill session 开始）打印一行 `TRIPWIRE: prefill session=%u prompt=%u base=%u reuse=<path>`（`static std::atomic<uint32>` 计数器）。**V3 触发语义（与 v2"session>1 即 throw"不同）**：`base==0 && 上一 session base==0 && 两次 prompt 不同`——即**连续两次全量 re-prefill**（v2 原始故障签名）才触发（打印 `TRIPWIRE:` + throw 中止 run）；压缩事件后的合法单次全量（单个 base==0）不触发。
**使用分工（§4.3/§4.4 的自洽化澄清）**：
- **多轮验收 run**：只开 `NINFER_ADMISSION_TRACE=1`、**tripwire 关** → "无 TRIPWIRE 行"天然成立，每轮 base/reuse 由 `prefill-end` trace 提供；
- **单轮冒烟 / 故障诊断 run**：加 `NINFER_PREFILL_TRIPWIRE=1`——正确 run：round1 base=0、round2 base>0 → 不触发；复用失效（每轮 base=0 且 prompt 不同）→ round2 触发并中止，此前已抓到全链 trace 直接定位根因。

**S-V3-4 金丝雀（serve 侧）**：同一会话（key = conversation_key，首条非 system 消息的摘要；原设计"session digest"为每轮必变的累计 ledger digest，已替换）连续两轮 `prefix_reuse_path=root` → serve 打一条 warning（提示客户端历史非单调重序列化；引擎修不了此类问题，但可一眼归因给客户端）。落点：`src/serve/generation_service.cpp` 的 `GenerationService::run()`（conversation_key → 上轮 `PrefixPathRecord{path, prompt_tokens}` map；原设计写的 `apps/serve/main.cpp` 无会话跟踪结构，实现期改落 generation_service）。

---

## 7. 实现状态与偏离（prefix-v3 @`4f0faf0c`，截至 2026-09-11）

**已实现**（全部落在 prefix-v3 分支，工作区未提交）：

- **M1**：`program.h` `SequenceState` 加 `endpoint_frontier`/`prompt_boundary` 两字段（`endpoint_valid` 之后）；`start_sequence` staging 校验后单点写 `sequence.prompt_boundary = staged.prompt_tokens;`；`finish()` 在 `endpoint_valid=true` 后单点写 `state.endpoint_frontier = state.prompt_boundary;`（populate 之前）；`populate_continuation_summary` endpoint 分支 min 前沿 + `make_prefill_work(0, endpoint_frontier, …)`；`publish_checkpoint_drop` 一致性检查同步改为比较发布前沿；7 处 `endpoint_valid=false` 重置点全部同步 `endpoint_frontier=0`；`inspect_lane` siH 条件改为 `selected.frontier==0 || (selected.frontier!=source->execution_frontier && (source->endpoint_frontier==0 || selected.frontier!=source->endpoint_frontier)) → throw catalog endpoint summary disagrees with Program state`（ep==0 退化回退旧行为）。
- **M2**：`prepare_consumed_source` drop 分支条件加 `&& details.reuse != ReusePath::PrivateEndpoint` + 分支内 `source.endpoint_frontier=0`。
- **M3**：(a1)-1 `inspect_admission` 源 slot 非 Catalogued → `return nullopt`（原 throw `admission source continuation is stale`）；(a1)-2 `inspect_lane` PrivateEndpoint 分支 prefix-match 后加 `!endpoint_valid || !valid(state.read) || role(state.read)!=CheckpointImmutable → nullopt`；(a1)-3 `reserve_materialization` preflight=StalePolicyState → `Aborted`（引擎转 Stale 优雅重试，InvariantFailure 保留 throw）；(a2) 匿名命名空间 `SourceUnavailableException`（`materialization source became unavailable before publication`），start_sequence 6 处 throw 改抛该异常，执行主循环 publish catch 特判 → 清 `has_source` → `cancel_pending` → `abort_transaction()`（infeasible-SKIP）；`prepare_consumed_source` slot 检查同样受控中止。
- **S-V3-4**：`src/serve/generation_service.cpp`（非 `apps/serve/main.cpp`，该文件无会话跟踪结构）`GenerationService::run()` 尾部：conversation_key（首条非 system 消息的摘要；原键 `session_digest` = 每轮必变的累计 ledger digest，为 warning 永不触发的 bug，4090 实跑修复轮已改）→ `last_prefix_path_` map（`PrefixPathRecord{path, prompt_tokens}`，mutex 保护，>8192 clear），连续两轮 root → stderr `[prefix-warn]` + `logger_->warn`；root 轮 + 存在上轮记录 → root-diag 21-token 窗口（见 §6）。
- **Trace**：新头 `src/runtime/contract/admission_trace.h`（`ninfer::runtime` ns；`admission_trace`/`prefill_tripwire_enabled` env 非空 static 缓存；`prefix_reuse_path_name`）；14 点位全埋（§6 点位表 = 实际字段）；`resource_manager.h` 6 处（include、verdict×2、RESERVE×3、index-rebuild）；tripwire = `NINFER_PREFILL_TRIPWIRE` 时每个 prefill session 起点（`staged.cursor==staged.base`）打 `TRIPWIRE: prefill session=%u prompt=%u base=%u reuse=%s`，V3 触发 = `base==0 && 上session base==0 && 两次 prompt FNV-1a hash 不同` → throw（单次合法 base==0 不触发）；4090 实跑修复轮追加：root-gate/root-gate-entry（引擎 `resource_manager.h` `inspect()`：per prefix-index 条目门控词汇表，`selected=none` 或选中候选走 root（`reuse_tokens=0`）时输出，见 §6）与 root-diag（serve 层，常开，21-token 窗口）。

**偏离设计的定案**：

1. finish() fork-abort 分支**保留**（原设计"删除"）——基线 v1 防御代码，删除会使零 token 轮 + pending fork 端点永久不可消费 + destination 泄漏；常规路径恒不触发（§1.1）。
2. S-V3-4 落点改 `src/serve/generation_service.cpp`（原设计写 `apps/serve/main.cpp`）。
3. (a2) 用自定义异常 `SourceUnavailableException` + 执行主循环 catch 特判（而非直接 inline abort），保证 staging 释放/ack 完整性走 `abort_transaction()` 现成路径。
4. `publish_checkpoint_drop` 一致性检查与 populate 同步改为发布前沿比较（文档未单独列出，M1 的必然配套：V3 按边界 key 发布、按同 key drop，比较对象必须一致）。

**编译/测试状态**：docker（`ninfer-local-build:latest`，无 GPU）首轮构建暴露 trace 命名空间/成员名问题，修复后目标全量编译、ctest 全绿；4090 两轮实跑后，S-V3-4 键 bug 修复（conversation_key）+ root-diag/root-gate trace 追加（本轮）：首轮 docker 构建发现 root-gate 行的 `prompt.summary()` 对测试 `FakePreparedPrompt` 不可用（改 `base.summary().prompt_tokens`），修复后 **BUILD_EXIT=0 + 100% 104 tests, 0 failed**（31 执行全过 / 73 GPU skip 属预期；细节与前序"25/79"误计的更正见 `DOING.md` 执行日志）。4090 同步编译与修复后重跑实跑待执行（root-gate/root-diag 组合判读方法见 §6 点位表与 HANDOFF §2.4）。

---

## 附录 A.1：对照锚点表（行号 = `d181dfd0` 树，仅 v2 对照）

| 锚点 | 位置（file:line） | v2 终态作用 | V3 对应 |
|---|---|---|---|
| `populate_continuation_summary` | `program_impl.h:7452-7521` | endpoint 分支 min 前沿（:7464-7468）+ rebuild_work 选择（:7474-7479）+ 尾部守卫 `private continuation has no checkpoint`（:7511-7512） | 保留 min 逻辑；守卫在 V3 天然通过（finish 的 populate 必有 endpoint） |
| D3 赋值 + 方案 B 守卫 | `program_impl.h:8662-8680` | publish 前双点赋值（:8665-8666 `sequence.endpoint_frontier = prefill.prompt_tokens`）+ `transaction.publishes_checkpoint \|\| sequence_has_checkpoints` populate 门控（:8672-8676） | **删除**（无 mid-prefill 边界发布）；改为 finish() 单点赋值 |
| D7c drop 守卫 | `program_impl.h:4693-4694` | `source.endpoint_valid && source.execution_frontier > details.reuse_base && details.reuse != ReusePath::PrivateEndpoint`；`release_if_unreferenced`（:4679-4686） | **保留**（M2）+ 专门测试 |
| `finish()` | `program_impl.h:9607-9664` | fork-abort（:9613-9620，实跑从未触发）/ freeze（:9631-9635）/ `endpoint_valid=true`（:9636）/ populate（:9643）/ finish trace（:9645-9663） | 保留 freeze/endpoint_valid/populate；**删除 fork-abort 分支**（常规路径无 fork）；加 `endpoint_frontier` 单点赋值 |
| `SequenceState` | `program.h:405-434` | `execution_frontier`（:414）、`endpoint_valid`（:427）、`endpoint_frontier`（:428）；**无 `prompt_tokens`**（它住在 `RequestControl::Prefill`，`program.h:480`） | 加 `endpoint_frontier` + `prompt_boundary` 两字段 |
| 状态图语义 | `state_image_store.h:180-459` | begin/commit/abort_fork、freeze、release、`move_checkpoint_to_active`（:366-384） | 不变（V3 常规路径只用 freeze + move） |
| 引擎侧 | `resource_manager.h` | `ActiveCaptureRecord.publishes_checkpoint`（3 个 record 构造点）+ settle 校验/合并门控 + `rebuild_prefix_index(trigger)` | **零改动**（V3 不新增任何引擎侧概念） |

## 附录 A.2：prefix-v3 @`4f0faf0c` 基线锚点表（实测行号；实现时以 grep 复核）

| 锚点 | 位置（prefix-v3 基线） |
|---|---|
| `SequenceState` 字段组 | `program.h:405-434`（`endpoint_valid`:427；V3 在 ~:427 后加 `endpoint_frontier`+`prompt_boundary`） |
| `RequestControl::Prefill.prompt_tokens` | `program.h:479`（`reuse`/`base`/`cursor` 同结构 :470-486） |
| `MaterializationTransaction` | `program.h:820` |
| 关键声明 | `start_sequence` :1024 / `selected_state` :1052 / `checkpoint_summary` :1088 / `settle_state_fork` :1200 |
| `finish()` | `program_impl.h:9117-9220`（fork_pending 块 :9173-9180；freeze :9191；`endpoint_valid=true` :9195；populate :9199） |
| `populate_continuation_summary` | `program_impl.h:7264-7333`（endpoint 分支 :7275 起；frontier :7280；endpoint_work :7282；尾守卫 `private continuation has no checkpoint` :7321） |
| `prepare_consumed_source` | `program_impl.h:4611+`（slot 检查 :4624-4629 throw `materialization source changed before dependency release`；**M2 drop 分支 :4650** `if (source.endpoint_valid && source.execution_frontier > details.reuse_base)`；long-anchor :4659-4674；rewrite :4667-4683；TruncateTarget 尾裁剪 :4694+） |
| `selected_state`/`consumed_references`/`requires_fork` | `program_impl.h:6868/6897/6928`（PrivateEndpoint 陈旧 throw `private endpoint StateImage is stale`） |
| `start_request` | `program_impl.h:7113-7179`（entitlement 校验 :7171-7177 `materialized sequence does not match its active entitlement`——基线为严格相等，v2 `2c26c74f` 的放宽不在基线） |
| `start_sequence` | `program_impl.h:9447+`（staging 校验 :9450-9457【V3 prompt_boundary 写入点】；Root :9465-9530；preserving :9477-9560；consume :9531-9553；尾裁剪 :9567-9675；prefix fork/activation :9676-9762；`activate_consumed_state` :9697-9741；**PrivateEndpoint 分支 :9771-9827**（role 检查 :9772-9776 throw `resident endpoint StateImage is not movable`）；rewrite-restore :9829-9862；分支后公共 :9873-9877 `endpoint_valid=false`【V3 同步 `endpoint_frontier=0`】） |
| `settle_state_fork` | `program_impl.h:10431-10445`（调用点 8958/10131/11526/12218-12227） |
| `valid_continuation` | `program_impl.h:6399`（noexcept；owner+generation+role==Catalogued） |
| `inspect_admission`/`revalidate`/`reserve` | `program_impl.h:1225-1310`（源 slot 检查 :1237-1239 throw `admission source continuation is stale`）/ :3990-4130（源 slot role!=Catalogued → StalePolicyState 不抛）/ :4187-4530（preflight :4190-4194） |
| 执行主循环（run） | `program_impl.h:6240-6339`（`prepare_consumed_source` :6268-6269；受控中止范式 cancel_pending→abort :6271-6277；`start_request` :6315 try/catch→release+rethrow） |
| `wrap_prefill`/`commit(PendingBatch)` | `program_impl.h:7131-7162` / :9090+（capture carrier OFFER/INCONSISTENT 检查） |
| `endpoint_valid=false` 重置点（V3 同步加 `endpoint_frontier=0`，共 7 处） | `publish_checkpoint_drop` :2800/:2820；:4652；`retire_continuation_slot` :6601/:6615；`start_sequence` :9877；`release_active_sequence_state_strict` :10518；`release_sequence_state_strict` :10588；`release_sequence_state` :10653 |
| `inspect_lane`（planner） | `request_plan_impl.h:447+`（**siH 端点校验 :538** `selected.frontier==0 \|\| selected.frontier != source->execution_frontier → throw catalog endpoint summary disagrees with Program state`【M1 改为接受 `source->endpoint_frontier`】；reuse=PrivateEndpoint :548、reuse_base=selected.frontier :549；MTP 门 :595-607；DFlash 门 :609-621；C4w 块 :658-702；`state_fork_required` :703-709） |
| `plan_request` | `request_plan_impl.h:~300-445`（V3 不加边界 CaptureGroup，与 v2 分道） |
| 引擎 `inspect`/`rebuild_prefix_index`/`valid_prefix_index_entry` | `resource_manager.h:273-405`（私有候选 `entry.state==Catalogued && handle && !private_has_active_edge(slot)` :343-346 = (a1) 引擎侧第二层）/ :1687 / :1726 |
| 引擎 `plan_materialization`/`reserve`/`adopt_materialization_progress` | `resource_manager.h:1938-2264` / :418-463（Stale :431 优雅重试通道；Aborted→rollback :448-452）/ :2797+（Aborted 终态合法 :2848-2852） |
| 引擎 `progress_context_transaction`/`try_admit_one`/`admit_planned_request`/`fail_all_locked` | `engine_core.h:1626-1795`（Aborted→`materializing_.reset()`+`complete_detached_cancelled` :1703-1713）/ :1797+（TemporarilyBlocked→留队下轮重 inspect）/ :1667-1795（Stale→留队重试）/ :2073-2093（`failed_=true` :2077 全局宕机） |
| 状态图语义 | `state_image_store.h:334+`（role : Free/ActiveMutable/CheckpointImmutable/ReservedDestination；freeze/begin/commit/abort_fork/move_checkpoint_to_active；**fork 期间 source 恒 CheckpointImmutable** → 即使 finish 保留 fork-abort 分支删除方案，role 检查仍会通过——实现最终选择保留分支，§1.1） |
| 测试/应用布局 | `tests/test_admission_policy.cpp`、`tests/test_resource_manager.cpp`、`tests/targets/qwen3_6/test_context_store.cpp`（前缀电池）、`tests/targets/qwen3_6_27b/test_engine_prefix_real.cpp`（27b 真引擎，GPU）、`src/serve/generation_service.cpp`（S-V3-4 实际落点）、`apps/cli`、`apps/perplexity` |
| trace 现状 | 基线全仓**无任何** `NINFER_ADMISSION_TRACE`/`NINFER_PREFILL_TRIPWIRE` 埋点（grep 全仓仅 PREFIX-V3.md 命中）→ V3 全部新埋（§6） |

## 附录 B：v2 的 7 条承重约定 → V3 的 2 条

| # | v2 约定（无测试保护，破了就出事） | V3 状态 |
|---|---|---|
| 1 | D3 赋值必须在 publish 内 populate 之前（重排 → key 静默退回执行前沿 → 100% key-miss 无报错） | 收敛为"finish() 内单点赋值在 populate 前" + 测试钉住 |
| 2 | `publishes_checkpoint` 3 个 record 构造点必须全贯穿 | **消失**（该概念本身不存在） |
| 3 | D7c 守卫 per-ReusePath（其余 4 路径仍 drop，实跑长期未验证） | 保留守卫 + 专门测试（断言只豁免 PrivateEndpoint） |
| 4 | 单槽池/DFlash 下原始 bug 静默复活（配置分叉） | **消失**（重锚无条件、全配置通用） |
| 5 | finish fork-abort 分支可达但从未验证 | **保留**（基线 v1 防御代码；V3 常规路径不触发，finish trace `fork_pending` 可观测） |
| 6 | entitlement / `checked_resource_difference` 单向语义（"加强"成双向相等会炸复用轮） | 不变（v2 继承约定，建议测试钉住）——基线无 `2c26c74f` 放宽，V3 首跑复用轮即验证（R-V3-2） |
| 7 | dangling 索引条目依赖被动自清 | 不变（建议测试钉住） |
