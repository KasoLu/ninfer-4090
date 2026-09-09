# HANDOFF.md — PREFIX-PLAN v2 前缀复用失效调查（当前状态快照）

> 生成时间：2026-09-09（本轮会话末尾，会话已因特殊控制符号反复异常停止）。本文件是**当前唯一权威交接**：旧版 HANDOFF.md（P0 时代，HEAD 4f0faf0c）已作废；NOTE.md 为本仓库持久工作日志（§0–§13，未跟踪），上下文被压缩后应重读本文件 + NOTE.md。
> **会话卫生警告**：后续会话的回复中**不要**复现聊天模板的控制序列文本；讨论相关 token 一律用数字 ID（如 198 / 271 / 248068 / 248069），否则模型输出会被截断。

## 1. 任务与现状一句话
PREFIX-PLAN v2（分支 prefix-v2，HEAD `0766142a`）在 4090 真实 serve 部署下**前缀复用完全失效**：每轮对话/工具回报都全量 re-prefill（`prefix_reuse_path=root, prefix_cache_hit_tokens=0`）。已确诊到 digest 链分歧的精确位置与形态；修复方案待按 §6 定案实施。相同客户端在 prefix-v1 分支上复用正常 → 客户端侧已排除（用户原话裁决）。

## 2. 环境
- 本地仓库：`C:\Workspace\codes\ninfer-4090-kaso`（分支 prefix-v2）。
- 本地编译/ctest：`docker run --rm -v C:\Workspace\codes\ninfer-4090-kaso:/src -w /src/build ninfer-local-build:latest sh -c "ninja && ctest --output-on-failure"`（无 GPU；GPU 测试本地 skip；期望 ninja 32–33/33 + ctest 100% 104/104）。
- 4090 远程：`py -3 scripts/remote_4090.py run "<cmd>" inf`（**cmd 语法直传，勿嵌套 powershell/管道**；同步代码用 `git merge --ff-only origin/prefix-v2`——`git pull --ff-only` 在该仓库报 "Cannot fast-forward to multiple branches" 歧义）。
- 4090 仓库：`C:\Data\ninfer\ninfer-4090-kaso`；devel 容器 `66e132d9aa6a`（ninfer-4090-kaso-devel，镜像 `ninfer-4090-kaso-devel:0905`，bind 挂载 + `/models` 卷，仅含 `qwen3_8_27b.ninfer` 18.2GB）。
- 4090 编译：`docker run --rm -v C:\Data\ninfer\ninfer-4090-kaso:/ninfer-4090-kaso -w /ninfer-4090-kaso/build ninfer-4090-kaso-devel:0905 ninja`（当前增量基线 43/43）。
- 4090 复现 serve（需先请用户停常驻推理容器，GPU 被占即 nvidia-smi 显存 >20GB）：
  `NINFER_ADMISSION_TRACE=1 build/apps/ninfer-serve /models/qwen3_8_27b.ninfer --preserve-thinking --chat-template v22_4 --spec mtp --draft-tokens 3 --lm-head-draft --kv-dtype rk8v4 --max-context 200000 --request-log-jsonl logs/log.txt`
- serve 引擎配置（实测自 startup 日志）：max_concurrency=1、kv_capacity=200000（3125 page groups）、prefix_reuse=true、prefill_chunk=1024；context_cache：device_state_slots=1、host_state_slots=0、host_kv_bytes=0、max_shared_prefixes=1、max_private_continuations=2、automatic_private_anchors=2、**total_device_state_slots=2**（device 池 1 + shared 池 1）。
- DSH 约束：file policy=danger-full-access；approval prompts **禁用**（不得设 sandbox_permissions）；文件修改走 PS1/write 范式（.tmp-prefix-plan/ + `pwsh -NoProfile -File`，单引号 here-string，CRLF 保持，UTF8 no-BOM，锚点断言 count==1）；edit 工具历史上易退化（小范围可用，退化立即切 write）；中间结论落盘 NOTE.md。

## 3. 提交链（全部已 push origin/prefix-v2 = KasoLu/ninfer-4090；4090 已同步并编译）
```
4f0faf0c 基线
→ 2c26c74f, 809537ff, bbb41e3b, a066cb15   （P0/P1/P2-A 实现）
→ 3a6b7d06  fix(P2-A): root state slot reservation capacity-adaptive
→ 83f60ca9  chore(debug): NINFER_ADMISSION_TRACE gates（埋点第一层）
→ e341e68f  fix(P1): hard protection UAF + M3 window exit + committed-record exclusion
→ 1300f493  fix(P1-M3): release queued hard-protection record on admission
→ ca208f76  chore(debug): trace prefix-index key-miss
→ 0766142a  chore(debug): prefix digest divergence trace（= 当前 HEAD = 4090 二进制）
```
已修复并验证的准入死锁链（勿回归）：UAF（protected_owners span 悬垂）→ M3 需求窗口只进不出 → 已终结请求的 committed 记录误参与硬保护 → admit 成功后 pending 记录永留自保护。四层修复后 4090 上**无死锁**、请求全部正常完成；唯一剩余缺陷 = §4 的复用失效。

## 4. 当前缺陷：前缀复用全失效（已确诊至 digest 分歧形态）
### 4.1 现象（logs/log.txt + logs/admission_serve_trace.log，2026-09-09 22:01–22:05，0766142a 二进制，单会话多轮 + tool history + thinking）
- 7 个请求全部 `prefix_reuse_path=root, prefix_cache_hit_tokens=0` → 每轮全量 re-prefill（prompt 58712/60328/61605/…，ttft 最高 37s）。
- 每轮 admission：私有索引恰 1 个 occupied 条目 = 上一轮 retained root 的 SessionEndpoint（kind=0），随后 **key-miss**；identity_tag 两侧一致（327937），排除 spec/draft/kv_dtype 记账差异 → root 全量重算 + 驱逐上一轮 owner（private_owners_evicted=1/轮）。
- 全程 **captures=0、forks/moves/restores=0、anchors=0**（automatic_private_anchors=2 从未生效）。

### 4.2 决定性分歧数据（divergence trace，0766142a 埋点输出；st@ 行 token_type 恒 0、三轴位置恒=index、rope_delta=0）
| 轮次 | stored_size | incoming_size | first（首个分歧 frontier） | stored 侧 [first-4..first] token ID 序列 |
|---|---|---|---|---|
| req2 | 58822 | 60328 | 58743 | 13, 198, 248069, 271, 97625 |
| req3 | 60734 | 61605 | 60574 | 13, 198, 248069, 271, 98428 |
| req4 | 63401 | 14751（新会话，incoming 远短于 stored）| 9186 | 198, 248068, 198, 760, 1156 |
| req5 | 15728 | 15931 | 15451 | 13, 198, 248069, 271, 100875 |
| req6 | 16251 | 16469 | 16024 | 13, 198, 248069, 271, 63 |
| req7 | 16698 | 16945 | 16504 | 13, 198, 248069, 271, 111708 |

（stored_size = 上轮 prompt+completion，如 req1: 58712+110=58822 ✓，ledger 记账自洽。）

**形态解读**：
- req2/3/5/6/7（同会话续轮）：分歧点 = stored 链中第一个**控制 token 簇**（13=换行；198/248068/248069/271 = 本模板的角色/边界类特殊 token；簇后紧跟正常内容 token）。簇位于**生成段前部**（req2 中约为生成偏移 30，即 thinking 段边界标记附近）。簇之前两侧链逐 token 完全一致（prompt 段 + 生成段前缀吻合），簇处 stored=原始特殊 token ID、incoming=客户端从解析后文本重新序列化产生的不同 token。
- req4（新会话）：incoming 表（14751）远短于 stored（63401），first=9186 处 stored 为模板边界簇 → 新会话 prompt 与 retained 旧会话链自然分歧，**属预期行为**（跨会话本不该复用）。
- 结论：**不是 v2 的记账 bug**（token/位置/token_type 三输入与 incoming 侧口径一致；v1/v2 的 digest 代码无 diff），而是**存储链把“原始生成 token 流（含模型在生成途中吐出的模板控制 token）”原样纳入 digest，而客户端回放的是“解析后文本的重序列化”**，在第一个控制 token 位置必然分叉。

### 4.3 为什么 v1 能复用而 v2 不能（核心未决问题）
- 本部署两条复用通道在 v2 下全部死亡：
  1. **Endpoint 通道**：key 打在生成段末尾 frontier（execution_frontier），受 §4.2 分叉影响 → 恒 miss。
  2. **Anchor/TurnClosure 通道**：`anchors=0` 恒成立 —— total 状态槽=2，retained root 占 1 + 运行中 root 占 1 = 池满，锚点 capture 目的地槽永远无空闲（state_image_store reserve_destination：device 池 free=0 且 host 池=0 → nullopt）→ 自动锚点结构性不可行。
- v1 在同客户端/同部署下复用正常（用户已验证）。**未核实**：v1 当时靠哪条通道命中（v1 恒 root state_slots=2 的槽位算术与本配置不同；v1 部署的 context_cache 参数是否与本 serve 一致未查）——若 v1 也走 endpoint，则需 diff v1 的 populate_continuation_summary 调用点与存储 frontier 口径（§5 代码锚）。
- 用户裁决记录（原话）："客户端肯定是没问题的，相同的客户端，已经在prefix-v1分支上验证过，可以直接排除客户端侧的原因。问题肯定出在推理端。"

### 4.4 修复候选（按性价比排序，尚未实施）
1. **Endpoint key 前移至“最后一个 prompt 边界”frontier**（推荐先做）：客户端重放时 prompt 段逐 token 字节一致（digest 输入同口径，已实证前缀全匹配），只有生成段受控制 token 重序列化影响 → 把 continuation summary 的存储 key frontier 取为上一轮 prompt 结束处（= 上轮 prompt_tokens，digest 表天然覆盖该前缀），或**同时**双索引（prompt 边界 + 末尾 frontier）。复用损失仅为生成段回放的少量 re-prefill（本 run 约 110–1800 token / 58k+）。
2. **锚点通道恢复**：让 P2-A 的第二状态槽（预置 capture destination）容量感知化（池紧张时 root 退回 1 槽），使 automatic_private_anchors 能在 total=2 配置下真正建立 prompt 边界锚点 —— 与 1 互补：1 不依赖槽位，2 依赖槽位。
3. 生成段控制 token 的 digest 归一（模板边界 token 从生成段 digest 输入中排除/归一）——改动面大、语义风险高，列最后选项。

## 5. 关键代码锚（行号 = 0766142a）
- key 存储：`src/targets/qwen3_6/impl/runtime/program_impl.h` `populate_continuation_summary`（~:7270-7340 区，v1 同区逐字节相同）：stored key = `sequence.prefix_digests.at(execution_frontier)` + identity_tag；`finish()`（:9293-9370）publish_continuation 时 endpoint_valid=true → Catalogued → 引擎侧 rebuild_prefix_index 入私有索引。
- digest 算法：`src/targets/qwen3_6/impl/runtime/prefix_identity.{h,cpp}`（v1/v2 无 diff）：`append_digest(token, token_type, positions[3], rewrite_frontiers)`；prompt 侧 `assign()`、生成侧 `PrefixShortlistDigests::append_generated(span, rope_delta)`（token_type=0，position=index+rope_delta，本部署恒 0）；`at(frontier)` = digests_[frontier]（覆盖前 frontier 个 token 的链 digest）。
- key 比对门：`src/runtime/engine/resource_manager.h` inspect 候选循环（~:295-460）：`base.prefix_shortlist_key(index.key.frontier)` 必须与 index.key 精确相等，否则 key-miss（0766142a 在此挂 divergence 扫描）；`prefix_shortlist_key`：`src/targets/qwen3_6/impl/runtime/api_impl.h:111-122`（frontier>size → nullopt → 候选静默跳过）。
- 锚点/capture：`src/targets/qwen3_6/impl/runtime/state_image_store.h`（device_capacity() = 池总 device 槽数；reserve_destination：with_device 且 free_device_count==0 → nullopt）；P2-A 预置槽消费：`program_impl.h` terminal capture `else if (sequence.reserved_state) { destination_state = *sequence.reserved_state; sequence.reserved_state.reset(); }`（~:8159-8162）；P2-A 槽数决策：`src/targets/qwen3_6/impl/runtime/request_plan_impl.h:370` 区 `root_active.state_slots = (state_store && state_store->device_capacity() >= 2U) ? 2U : 1U;`。
- 埋点（commit 83f60ca9/ca208f76/0766142a，全部 NINFER_ADMISSION_TRACE=1 门控）：engine_core.h 抛错前 IDLE-BLOCK；resource_manager.h inspect 三处 TB + T1 索引 dump/T2 index-invalid/T3 key-miss；materialization_planner.h root 路径（status/owners/outcomes/protected + PRUNED + goal-null）；program_impl.h inspect_admission 候选明细 + `debug_trace_prefix_divergence`（LCP 首处分歧 + st@ token 行；声明链 runtime.h → api_impl.h → program.h → program_impl.h 实现于 valid_capture_offer 前 ~:6452 区；测试假件 FakeProgram 有 no-op stub）。
- 测试：`tests/test_resource_manager.cpp`（FakeProgram/FakeShortlistKey 假件须与引擎新增 API 同步）；`tests/targets/qwen3_6_27b/test_engine_prefix_real.cpp`（GPU 电池；本部署配置组合 = 同会话多轮 + device1/host0/shared1 + MTP + preserve_thinking 是覆盖盲区，测试通过但实跑失效）。

## 6. 下一步（按序）
1. **定案 §4.4-1**：endpoint/continuation 存储 key 的 frontier 前移至最后 prompt 边界（可双索引末尾 frontier 兜底）。先纸面设计（“prompt 结束”= prepared prompt 的 token 数 = 上轮 prompt_tokens，生成段之前的边界，digest 表天然覆盖；注意 MTP 与 thinking 段不影响该边界取值，因边界在生成段之前），再 PS1 实施。
2. 本地 docker build+ctest 全绿 → commit/push → 4090 `git merge --ff-only origin/prefix-v2` + ninja → 请用户停常驻容器后按 §2 serve 命令重跑 2-3 轮（带 NINFER_ADMISSION_TRACE=1）→ 预期：key-miss 消失（或仅残留在末尾 frontier 的次要条目），`prefix_cache_hit_tokens` > 0（≈上轮 prompt_tokens），re-prefill 仅剩生成段。
3. 若仍有 miss → 实施 §4.4-2（P2-A 容量感知，恢复锚点通道）。
4. 验证通过后：决定三层 trace 门控（83f60ca9/ca208f76/0766142a）去留（建议保留门控）；补 prefix_real 测试电池盲区 scenario（同会话多轮 + 本部署状态槽配置）。
5. 收尾：NOTE.md 增记新 §14（根因定案 + 修复 + 验证）；P2 剩余 GPU 验证项（serve 长生成 / stall / M5 strict settle / 双协议 / 25 轮 M2 / re-prefill=0 / 崩溃恢复）。

## 7. 踩坑备忘（后续会话直接照抄）
- `git pull --ff-only` 在 4090 仓库报 upstream 歧义 → 一律 `git merge --ff-only origin/prefix-v2`。
- remote_4090.py 传 cmd 语法，勿嵌套 `powershell -Command`（'inf' 会被拆成 timeout 参数报 ValueError）；拉远程文件用主机侧 `powershell -NoProfile -Command "Get-Content ... -Tail N"` 且避免引号嵌套。
- PS1 here-string @'...'@ 是单字符串：取下标须先 `$t -split "`r?`n"` 成数组，否则拆成逐字符行（曾把 resource_manager.h 打坏）；插入 = 重建数组 `Lines[0..i] + block + Lines[i+1..]`；连续多处插入先小索引；PS1 行正则不用 $ 尾锚（CRLF）。
- FakeShortlistKey 无 identity_tag → 打印用 `if constexpr (requires { x.identity_tag; })`；FakeProgram 缺新方法 → 加 no-op stub；注意 `base.summary` 是方法 `summary()` 非成员。
- 4090 GPU 常驻容器（近期为 admiring_blackburn 类命名，端口 1234，占满显存）需用户自行停；nvidia-smi 显存 >20GB 占用即被占。
- 会话卫生：回复与文档中不要复现聊天模板控制序列（用数字 token ID 代替），否则模型输出会被截断（本会话已因此多次异常停止）。
- 本地 ctest 基线：104 项，GPU 相关本地 skip（无权重/无 CUDA），ninfer_resource_manager_test 为前缀准入核心单测（含 pending demand 窗口/硬保护/terminal release 回归）。
