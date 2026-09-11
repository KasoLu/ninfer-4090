# RK6V4E8 — `rk6v4-e8` 量化模式实现最小性评估

> 评估对象：`rk6v4e8` 分支（基于共同基线 `5474313b`）上 6 个 KV 特性提交
> `a8e5716c..69a5f87b`（feat → sm_89 稳定 → 测试 → 修复 → bench → docs）所实现的
> `KvCacheStorage::RK6V4E8`（显示名 `rk6v4-e8`）。
> 评估问题：**该实现是否为"改动最小、影响面最小、实现最简单"的方式**。
> 评估参考：`KVMODE.md`（现有全部 KV 模式的管线梳理，含新增模式必须同步的 §7 接线清单）、
> 分支内 `RK6V4E8.md`（v1 设计文档）、`HANDOFF_RK6V4E8.md`（sm_89 卡死事件记录）。
> 结论先行见 §6。本文服务于把该模式重新落到 `rk6v4e8-v2` 分支的决策。

---

## 0. 被评估的模式是什么

`rk6v4-e8`：K 侧 **真 6-bit 码**（value ∈ [-32, +31]，per-64 组 FP16 scale `fp16(absmax/31)`）
+ **E8 Conway-Sloane 格投影**（复用 `e8_project_8d_warp`，尺度无关）+ H64 旋转；
V 侧 4-bit（±7）+ H64，与 rk8v4/rk4v4 完全相同。
存储：K 平面 U8、head_extent=192（4 个 6-bit 码打包进 24-bit 字 = 3 字节；16-dim 块 = 12 B = 3×u32）；
V 平面 U8 128；k/v scale 各 8（4 组 × FP16）。
**容量档 336 B/token/kv_head**，介于 rk8v4(400) 与 rk4v4(272) 之间。
唯一新增模式位：第 7 个 flag `kv_k6_bit`（快照 bitfield `kKvFlagK6Bit = 1U << 6`，旧快照缺位 = 0，天然兼容）。

生产代码改动面（`a8e5716c^..69a5f87b`，src/ + include/ + apps/ + bench/）约 **550 行新增 / 73 行删除、26 个文件**；
其余 43 文件中的 tools（1437 行：PPL/needle/余弦/ktrace 验证工具）、3 个根文档（834 行）、测试（158 行）属验证侧，不计入生产影响面。

## 1. 备选方案对比：为什么"真 6-bit 打包"是该目标下的最小解

"rk6v4e8" 的产品目标是一个**容量档**（336 B，bf16 的 1/3.04），不是"同一容量下的精度变体"。
在这个目标约束下枚举可行编码：

| 方案 | K 存储 | 容量档 | 相对改动量 | 判定 |
|---|---|---|---|---|
| **A. 真 6-bit（4 码/3B 打包）+ E8 投影 + 读回 int8 走 `mma_s8`（实际采用）** | U8，192 B/head | **336 B ✓** | 新 i6 codec（~62 行）+ 第 7 flag + append/small_t/prompt 各一个 K 分支 | **达成目标的最小解**（§1.2 说明其近乎唯一） |
| B. int8 平面（extent 256）+ E8 投影把码压到 ±31 | I8，256 B/head | 400 B（= rk8v4，**无新档**） | 最小：无新 codec、无 U8 平面、无读路径改动、无 12B 非对齐寻址；但需放宽 `e8_lattice && !packed_k` 校验 | 改动确实更小，但产出的是"rk8v4 的精度变体"而非新容量档——**目标不同时它才是最小解**（v1 设计文档 §1.1 的论据成立） |
| C. 4-bit + 2-bit 残差双平面 | i4 128 B + i2 64 B | 336 B（密度同 A） | 两个 code 平面 + 两个 scale 平面：布局/校验/append/读路径/dtype 校验全部翻倍，且解码是两级还原 | 密度相同、复杂度严格更高，无收益 |
| D. E8 根族（rk2v4-e8 思路）+ 更高精度半径 | 240 根 = 8-bit 根码已超 6-bit；取 64 根子集则破坏根几何与 vadd4 查表 | ≥ 336 B（8-bit/8-d → 256 B） | 新码表、新解码路径、读路径无法复用 `mma_s8` | 不可行/更差 |
| E. 2 码/1.5B 打包（比 A 更细的粒度） | 同 6-bit | 336 B | 字节偏移不再 4 对齐，向量读退化 | 无收益的变体；4 码/3B 是 6-bit 唯一 ≥3B 对齐的打包单元 |

**1.1** 6-bit 不整字节对齐是硬事实：任何"真 6-bit 存储"都必须引入跨码位打包，A 的 4 码/24-bit 字是
最小对齐单元（3 字节），16-dim 块 12 B 保持 4B 对齐——codec 没有"更简单"的同密度替代。

**1.2** 读路径选型是关键的最小化点：K 码 unpack 回 **int8（±32..31 仍是 int8 值域）** 后，
QK 的 `mma_s8`/swizzle/softmax/scale 折乘**整条不动**，只换 K 的 staging 分支
（`prompt_i8.cuh` / `small_t_i8.cuh` 各一个 `else if constexpr (K6)`）。这正是现网所有
packed/E8 模式（rk4v4、rk4v4-e8、rk2v4-e8）的既有惯例——存储更密、读回还原 int8 进张量核。
V 侧、Q 侧、逆 H64 输出、页池对齐、前缀 append（纯 BF16 拷贝）、host 层字节拷贝、MTP/DFlash
（共用 `DecoderStateSpec`，`plan_cache` 自动同模式）**零改动**。

**1.3** 因此：*给定"336 B 容量档 + K 有效 6-bit 精度"这一目标*，方案 A 在编码语义上近乎唯一，
在工程实现上复用了它能复用的全部既有原语（E8 投影、H64、i4 V 路径、`mma_s8` 读路径、页几何、
快照 bitfield 扩展位）。它不是"所有可能做法里改动最少的"（B 更少），但它是**满足目标的最小实现**；
B 的更小是以放弃目标为代价的。

## 2. 改动面核查：逐点对照 `KVMODE.md §7` 接线清单

| 必改点 | 实际改动 | 核查 |
|---|---|---|
| `include/ninfer/types.h` 枚举末尾追加 | +`RK6V4E8`（1 行，ordinal 不动） | ✓ 最小 |
| 枚举→(DType,group) 与枚举→flags 映射（`layouts_impl.h`） | `target_kv_cache_profile` 并入 I8 分支（1 行）；flags 推导**收敛为新表** `src/core/kv_cache_mode.h::flags_for()`（+50 行新文件） | ✓ 见 §2.1，属合理收敛 |
| `decoder_state.cpp::plan_cache` extent/校验/透传 | `k_head_extent = k6_bit ? head_dim*3/4 : ...`（分支置于 packed_k 之前）；新校验 `k6_bit ⇒ packed_k && e8_lattice && packed_v && rotate_k && rotate_v && !e8_root`；U8 平面；`PagedKVCacheLayout`/视图透传 | ✓ 最小 |
| `kv_flags` 序列化（`session_snapshot_impl.h`） | 新增 `1U<<6` 位，save/restore 对称 | ✓ 最小且向后兼容（旧快照该位 = 0） |
| 第 7 flag 贯穿结构体 | `layouts.h`（Inputs/Plan）、`decoder_state.h`（Spec/Layout/PagedKVCache）、`program.h`/`program_impl.h`（成员 + 构造 + `memory_summary` 反向映射最前插 `k6_bit`）、`paged_kv_cache.h`（两个视图结构体）——共 7 处、每处 1 行 | ✓ 与当年加 `e8_lattice`/`e8_root` 时完全同构，属"新模式的固有成本"，无法更少 |
| append 编码（`src/ops/kv_cache/append/`） | 两个 kernel（full/page）模板 +`K6` 参（默认 false，既有实例逐位不变）；K6 分支（E8 投影→rintf→i6 码→4-lane shuffle gather→leader 写 3B×2 半区）；`launch.cu` 分发在 `e8_lattice` 之前插一支；`kv_cache_append.cpp` 校验 +U8/192 分支 | ✓ 见 §3 复杂度注记 |
| attention 解码（`src/ops/softmax_attention/dense/causal_cache/`） | `prompt_i8.cuh` staging K 分支（`kv_cache_unpack_i6x16` 12B 对齐读→int8 smem）；`small_t_i8.cuh` 读支 + 写支（decode 新 token 也按 i6 回写）；`prompt.cu`/`small_t.cu` 分发各插一支；`causal_softmax_attention.cpp` 两个 validate 函数 | ✓ 最小；smem K tile 仍 64×256 int8，`mma_s8`/swizzle 不变 |
| CLI/serve/bench/perplexity parse+名称+help | 各 1~2 行；`apps/perplexity/main.cpp` 额外把 `--kv-dtype` 从 3 档扩到全 7 档（+26 行，见 §2.2） | ✓ 基本最小，一处合理回填 |
| 文档/测试 | `docs/cli.md`、`docs/serving.md`、`README.md` 各 1~2 行；`tests/test_serve_options.cpp`、`tests/test_ninfer_bench_support.cpp`、`tests/test_kv_cache.cpp`（新增 `{U8,192,2,256}+{U8,128,2,256}+{FP16,4,2,256}×2` 的 PageMajorK6 几何用例）、新增 `tests/test_kv_mode.cpp`（121 行，纯 host） | ✓ |
| CMake | 仅新增测试/工具 target，生产 kernel 是模板、实例由分发选择，无需改 | ✓ |

**结论：生产改动面 == 该模式的依赖锥，无锥外改动。** 每一项必改点都只做了"1 行/1 分支"级别的最小修改。

### 2.1 清单外的合理增加：`src/core/kv_cache_mode.h` 共享 flag 表

v1 把原先散落在 `layouts_impl.h` 的 6 组 `options.kv_cache == X || ... == Y || ...` 布尔式
收敛成一张 `flags_for(KvCacheStorage)` 表（`src/core/kv_cache_mode.h`，host-only，50 行），
并配套 `tests/test_kv_mode.cpp`：无 default 的 switch 使任何新增枚举值漏表都变成 `-Wswitch` 构建错误，
测试再逐模式断言 flag 组合与 `plan_cache`/校验器不变量。这不是"最小 diff"的选择（直接在 7 个布尔式
各加一个析取项 diff 更小），但它是**被现实验证过的**：v1 开发中首次实现确实漏了 RK6V4E8 的
`e8_lattice` 项（`HANDOFF_RK6V4E8.md` 记录的 `bcecacf2` 提交即修此漏项）——散式布尔写法使这种漏项
静默发生，集中表 + 回归测试把它变成构建期可捕获。净成本 50 行，收益是下一个模式（若再新增）的
改动面从"7 处布尔式 + 校验 + 测试"降为"表里 1 行 + 测试断言"。**判定：合理，保留。**

### 2.2 清单外的合理回填：`apps/perplexity` 全档 `--kv-dtype`

perplexity 工具原先只接受 `bf16|int8|fp8`，bench 工具早已全档。本次把 perplexity 补齐到全 7 档
（+26 行），使新档的 PPL 验收（§7.7 的 +4.124% 字节错位 bug 就是靠全档 PPL 暴露的）成为可能。
属验证能力回填，代价小；副作用是该工具 CLI 面变宽（原先报错的写法现在合法），文档未单列，可接受。

## 3. 固有复杂度点与两处"不最小"的残留

### 3.1 固有：6-bit 跨 lane 打包（写路径）

每 warp 持一个 64-d 组、每 lane 2 维（`d0=group*64+lane`, `d1=d0+32`）的既有 lane 契约下，
4 个 6-bit 码横跨 4 个连续 lane，打包必须跨 lane 收集。i4 模式早有同类需求（`__shfl_down_sync(FullMask,·,1)`
配对打包），K6 是它向 4 码/3B 的推广——**复杂度种类不新，幅度增大**。开发中的真实代价：
初版用**非 uniform 子掩码** `sub_mask = 0xFF << (lane & 0x1FC)` + `shfl_down(o=1..3)`，
非 leader lane 越出自身子掩码读源 = PTX UB，在 sm_89（4090）上两次 warp 死锁
（`HANDOFF_RK6V4E8.md` §2：ktrace v3 锁定自旋者为 `kv_cache_append_full_i8_page_kernel` K6 写支）。
最终形态改为**全掩码显式源 gather** `__shfl_sync(FullMask, c, qbase+m)`（m=0..3，`qbase = lane & ~3`）
——代码库唯一长期安全使用的 shuffle 形态，对 leader 读位精确等价，sm_89 实测通过。
**判定：跨 lane gather 是 6-bit 位密度的固有成本，不是设计失误；最终形态正确。** 但注意两点：
- 该 ~40 行 gather 块在 **3 处复制粘贴**（`append/kernel.cuh` 的 full kernel 与 page kernel、
  `small_t_i8.cuh` 写支）。v1 设计文档 §5.2 自己建议"抽为共享 `__forceinline__` 设备函数，
  append 与 small_t 共用，避免两份位运算代码漂移"——实现没有照做。sm_89 修复当时就必须在 3 处同步落盘
  （handoff 文档的"要改的 3 处"清单即其代价），漂移风险是真实的。**这是"实现最简单"维度上最实质的减分项。**
- 读路径无此问题：`kv_cache_unpack_i6x16` 一次读 12 B（3×u32 对齐）同步展开，无任何 warp 协作。

### 3.2 残留：`NINFER_K6_BISECT` 死锁排查脚手架仍在生产热路径

`src/ops/softmax_attention/dense/causal_cache/small_t.cu` 的 `k6_bisect_arm()`（`getenv` 读
`NINFER_K6_BISECT`，0..4 五臂）+ `launch_tc_partial_i8` 签名多出的 `std::int32_t k6_bisect` 参数
（**所有** i8 实例、所有分发臂都传它）+ `small_t_i8.cuh` 内核里 `k6_bisect == 1/2/3` 的
"用已验证的 i4 路径替换 i6 读/写" 分支 + 分发处 `k6_bisect == 4 && cache.k6_bit → k6_bit = false`
+ 仅为它而加的 `#include <cstdio>/<cstdlib>`。handoff 文档自评：**"方向搞错、无效但保留"**。
死锁根因修复（全掩码 gather）落盘后，这套脚手架已无任何用途，却永久留在引擎最热的
decode attention 分发与 kernel 里：多一个运行期内核参数、若干热路径分支、一处 getenv，
且任何机器上若设置了该环境变量，生产行为会被静默改写。**这是与"改动最小、影响面最小"
直接冲突的残留，落 v2 前必须整体删除**（sm_89 回归由 `tools/test_kv/test_kv6_cosine.cu` 的
pack/unpack 往返硬门禁 + needle 检索门覆盖，不依赖 bisect 臂）。

### 3.3 微小瑕疵

`small_t_i8.cuh` K6 分支后有两行尾随空白；`kv_cache_i6_code_index` 依赖"d 为 16 的倍数"的
调用方纪律（以注释约束，无静态断言）——均可接受，非阻塞。

## 4. 影响面评估（对既有 7 模式与运行时的副作用）

- **既有模式逐位不变**：所有新逻辑都在 `else if constexpr (K6)` / `else if (cache.k6_bit)` 支内，
  模板新参默认 `false`；`kv_k6_bit` 默认 `false`；`memory_summary` 反向映射仅在 `k6_bit` 为真时
  改变结果。既有 7 模式的实例化代码、平面几何、kernel 二进制路径不变。
- **序列化/快照**：`kv_flags` 新位 `1U<<6`；旧快照读入时该位为 0，`restore_continuation` 的
  `expected_flags` 比对对旧引擎写出的快照依然通过；新快照对旧引擎不可读（新增枚举 ordinal 在
  `Fp8E4M3Row256` 之后追加，`kv_dtype` 字段值域扩大）——与当年加 RK2V4E8/RK4V4E8 时的兼容性
  语义一致，属项目自有格式、不做向后兼容约定的范围内。
- **性能侧**：K6 的 prefill/decode K 侧 unpack 每 16-dim 块读 12 B（i4 为 8 B），QK/PV 张量核
  路径不变；实测（v1 分支 4090 数据）full PPL 1.04M token 下 rk6v4-e8 相对 bf16 为 +0.130%
  （rk8v4 +0.101%、rk4v4-e8 +0.372%），吞吐与 bf16 齐平（1046–1056 tok/s）——量化开销未见
  性能回归；检索门（single/5-needle/code-detail，含满配 262144 单针门）全过。
- **文档/工具**：3 个根文档（`KVMODE.md`/`RK6V4E8.md`/`HANDOFF_RK6V4E8.md`）与 tools/ 脚本
  只增不改；`.gitattributes` +6 行（eval 语料行尾固定）。

## 5. 过程质量佐证（为什么这些复杂度是值得付的）

v1 的两个真实缺陷都出现在 6-bit 打包这一"新"的部分，且都被本分支自建的客观门禁抓出：
1. `kv_cache_unpack_i6x16` 字节错位（quad2 高字节误取 quad3 的 byte11 → K 平面 12.5% 维度污染，
   PPL +4.124%）——合成高斯余弦与 12 题短答均无法暴露，全 token PPL 才显形；修复为一行
   `| ((b2 & 0xffu) << 16)` 并加了 16×64 全码值往返硬门禁（`test_kv6_cosine.cu`）。
2. sm_89 非 uniform 子掩码 shuffle 死锁（§3.1）。
若按"绝对最小 diff"的思路省掉 PPL 套件与余弦/needle 工具（-1437 行 tools），这两类缺陷
都没有可靠的暴露手段——工具侧的"不省"正是生产侧"最小"的保险。**验证侧投入与生产侧最小化
是同一枚硬币的两面，评估时不应把 tools/ 的体量算作实现的"改动不最小"。**

## 6. 结论

1. **编码模式层面：是（在目标约束下）。** 真 6-bit（4 码/3B）+ E8 格投影 + 读回 int8 复用
   `mma_s8`，是达成"336 B 容量档"这一目标的**本质唯一的最小方案**：E8 投影原语尺度无关、直接复用；
   H64、V 路径、页池、前缀 append、host 层、MTP/DFlash 零改动；6-bit 位密度决定了跨 lane 打包
   不可避免，而 4 码/24-bit 字是其最小对齐打包单元。"改动更少"的 B 方案（int8 平面装 E8 投影码）
   只有同时放弃容量档目标才成立——它产出的是 rk8v4 的精度变体，不是新档。
2. **接线方式层面：是。** 生产改动面与该模式的依赖锥完全重合（约 550+/73- 行、26 文件），
   每一项必改点均为 1 行/1 分支级最小修改；7 处 flag 贯穿与当年 e8 系模式的添加完全同构，
   是该代码库 flag 驱动架构下"新增一个存储编码"的固有成本；`kv_cache_mode.h` 共享表是唯一
   超出裸清单的改动，且已被真实漏项 bug 验证有价值。
3. **两处减分项（落 v2 前处理）：**
   - **必须删**：`NINFER_K6_BISECT` 死锁排查脚手架（`small_t.cu` 的 `k6_bisect_arm()` 与分发处
     臂 4、`small_t_i8.cuh` 的臂 1/2/3 分支、`launch_tc_partial_i8` 的 `k6_bisect` 参数及所有
     分发臂的透传、`<cstdio>/<cstdlib>` 包含）——无效调试残留留在生产 decode 热路径，与"影响面
     最小"直接冲突。
   - **建议做**：把 3 处复制的 4-lane gather/pack 块抽成共享 `__forceinline__` 设备函数
     （append 两个 kernel 与 small_t 写支共用），兑现设计文档 §5.2 的原始建议，消除漂移风险。
4. **数值/性能结论（v1 实测，4090/sm_89）支持该模式值得落 v2**：full 1.04M-token PPL
   `bf16 4.6504 < rk8v4 4.6551 < rk6v4-e8 4.6564 < rk4v4-e8 4.6677`（严格随 K 侧 bit 数单调，
   K6 比 rk8v4 仅 +0.028%）；三类检索门全过（含满配 262144 单针门，24 GB 卡上唯一能满上下文
   且通过最深单针门的档位）；12 题短答三档逐题一致。

## 7. 落到 `rk6v4e8-v2` 的操作要点

- v2 与 v1 共享基线 `5474313b`，且 v2 自基线以来**未动任何 KV 族文件**（仅 `types.h` 的 host 层
  默认值等）——6 个 KV 提交可整体 cherry-pick，非 KV 提交（jinja 模板、tool-call salvage、AGENTS
  合并、部署基线）**不要**带入。
- 预期冲突点：`include/ninfer/types.h`（v2 已含 host 默认值改动，KV 侧仅追加 1 行枚举，区域相邻，
  手工合并即可）；`src/targets/qwen3_6/impl/runtime/program_impl.h`（v2 的 prefix-v3 `dc6c9f24`
  同区改动）；`tests/CMakeLists.txt`。
- cherry-pick 后**先做 §6 第 3 条的两项清理**再构建验证：dev 容器语法门禁 → 完整编译 → CPU 侧 ctest
  （`ninfer_kv_mode_test`、`ninfer_kv_cache_test`、`test_serve_options`、`test_ninfer_bench_support`
  等，GPU 项照例跳过）→ 4090 上 `ninfer_test_kv6_cosine` 往返门禁 + 4 档 PPL/needle 回归
  （需先请用户关闭常驻推理容器释放显存）。
- 验证口径沿用 v1 既有门禁即可，无需新造：codec 位精确往返、quick+full PPL 四档对照、
  三类检索门（`tools/bench/make_needle_probes.py` 探针）。
## 8. v2 落地与验证结果（2026-09-05，本分支）

### 8.1 移植范围

v2 与 v1 共享基线 `5474313b`，v2 侧已含 v1 的测试与目标文件；本次在 v2 上落地的生产改动共
9 个文件（均为 §2 清单内或 §6.3 清理项），另新增 1 个头文件：

- `src/core/kv_trace.h`（新）：`NINFER_KV_TRACE` 追踪开关，见 §8.3。
- `src/core/kv_cache_mode.h`：新增 `dispatch_path_name()`（路径名优先级 e8-root > k6 >
  e8-lattice > packed-k > packed-v > int8，与三个分发点的实选臂一致），供全部 trace 点复用。
- `src/ops/kv_cache/int8_g64_codec.cuh`：新增共享设备助手 `kv_cache_lanes_write_i6_quad`
  （4-lane 显式源 full-mask gather + 24-bit quad 写，d0 写 `row`、d1 写 `row+24`；注释保留
  sm_89 死锁机理说明）。
- `src/ops/kv_cache/append/kernel.cuh`：full/page 两个 kernel 的 K6 写臂内联 gather 块改为调用
  共享助手（-50 行重复）。
- `src/ops/softmax_attention/dense/causal_cache/small_t_i8.cuh`：kernel 签名去 `k6_bisect` 参数；
  写臂/读臂的 4 臂二分 if/else 收敛为生产 i6 直路（-58 行）。
- `src/ops/softmax_attention/dense/causal_cache/small_t.cu`：`launch_tc_partial_i8` 签名与 kernel
  调用去参；删 `k6_bisect_arm()` 与分发处臂 4；删 `<cstdio>/<cstdlib>`；6 处分发臂透参移除；
  新增 decode.dispatch trace。
- `src/ops/softmax_attention/dense/causal_cache/prompt.cu`：新增 prompt.dispatch trace。
- `src/ops/kv_cache/append/launch.cu`：新增 append.dispatch trace（fill kernel 选择依赖
  `tokens>=32`，故 key 含 tokens）。
- `src/targets/qwen3_6/impl/state/decoder_state.cpp`：`plan_cache` 内循环不变的 k/v extent 与
  平面 dtype 提升出 layer 循环（v1 小瑕疵顺手清掉）；新增 plan_cache trace。
- `src/targets/qwen3_6/impl/runtime/session_snapshot_impl.h`：新增 snapshot.save /
  snapshot.restore trace（kv_flags 保存与恢复校验两处）。

### 8.2 两处减分项的处理

1. **`NINFER_K6_BISECT` 脚手架（必删项）**：已全部移除——`k6_bisect_arm()`、分发臂 4、kernel
   参数、写/读臂的 1/2/3 臂分支、`<cstdio>/<cstdlib>` 包含。`grep -r 'k6_bisect|K6_BISECT'` 在
   `src/` 与 `tests/` 均为 0 命中。生产 decode 热路径恢复单一路径，与影响面最小一致。
2. **4-lane gather 去重（建议项）**：4 处内联拷贝（append 两个 kernel + small_t 写臂）收敛为
   `int8_g64_codec.cuh` 中唯一的 `kv_cache_lanes_write_i6_quad`；位精确语义与 v1 生产路径逐位
   一致（显式源 full-mask `__shfl_sync` gather，非 `__shfl_down_sync` 子掩码形态）。
### 8.3 关键点位 trace（`NINFER_KV_TRACE`）

`src/core/kv_trace.h`：环境变量 `NINFER_KV_TRACE` 非空时启用，输出单行
`[kv-trace] <point>: k=v ...` 到 stderr；未启用时各点位仅一次 `static` 分支判断（零成本）。
`kv_trace_once(key)` 按 64-bit 配置键去重（缓存 128 键、满则清空），长会话下每配置最多一行。
点位（6 个）：

| 点位 | 文件 | 输出 | 触发键 |
|---|---|---|---|
| `plan_cache` | decoder_state.cpp | layers/capacity/kv_heads/head_dim/dtype/k_extent/v_extent/mode | 仅当任一模式 flag 置位；layers/capacity/模式位 |
| `append.dispatch` | append/launch.cu | kv_heads/tokens/path | KVHeads/tokens/模式位 |
| `prompt.dispatch` | prompt.cu | q_heads/tokens/path | QHeads/模式位（模板实例化只取决于模式） |
| `decode.dispatch` | small_t.cu | q_heads/width/path/splits | QHeads/width/模式位 |
| `snapshot.save` | session_snapshot_impl.h | kv_flags/kv_dtype/quant_group | kv_flags 非零时 |
| `snapshot.restore` | session_snapshot_impl.h | kv_flags/expected/kv_dtype | expected_flags 非零时（校验前打印，便于定位失配） |

`path` 一律取自 `kv_cache_mode::dispatch_path_name()`，与 kernel 实选臂同名。用法：
`NINFER_KV_TRACE=1 ninfer-serve <artifact> ...`（CLI/bench 同理），stderr 过滤 `[kv-trace]` 即可核对：
启动期应看到一条 `plan_cache`（模式非默认时），随后首个 append / prompt / decode 分发各一行；
会话快照保存/恢复各一行（`snapshot.restore` 的 `expected` 与 `kv_flags` 不等即失配，
紧随其后的 `invalid_argument` 会抛出）。

### 8.4 验证状态

- dev 容器（`ninfer-local-build`，CUDA 13.1.2，sm_89）：完整编译通过（apps 与测试全部链接），
  ctest 全量 106 项：**32 CPU 项通过 / 74 GPU 项跳过 / 0 失败**（CPU 项含
  `ninfer_kv_mode_test`、`ninfer_qwen3_6_state_image_layout_test`、serve/前端全组；GPU 项即 v1
  移植的 `ninfer_kv_cache_*` 系列与 `ninfer_test_kv6_cosine`、`ninfer_test_e8_codec`）。
- `git diff --check` 干净；源文件为 CRLF（Windows checkout 常态，v1 移植即如此）。
- **待 4090 机验证**（本机无 GPU）：先请用户关闭常驻推理容器释放显存，然后在 4090 开发容器
  （`ninfer-4090-kaso-dev`）跑 GPU ctest + `ninfer_test_kv6_cosine` 往返门禁 + 4 档 quick/full
  PPL 与三类检索门回归（§7 口径）。

### 8.5 结论更新

§6 的两处减分项在 v2 均已消除；`rk6v4-e8` 在 v2 上的实现即 336 B 容量档的最小影响面实现：
模式枚举 + flag 贯穿 + 三条 kernel 臂 + 共享 codec 助手，无任何调试脚手架残留，热路径零额外
分支（trace 关闭时仅一次 static 判断）。
