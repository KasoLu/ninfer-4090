# RK6V4E8 — 新增 KV 量化模式 `rk6v4-e8` 设计文档

> 基线：`KVMODE.md`（本仓库现有全部 KV 量化方案的完整梳理）。
> 目标：在 `KvCacheStorage` 中新增 **`RK6V4E8`（显示名 `rk6v4-e8`）**：
> **K = 6-bit 码 + E8 格投影 + H64 旋转；V = 4-bit 码 + H64 旋转**，
> 容量档介于 `rk8v4`(400 B) 与 `rk4v4`(272 B) 之间：**336 B/token/kv_head**。

---

## 1. 设计决策与理由

### 1.1 为什么 K 用真 6-bit 位打包（而不是"int8 平面装 E8 投影"）
- "复用 int8 平面 + E8 投影把动态范围压到 ~6-bit" 不省容量：K 平面仍是 256 B，
  总容量 = rk8v4 的 400 B，容量档没有意义。
- 真 6-bit：K 平面 `256 dim × 6 bit = 192 B`（`head_dim × 3/4`），总 336 B，
  比 rk8v4 省 16%，比 rk4v4 多 23%（换来 K 从 4-bit → 6-bit 的精度）。
- 6-bit 不整字节对齐，但 **4 code 恰好 = 3 字节（24 bit）**，天然对齐 3 字节；
  16-dim 块 = 12 B（3×u32，4B 对齐）、64-dim 组 = 48 B（3×16B），向量化读写都成立。

### 1.2 为什么 K 侧沿用 "E8 格投影 + 整数码 + mma_s8" 读路径
- 现网所有 packed/E8 模式 K 侧读回都还原成 **int8 进 smem 走 `mma_s8` 张量核**
  （`prompt_i8.cuh` / `small_t_i8.cuh` 的 K 分支）。6-bit 码 unpack 成 int8（±32..31）
  后完全复用这条路径，**QK 的 mma/softmax/scale 折乘全部不动**，只换 K 的 staging 分支。
- E8 投影（`e8_project_8d_warp`，`src/ops/kernel/e8_lattice.cuh`）是**尺度无关**的：
  rk4v4-e8 在 `absmax/7` 域投影，rk6v4-e8 在 `absmax/31` 域投影，同一套 8-lane 子组
  shuffle 原语、同样的 D8/D8+0.5 双陪集最近格点 + `rintf` 塌缩（半陪集近似与
  rk4v4-e8 一致，整数域最终结果与是否保留 coset bit 无关——见 `e8_lattice.cuh` 的
  `e8_project_8d_warp_single`：输出只取最近格点坐标，后续 `rintf` 到整数）。
- `E8 ⊇ Z^8`：E8 最近格点量化**不差于**逐维整数舍入，在 D8+0.5 半陪集上有增益；
  6-bit 只是把整数码值域从 ±7 放宽到 ±32/31，格结构不变。

### 1.3 为什么 V 不动
- V 侧所有 rk* 模式都是 i4（±7）+ H64 旋转，`rk6v4-e8` 的 "v4" 与 rk8v4/rk4v4 完全相同；
  改动集中在 K 侧，V4 的 append/staging/逆旋转输出路径零改动。

### 1.4 flag 组合
```
packed_v=true  rotate_k=true  rotate_v=true
packed_k=true  e8_lattice=true     // 复用：K 打包 + E8 投影
kv_k6_bit=true（新增 flag，唯一区分位）  e8_root=false
```
`packed_k`/`e8_lattice` 组合已存在（= rk4v4-e8），**无法从现有 6 flag 反推 6-bit**，
因此必须新增第 7 个 flag `kv_k6_bit`（贯穿 §4 所有结构体与序列化 bitfield）。

---

## 2. 字节布局与容量

head_dim=256、I8 家族 quant_group=64（4 组）、scale=FP16、页大小 64（PageMajor）。

| 平面 | dtype | 每 (token, kv_head) 字节 | 说明 |
|---|---|---:|---|
| k_code | **U8** | **192** | 256 个 6-bit 码，4 code/3B，64 组×… = 4 组×48 B |
| v_code | U8 | 128 | 256 个 i4 码，2/字节（同 rk8v4） |
| k_scale | FP16 | 8 | 4 组 × `fp16(absmax_rotK/31)` |
| v_scale | FP16 | 8 | 4 组 × `fp16(absmax_rotV/7)` |
| **合计** | | **336** | |

容量排序（B/token/kv_head）：`bf16 1024 > int8 528 ≈ fp8 516 > rk8v4 400 >`
**`rk6v4-e8 336`** `> rk4v4 = rk4v4-e8 272 > rk2v4-e8 208`；bf16 的 1/3.04。

对齐核算（`decoder_state.cpp:plan_cache` 的 `KVPlaneGeometry{dtype, LE, HE, 256}`）：
- K 平面页内头步长 = `64 × 192 = 12288 = 256 × 48` ✓（页/头边界保持 256 对齐）；
  token 步长 192 B 无需对齐（现有 `packed_k` token 步长 128 B 同样非 256 倍数）。
- 页 stride（kv_heads=4）：K 49152 + V 32768 + 2×scale 8192，均 256 倍数 ✓；kv_heads=2 同理。
  **页池/对齐逻辑零改动。**

### 2.1 6-bit 码定义与打包（新 codec 原语，落 `src/ops/kv_cache/int8_g64_codec.cuh`）
- 码值：`value ∈ [-32, +31]`（满 6-bit 域，镜像 i4 现有非对称满域约定
  `max(-8, min(7, q))`，见 append kernel `p1W`/`foX`）；存储码 `u6 = value + 32 ∈ [0, 63]`。
- **打包**（4 code → 3 字节，小端 24-bit 字 `w = c0 | c1<<6 | c2<<12 | c3<<18`）：
  ```
  b0 = w & 0xFF
  b1 = (w >> 8) & 0xFF
  b2 = (w >> 16) & 0xFF
  ```
  quad j（dim 4j..4j+3）位于该 head 行字节 `3j`；64-dim 组 = 48 B；head 行 = 192 B。
- **解包**：`c0 = w&0x3F; c1 = (w>>6)&0x3F; c2 = (w>>12)&0x3F; c3 = (w>>18)&0x3F;`
  `value = (c ^ 32) - 32` → int8（`mma_s8` 操作数，同 i4 unpack 的 `nib^8-8` 约定）。
- 新增 API（与 :69/:94/:160 的 i4 原语并列）：
  ```cpp
  // 量化：x 已乘 kinv；clamp [-32,+31] → +32
  __device__ std::uint8_t kv_cache_i6_quant_code(float x, float inv_scale);
  __device__ void kv_cache_pack_i6_quad(const std::uint8_t c[4], std::uint8_t* out3); // 24-bit 字
  __device__ std::int8_t  kv_cache_unpack_i6(std::uint8_t code);          // (code^32)-32
  __device__ void kv_cache_unpack_i6x16(const std::uint8_t* src12, std::int8_t* dst16); // 3×u32 读
  // 地址：LE=192（d 须为 16 的倍数 → 字节偏移 d*3/4 为 4 的倍数）
  template <typename Geometry>
  __device__ std::int64_t kv_cache_i6_code_index(int physical_page, int kv_head,
                                                 int d, int page_off);
  //   = paged_kv_page_head_offset<192, Geometry::KVHeads>(physical_page, kv_head)
  //   + 192 * page_off + (d * 3) / 4
  ```

### 2.2 编码数值合同（与现有模式同构）
```
K: ksh = fp16( warp_absmax(H64(K)) / 31 )            // 旋转域，FP16 RNE 边界
   x   = H64(K) / ksh_represented                     // kinv = 1/ksh_fp16
   x   = e8_project_8d_warp(x)                        // 8-lane 子组，原语不变
   q   = rintf(x);  code = clamp(q, -32, +31) + 32
V: vsh = fp16( warp_absmax(H64(V)) / 7 );  code = clamp(rintf(H64(V)*vinv), ±7)+8（不变）
scale 平面：lane0 写 ksh/vsh（FP16），per-64 组，4 组 —— 与全部 i8 系相同
```
溢出域：E8 投影最坏把码推到 `absmax/31 × ~1.0 + 0.5` ≈ 32（half-coset 时），
`[-32,+31]` 满域 clamp 恰好兜住（镜像 i4 的 `[-8,+7]`）。
QK 溢出检查：`|q_i8|≤127, |k_i8|≤32 → 256×127×32 ≈ 1.04e6 < INT32_MAX` ✓。

---

## 3. 模式注册：枚举 → flags → 布局

| 位置 | 改动 |
|---|---|
| `include/ninfer/types.h:31` | 枚举**末尾**追加 `RK6V4E8`（保持既有 ordinal 不动） |
| `src/targets/qwen3_6/impl/runtime/layouts.h:78-85, 109-116` | `SequencePlanningInputs`/`SequencePlanningOutputs` 增 `bool kv_k6_bit = false;` |
| `src/targets/qwen3_6/impl/runtime/layouts_impl.h:783-798` | `.kv_k6_bit = options.kv_cache == KvCacheStorage::RK6V4E8`；
  `:144-151`（→`DecoderStateSpec`）、`:697-704`（core→plan）同步加字段 |
| `src/targets/qwen3_6/export/ninfer/targets/qwen3_6/decoder_state.h:21-28` | `DecoderStateSpec` 增 `bool kv_k6_bit = false;`（quant_group 常量 :12-13 不变） |
| `src/targets/qwen3_6/impl/state/decoder_state.cpp` `plan_cache`（`930` 起） | ① 签名/layout 结构增 `k6_bit`；
  ② extent：`k_head_extent = k6_bit ? head_dim*3/4 : e8_root ? head_dim/4 : (packed_k ? head_dim/2 : head_dim)`
  （K6 蕴含 packed_k，分支必须放在 packed_k **之前**）；`k_plane_dtype = U8`（经 packed_k 已成立，保持显式）；
  ③ 校验：`k6_bit` 必须同时满足 `packed_k && e8_lattice && packed_v && rotate_k && rotate_v` 且 `!e8_root`，否则 `std::invalid_argument`（与 :85-92 现有三条约束并列）；
  ④ `PagedKVCacheLayout`/`PagedKVCache` 成员、`layer_view`/`batch_layer_view`（`Fxf`/`K4F` 起）透传 `k6_bit` |
| `src/targets/qwen3_6/impl/runtime/program.h:633-638` + `program_impl.h:748-749` | 成员/初始化透传 |
| `src/targets/qwen3_6/impl/runtime/program_impl.h:11530`（memory_summary 反向映射） | 在 `e8_lattice → RK4V4E8` **之前**插入 `kv_k6_bit → RK6V4E8` |
| `src/targets/qwen3_6/impl/runtime/session_snapshot_impl.h:429-432, 630-632` | `kv_flags` bitfield 新增 `kKvFlagK6Bit`（POD uint32，旧快照缺位=0，向后兼容） |
| `src/core/paged_kv_cache.h`（`PagedKVLayerView`/`PagedKVBatchLayerView`） | 视图结构体增 `bool k6_bit = false;`（现有 6 bool 之后） |

---

## 4. 编码路径改动（`src/ops/kv_cache/append/`）

`kernel.cuh` 两个 kernel（`kv_cache_append_full_i8_kernel` `S2O` 起 / `_page_kernel` `C14` 起）
模板追加第 7 参 `bool K6 = false`（默认 false，既有 6 参实例全部不受影响）：

1. scale 行（`tud`/`H9w`）：
   `k_abs / (K6 ? 31.0f : ((PackedK || E8Root) ? 7.0f : 127.0f))`
2. K 分支优先序变为 `E8Root → K6 → PackedK → 直写`。新 `K6` 分支（镜像 `PackedK+E8Lattice`
   分支 `MTE`-`XOH` / `SbK`-`sN0`）：
   ```cpp
   } else if constexpr (K6) {
       float k0_s = k0 * kinv, k1_s = k1 * kinv;
       e8_project_8d_warp(k0_s, k1_s, lane);      // 半陪集塌缩注记同 rk4v4-e8
       std::uint8_t c0 = std::uint8_t(max(-32, min(31, int(rintf(k0_s)))) + 32);
       std::uint8_t c1 = std::uint8_t(max(-32, min(31, int(rintf(k1_s)))) + 32);
       // 位打包：4 连续 lane = 1 quad(3B)；8-lane 块 leader(lane&7)==0 打包 32 码=24B(d0 半)，
       // 同法再打 d1 半（偏移 +24）。shfl_down 1/2/3/4/5/6/7 取同块其余 lane 的 c0/c1：
       //   16-dim 块 3 个对齐 u32（12B）；块内偏移 12*(lane/8)，d1 半 +24
       // 组基址 = paged_kv_page_head_offset<192,HE>(page, kv_head) + 192*page_off + group*48
       // V 侧与 PackedV 分支完全相同（pack_i4 写 kv_cache_i4_code_index）
   }
   ```
   尾部 lane（shfl_down 溢出回自身）只影响非写者 lane，写者（`lane&7==0`）的
   `lane+1..+7` 均在本 warp 同块内，无跨界风险（与现有 `__shfl_down_sync(...,1)`
   配对 i4 的技巧一致）。
3. `launch.cu:RTL-yOp`（I8 段 `Dj1` 起）分发新增一支（在 `e8_lattice` 之前）：
   ```cpp
   } else if (cache.k6_bit) {
       launch_fill.template operator()<true, true, true, true, true, false, true>();
       // (PackedV, RotateK, RotateV, PackedK, E8Lattice, E8Root, K6)
   ```
   `tokens>=32` 走 `_page_kernel` 的既有调度（grid `(max_tiles, KVHeads, 4 组)`）不变。
   前缀 append（`kv_cache_append_prefix_*`，纯 BF16 拷贝）不受影响。

---

## 5. 解码路径改动（causal attention）

Q 侧（int8 ±127 + 可选 H64）、V 侧（i4 → fp16）、softmax、sm_89 fp16-acc 折叠、
`rotate_v` 逆 H64 输出（`prompt.cu` 尾部 `kv_cache_inverse_rotate_output_kernel`）**全部不变**。
只改 K 的 staging/写分支：

### 5.1 `src/ops/softmax_attention/dense/causal_cache/prompt_i8.cuh`
kernel 模板增 `bool K6`（`S2O` 的 6 参模板之后）；`issue_kv_tile` 的 K 分支
（`KMg`-`HUK`）在 `PackedK` 支前插入：
```cpp
} else if constexpr (K6) {
    // 16-dim 块 = 12B（3×u32 对齐读，偏移 d*3/4）→ unpack 16 个 int8 进 smem
    const std::int64_t koff = kv_cache_i6_code_index<Geometry>(physical_page, kv_head, d, key_l);
    kv_cache_unpack_i6x16(&reinterpret_cast<const std::uint8_t*>(cache_k)[koff], kd);
    // V 与 PackedV 支相同（unpack_i4x16）
}
```
smem K tile 仍是 `64×256` int8（`kCausalPromptI8KBytes`），`mma_s8`/swizzle 不变。

### 5.2 `src/ops/softmax_attention/dense/causal_cache/small_t_i8.cu/.cuh`
- kernel 模板（`small_t_i8.cuh:60-61`）增 `K6` 参；
- **读分支**（`420-453`）：`E8Root` 支后插 `K6` 支，同 5.1 的 `kv_cache_unpack_i6x16`
  写 `k_i8[key_l*D + causal_small_t_tc_swz(...)]`；
- **写分支**（decode 时把新 token 的 K/V 码写回缓存，`235-310`）：`PackedK` 支前插 `K6` 支，
  复用与 append kernel §4 相同的 6-bit 打包（可把 §4 的打包/码函数抽为共享
  `__forceinline__` 设备函数，append 与 small_t 共用，避免两份位运算代码漂移）；
- `small_t.cu:260-314` `NINFER_CAUSAL_SMALL_T_DISPATCH`：`e8_root` 支后插
  `cache.k6_bit → launch_tc_partial_i8<Geometry, T, true,true,true, true, true,false, true, MB, Masked>`。

### 5.3 `prompt.cu` 分发（`IYu` 起）
`cache.e8_root` 支后插 `else if (cache.k6_bit) launch_i8.template operator()<true, true, true, true, true, false, true>();`
（`cache.dtype == DType::I8` 已满足——RK6V4E8 归 I8 家族，见 `layouts_impl.h:69`
`target_kv_cache_profile` 需把 `RK6V4E8 → {DType::I8, kKvInt8QuantGroup}` 并入 I8 分支）。

其它 attention op（context/packed/sliding_window）继续不消费量化 paged 缓存；
`small_t.cu:242-254` 的 dtype 断言（仅 `BF16/I8/FP8_E4M3FN`）不受影响。

---

## 6. 入口/名称/文档/测试接线

| 位置 | 改动 |
|---|---|
| `apps/cli/options.cpp:56-62`（parse）、`apps/cli/main.cpp:97-109`（`format_kv_cache`）、`options.h` | `rk6v4-e8` ↔ `RK6V4E8`；help 串 `options.cpp:86` 增字 |
| `src/serve/serve_options.cpp:50-56`（parse）、`src/serve/request_log.cpp:134-146`（`kv_cache_name`） | 同上；help `serve_options.cpp:85` |
| `bench/targets/qwen3_6_27b/ninfer_bench_support.cpp:51-68, 862-876`（parse + `memory.kv_cache` 名）、bench main help | 同上 |
| `apps/perplexity/main.cpp:102-132` | 同上 |
| 文档 | `docs/serving.md:734`（`--kv-dtype` 表）、`docs/cli.md:203`、`README.md:316/382`、`docs/udp-fork-comparison.md` 追加容量档说明 |
| 测试 | `tests/test_serve_options.cpp`（`rk6v4-e8 → RK6V4E8`）、`tests/test_ninfer_bench_support.cpp`、`tests/test_kv_cache.cpp`（RK6V4E8 布局断言：k 平面 U8 LE=192 / v U8 LE=128 / scale FP16 LE=4；页 stride 256 对齐）、`tests/targets/*/test_engine_*_real.cpp` 加 rk6v4-e8 用例 |
| CMake | 无需改（kernel 为模板，实例由分发选择） |

---

## 7. 数值验证与质量评估（按 AGENTS.md 精度合同）

1. **codec 位精确微基准**（`tools/test_kv/`，仿 `test_e8_codec.cu/.cuh`）：
   新增 `E8Packed6BitTile{ uint8_t codes[64][48]; half scales[64][4]; }`
   （128B 对齐），设备 codec（§2.1 原语 + `e8_project_8d_warp`）对 CPU 参考实现
   **逐位一致**；覆盖：4-code 打包/解包往返、满域 ±32/31、half-coset 样本、
   `rintf` 银行舍入边界（x.5 → 偶数）。
2. **独立 FP64 oracle**（`kv_oracle.py` 扩展）：append+attention 链从公共 BF16 输入出发，
   K̂ = Σᵢ `unpack_i6(codeᵢ) × ksh_fp16`（精确 code+scale 解码，不复制 kernel 的
   E8 投影路径——E8 投影用参考实现），V̂ 同 i4；FP64 算完整 attention 公式；
   整条测试集单一容差（与 i8 系现有容差口径一致）。
3. **长程检索**：`tools/test_kv/verify_1m_retrieval.cu` 在 `rk6v4-e8` 下跑 1M 可见 key 检索。
4. **质量/容量实测**（4090）：`tools/bench/run_rk8v4_quality.py` 的 `KV_MODES` 增
   `"rk6v4-e8"`（int8 / rk8v4 / rk6v4-e8 三档对比 1k 语料 PPL + 检索命中率）；
   C1 自动 sizing 的 capacity 数字对照（336 B 档应落在 rk8v4 与 rk4v4 之间）。
   预期：K 侧 6-bit E8 的精度应介于 rk4v4-e8（4-bit E8）与 rk8v4（8-bit）之间且
   靠近 rk8v4；V 与 rk8v4 完全一致。
5. **回归**：既有 7 模式全部测试保持绿（新 flag 默认 false，模板默认参不变 →
   既有实例的生成代码逐位不变）。

### 7.6 4090 实测（2026-09-04，sm_89 修复后的客观评测）

前提：K6 在 sm_89 的 4 点 `__shfl_down_sync` 死锁修复完成并已推送（`8f05de52`，见 `HANDOFF_RK6V4E8.md`），`--kv-dtype rk6v4-e8` 全链路可用。

**1. Codec 级余弦**（合成高斯 64d，4096 组，H64 旋转，生产 codec 算术）

| 模式 | K mean/min/p99 | V mean/min/p99 |
|---|---|---|
| rk8v4 | 0.249998 / 0.0 / 0.999990 | 0.248577 / 0.0 / 0.996585 |
| rk6v4-e8 | 0.249830 / 0.0 / 0.999681 | 同 K |

合成高斯下是保守下界（i8 也仅 0.25），只看相对序：K 侧 rk6v4-e8 与 rk8v4 持平（0.2498 vs 0.2500）。

**2. 端到端客观评测**（`tools/bench/run_kv_precision_suite.py`，greedy、无 MTP，4090 实跑）

| 指标 | bf16 (1024B) | rk8v4 (400B) | rk6v4-e8 (336B) |
|---|---|---|---|
| T0 确定性（两遍自洽） | Y | Y | Y |
| 12 题数值真值短答 | 11/12 | 11/12 | 11/12 |
| 短答与 bf16 逐题一致率 | — | 1.000 | 1.000 |
| 长生成字符数（1024 new） | 3980 | 4043 | 4128 |
| 长文词级 LCS / 词频（参考项） | 1.000 / 1.000 | 0.524 / 0.693 | 0.213 / 0.499 |

- 三档唯一共同失分题："840 的最大质因数"答 21（质因数/因数混淆，属模型推理错误，与量化无关）。
- 长生成代码（离线动态连通性完整实现）抽查三档代码质量均正常。
- 指标口径注记：词级 LCS/词频衡量的是"与参考稿的文本重合"，不是正确性——greedy 下首个分叉 token 之后两档都正确但词重合自然低，该指标仅作旁证，不进结论。

**结论**：K6 档（3.05× KV 压缩，336 vs 1024 B/token/head）在所有带真值的客观指标上与 bf16 无差异（短答逐题一致），长生成正常——rk6v4-e8 无客观退化。

**评测缺口（后续补）**：
- Perplexity（NLL）是最客观的单标量，但 `apps/perplexity` 目前仅支持 `--kv-dtype bf16|int8|fp8`，需扩展到 rk 族后在 `eval/corpora/perplexity-1m` 1k 语料上补三档 PPL。
- 检索门（needle-in-a-haystack / code-detail，README 的既有验收口径）尚未在 rk6v4-e8 下跑过。
- 12 题短答样本量小且已达模型自身错误上限（三档同题同错），后续可扩题或换代码执行类验证。

---

### 7.7 PPL 异常根因：`kv_cache_unpack_i6x16` 字节错位（2026-09 已修复）

`--kv-dtype` 白名单扩到全档后，quick PPL（261167 scored tokens，`eval/corpora/perplexity-1m`）实测：

| 模式 | overall PPL | vs bf16 |
|---|---|---|
| bf16 | 4.3431 | — |
| rk8v4 | 4.3460 | +0.067% |
| rk4v4-e8 | 4.3614 | +0.421% |
| rk6v4-e8 | 4.5222 | **+4.124%**（分域：zh 5.0046→5.1866，en_long 6.8924→7.1136，en_ref 6.1943→6.5719，code 1.6527→1.7119） |

rk6v4-e8 比 4-bit 差 10×，明显偏离“6-bit 应介于 4-bit 与 bf16 之间”的理论序。

**根因**（`src/ops/kv_cache/int8_g64_codec.cuh:220` 原行）：
`q2 = ((b1 >> 16) & 0xffffu) | (((b2 >> 24) & 0xffu) << 16)` —— quad2 的高字节取 `b2` 的 byte11（属于 quad3），
正确应为 `b2` 的 byte0（即 `b2 & 0xffu`）。后果：每个 16-dim 块的 dim 10 错取 quad3 低 6 位、
dim 11 错取 quad3 高 2 位，K 平面 12.5% 维度被污染。写侧（append full/page kernel + small_t 写支的
`kv_cache_pack_i6_quad` + `__shfl_sync` quad gather）、布局（`kKVCacheI6HeadExtent=192` 与
`decoder_state.cpp` 的 `k6_bit ? head_dim*3/4` 一致）、prompt 读侧偏移全部核对无误——仅此一处字节错位。

**为何旧评测没抓到**：`test_kv6_cosine` 用 bench 头内联参考解码（不复用生产 unpack），合成高斯余弦下
每 dim 1/32 的污染被高斯分布稀释；12 题短答样本小。PPL 是全 token 级 NLL，放大后显形。

**修复**：一行 `| ((b2 & 0xffu) << 16)`；并在 `tools/test_kv/test_kv6_cosine.cu` 加硬门禁回归
`kv6_roundtrip_check_kernel`（生产 `kv_cache_pack_i6_quad`→`kv_cache_unpack_i6x16` 往返，16×64 全码值全位置
扫 + 4096 随机块，任何一位不一致 exit 1），防止再犯。

**验证**（4090，修后重跑）：`ninfer_test_kv6_cosine` 回归绿；quick PPL 三档复测（bf16/rk8v4/rk4v4-e8
不受影响，预期不变；rk6v4-e8 应落回 4.34–4.37 区间，与 rk4v4-e8 同量级）。

---

## 8. 边界与已知取舍

- **head_dim=256 硬依赖**：i8 系全部 kernel 断言 256/64 几何（`kKVCacheAppendFullHeadDim=256`、
  `kCausalPromptI8Groups==4`）；rk6v4-e8 与现有量化模式同域，不引入新依赖。
- **位打包跨 lane 协作**：打包靠 `__shfl_down_sync` 块内取码，写者是 8-lane 块 leader；
  与 `e8_root`/`packed_k` 分支同一 warp 协作纪律——**任何 early-return 必须发生在
  全部 shuffle 之后**（参照 `e8_root_codec.cuh` 的 rad_idx 注记），否则污染 code 平面。
- **半陪集塌缩**：与 rk4v4-e8 相同的刻意近似（D8+0.5 陪集经 `rintf` 塌缩进整数域、
  不重建）；对"存整数码"的 E8 模式，最终整数与保留 coset bit 与否无关（见 §1.2）。
- **快照兼容**：`kv_flags` 是 uint32 bitfield，新增位对旧快照天然为 0 →
  旧 session 快照不会误解析成 rk6v4-e8。
- **host 层 / 页拷贝**：prefix-reuse 的 device↔host 页拷贝与 slot 持久化是
  字节级拷贝，对编码格式透明，零改动。
- **MTP/DFlash**：MTP 层与主层共用 `DecoderStateSpec`（`plan_cache` 两次调用），
  自动同模式；前缀 append 保持 BF16 拷贝不变。

---

## 9. 实施顺序（建议）

1. `types.h` 枚举 + `layouts.h`/`decoder_state.h`/`program.h` 的 `kv_k6_bit` 字段 +
   `layouts_impl.h` 映射 + `session_snapshot_impl.h` bit + `paged_kv_cache.h` 视图字段。
2. `int8_g64_codec.cuh` 新增 i6 原语（pack/unpack/index）+ CPU 参考（oracle 用）。
3. `decoder_state.cpp` `plan_cache`（extent `head_dim*3/4`、U8 平面、新校验、透传）。
4. append：`kernel.cuh` K6 分支（+`launch.cu` 分发）；跑 §7.1 位精确微基准。
5. small_t：`small_t_i8.cuh` 读/写分支 + `small_t.cu` 分发宏。
6. prompt：`prompt_i8.cuh` staging 分支 + `prompt.cu` 分发。
7. 全链路 parse/名称/help/文档/单测（§6）。
8. 质量与容量实测（§7.3-7.4），4090 上跑三档对比后定稿。

> 提交仅在用户要求时创建（Conventional Commit，如 `feat(kv-cache): add rk6v4-e8 (6-bit E8 key + 4-bit value)`）。
