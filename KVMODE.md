# KVMODE — KV-Cache 量化方案梳理

> 目的：完整梳理本仓库（`ninfer-4090-kaso`，sm_89 fork）**当前已存在的各种 KV-Cache 量化方案
> 及其具体实现**，作为后续新增 `rk6v4e8` 类型的基线参考。
>
> 关键结论先说：**KV 量化模式 = 一个 `KvCacheStorage` 枚举值 + 一组布尔 feature-flag**
> （`packed_v / rotate_k / rotate_v / packed_k / e8_lattice / e8_root`）。所有"旋转 / 打包 / E8"
> 模式在存储 dtype 上都归到 **I8 家族**（`DType::I8`），打包/E8 只改变编码方式，不新增 dtype；
> `packed_k`/`e8_root` 的 K 平面用 `DType::U8` 承载 2-bit 打包或 2 字节/8-d 的 code。
> 新增一种模式 = 新增一个枚举值 + 定义它的 flag 组合 + 打通（布局、append 编码、attention 解码、
> 各入口的 name/parse）这几条管线。

---

## 1. 模式总表

| 显示名 (`--kv-dtype`) | `KvCacheStorage` 枚举 | 存储 DType | K 编码 | V 编码 | feature flags | 每 (token, kv_head) 字节* |
|---|---|---|---|---|---|---:|
| `bf16` | `BFloat16` | `BF16` | bf16（无量化） | bf16（无量化） | 全关 | 1024 |
| `int8-group64` | `Int8Group64` | `I8` | int8 ±127，per-64 组 | int8 ±127，per-64 组 | 全关 | 528 |
| `fp8-e4m3-row256` | `Fp8E4M3Row256` | `FP8_E4M3FN` | E4M3，整 256 行一组 | E4M3，整 256 行一组 | 全关 | 516 |
| `rk8v4` | `RotatedInt8KeyInt4ValueGroup64` | `I8` | int8 ±127，**H64 旋转** | int4 ±7 打包 2/字节，**H64 旋转** | `packed_v` | 400 |
| `rk4v4` | `RotatedInt4KeyInt4ValueGroup64` | `I8` | int4 ±7 打包 2/字节，**H64 旋转** | int4 ±7 打包 2/字节，**H64 旋转** | `packed_v`+`packed_k` | 272 |
| `rk4v4-e8` | `RK4V4E8` | `I8` | int4，**E8 格投影**（Conway-Sloane），H64 旋转 | int4 ±7 打包 2/字节，H64 旋转 | `packed_v`+`packed_k`+`e8_lattice` | 272 |
| `rk2v4-e8` | `RK2V4E8` | `I8`（K 用 U8 平面） | **E8 根**（240 根，2 字节/8-d） | int4 ±7 打包 2/字节 | `packed_v`+`e8_root` | 208 |

\* head_dim=256、K/V 各一份、I8 系 quant_group=64（4 组）、FP8 系 quant_group=256（1 组）、
scale 用 FP16 计算得出（见 §4/§5）。字节 = K code 平面 + V code 平面 + K scale + V scale。

### 枚举定义（`include/ninfer/types.h:31`）
```cpp
enum class KvCacheStorage : std::uint8_t {
    BFloat16,                                  // bf16
    Int8Group64,                               // int8-group64
    RotatedInt8KeyInt4ValueGroup64,            // rk8v4
    RotatedInt4KeyInt4ValueGroup64,            // rk4v4
    RK4V4E8,                                   // rk4v4-e8
    RK2V4E8,                                   // rk2v4-e8
    Fp8E4M3Row256,                             // fp8
};
```
> **新增 `rk6v4e8` 时应追加到枚举末尾**（避免改变既有 ordinal；序列化走 flag bitfield 而非 ordinal，
> 但代码里大量 `switch (KvCacheStorage)` 需同步覆盖）。

### DType（`src/core/dtype.h`）
`BF16=0, FP32=1, I32=2, U8=3, I64=4, I8=5, FP16=6, FP8_E4M3FN=7`。
注意：`U8` 用来承载 packed（2 i4/字节）与 e8_root（2 字节/8-d）的 K/V code 平面。

---

## 2. 枚举 → (DType, quant_group, flags) 映射（布局侧唯一权威）

`src/targets/qwen3_6/impl/runtime/layouts_impl.h:69`（`target_kv_cache_profile`）：
```cpp
BFloat16                    → {DType::BF16, 0}
Int8Group64                 → {DType::I8,   kKvInt8QuantGroup /*64*/}
Fp8E4M3Row256               → {DType::FP8_E4M3FN, kKvFp8QuantGroup /*256*/}
RotatedInt8KInt4V / RotatedInt4KInt4V / RK4V4E8 / RK2V4E8
                              → {DType::I8,   kKvInt8QuantGroup /*64*/}   // 全部归 I8 家族
```
常量：`kKvInt8QuantGroup=64`、`kKvFp8QuantGroup=256`
（`src/targets/qwen3_6/export/ninfer/targets/qwen3_6/decoder_state.h:12-13`）。

`layouts_impl.h:783-798`（`SequencePlanningInputs` 从 `options.kv_cache` 推导 flag）：
```cpp
kv_packed_v   = (rk8v4 || rk4v4 || rk4v4-e8 || rk2v4-e8)
kv_rotate_k   = (rk8v4 || rk4v4 || rk4v4-e8 || rk2v4-e8)   // 所有旋转模式都旋转 K
kv_rotate_v   = (rk8v4 || rk4v4 || rk4v4-e8 || rk2v4-e8)   // 所有旋转模式都旋转 V
kv_packed_k   = (rk4v4 || rk4v4-e8)
kv_e8_lattice = (rk4v4-e8)
kv_e8_root    = (rk2v4-e8)
```
> 注意：`kv_rotate_*` 恒等于"是否属于旋转家族"。`rk8v4` 是"K 不 packed、V packed"；
> `rk4v4` 起 K 也 packed；`rk4v4-e8` 在 `rk4v4` 上叠加 `e8_lattice`；`rk2v4-e8` 用 `e8_root` 替代 `packed_k`。

**反向**（flags → 枚举，`src/targets/qwen3_6/impl/runtime/program_impl.h:11530`，用于 `memory_summary`）：
```cpp
I8 家族下:
  e8_root     → RK2V4E8
  e8_lattice  → RK4V4E8
  packed_k    → RotatedInt4KeyInt4ValueGroup64   // rk4v4
  rotate_v    → RotatedInt8KeyInt4ValueGroup64   // rk8v4
  否则        → Int8Group64
```

flag 常量：`layouts.h:78-85`（`SequencePlanningInputs` 字段）、
`decoder_state.h:21-28`（`DecoderStateSpec` 字段）、`program.h:633-638`（`ProgramImplCore` 成员）、
序列化：`session_snapshot_impl.h:429-432`（`kv_flags` bitfield，`kKvFlagPackedV/RotateK/RotateV/PackedK/E8Lattice/E8Root`）。

---

## 3. 存储布局（paged KV 平面）

- 页大小 `kPagedKVPageSize=64`（`src/core/paged_kv_cache.h:150`）；`device_plane_order = PageMajor`。
- 索引（`src/ops/kernel/paged_kv_address.cuh`）：
  ```
  page_head_offset<LE,HE>(page, head)  = LE * 64 * (head + HE * page)
  element_offset<LE,HE>(page,head,page_off,leading) = page_head_offset + LE*page_off + leading
  ```
  即物理内存序：**page(64 token) → kv_head → token-in-page(64) → code_extent(LE)**。
  注意 LE（`leading_extent`）是**每 head 的 code 维度字节数**（I8/U8 每 code 1 字节、BF16 每 code 2 字节）。
- 每个 attention 层占 **2 平面（BF16）或 4 平面（量化）**：`k_code, v_code, k_scale, v_scale`。
  平面几何 `KVPlaneGeometry{dtype, leading_extent, head_extent=kv_heads, alignment=256}`。
- 平面 extent 推导（`src/targets/qwen3_6/impl/state/decoder_state.cpp:plan_cache`，`4JC-930`）：
  ```cpp
  k_head_extent = e8_root ? head_dim/4 : (packed_k ? head_dim/2 : head_dim);  // 256 / 128 / 64
  v_head_extent = packed_v ? head_dim/2 : head_dim;                            // 128 / 256
  k_plane_dtype = (packed_k || e8_root) ? U8 : dtype;
  v_plane_dtype = packed_v ? U8 : dtype;
  scale_extent  = head_dim / quant_group;   // I8→4, FP8→1
  ```
- 平面字节 = `kv_heads × 64 × leading_extent × dtype_size`；scale 平面 = `kv_heads × 64 × scale_extent × 2(FP16)`。
- 校验约束（`decoder_state.cpp:85-92`，**新增模式必须满足**）：
  ```
  (packed_v || rotate_k || rotate_v || packed_k || e8_lattice || e8_root) 且 dtype 非量化 → 报错
  rotate_v && !packed_v        → 报错（旋转 V 必须 V4 打包）
  e8_lattice && !packed_k      → 报错（E8 格必须 K 打包）
  e8_root 允许独立于 packed_k（走 U8 平面，extent head_dim/4）
  ```
- 平面视图：`PagedKVLayerView`/`PagedKVBatchLayerView`（`paged_kv_cache.h`）携带
  `k_pages/v_pages/k_scale_pages/v_scale_pages` + `packed_v/rotate_k/rotate_v/packed_k/e8_lattice/e8_root`
  六个 bool，attention/append kernel 直接据此选分支。

---

## 4. 编码（写入）：`kv_cache_append`

入口 `include/ninfer/ops/kv_cache_append.h` → `src/ops/kv_cache/append/{kernel.cuh,launch.cu,launch.h}`。
把新 token 的 BF16 K/V 量化后写进 paged 平面。**没有独立 transcode kernel**，编码与（可选）H64 旋转
融进 append kernel。head_dim 恒 256（`kKVCacheAppendFullHeadDim`），KVHeads ∈ {4,2}。

### 4.1 各 codec 原语（`src/ops/kv_cache/*.cuh`）

**int8 / i4（`int8_g64_codec.cuh`）**
- `kv_cache_int8_quant_params(absmax)`：`scale = fp16(absmax/127)`，`inv = 1/represented_scale`
  （**scale 过一次 FP16 RNE 边界**，code 用该表示值的倒数——这是所有 int8 系的精度合同）。
- `kv_cache_int8_quant_code` = clamp(±127)；`kv_cache_i4_quant_code` = clamp(±7)。
- `kv_cache_pack_i4(lo,hi)` = `(lo&0x0f)|((hi&0x0f)<<4)`；`kv_cache_unpack_i4` 做 `nib^8 - 8` 恢复 ±7。
- `kv_cache_hadamard64(x0,x1,mask)`：每 64 组内的 H64 Sylvester 旋转（32-lane butterfly + 尾 `*0.125`），
  用于**旋转模式**对 K/V 逐 64 组旋转后再量化。

**Hadamard 旋转（`hadamard_d256.cuh`）**
- `normalized_hadamard_d256_inplace`：一整条 D256 行（8-dim/warp lane）的**归一化** Sylvester 变换，
  FP8 行的 K/V 旋转用它；`hadamard_d32_columns_inplace` / `hadamard_d64_fragment_inplace` 是其因子分解。

**FP8（`fp8_e4m3_row_codec.cuh`）**
- `kKVCacheFp8MaxFinite=448`、scale 上下界 `[0x1p-24, 65504]`；整 256 行一组（1 scale/token/head）。
- `kv_cache_fp8_quant_code*` = `__nv_cvt_float_to_fp8(..., SATFINITE, E4M3)`。

**E8 格投影（`src/ops/kernel/e8_lattice.cuh`）— 供 `rk4v4-e8` 的 K**
- `e8_project_8d_fast`（标量版）/ `e8_project_8d_warp_single`（8-lane subgroup warp 版）/
  `e8_project_8d_warp(x0,x1,lane)`：Conway-Sloane 最近格点——先在 D8（偶和整数格）取 `rintf` 候选，
  若和为奇数在最坏维 ±1 修正；再在 `D8+0.5` 陪集取候选；取 L2 距离更近者。
- 8 维由 8 个 lane 各持 1 维，用 `__shfl_xor_sync`（mask∈{1,2,4}）在 8-lane 子组内归约 sum/err/dist。
- **刻意近似**：投影后 `rintf()` 再 clamp 到 ±8 存成 int4——`D8+0.5` 半陪集被塌缩、从不重建
  （int4 里没有 coset bit）。这是 `rk4v4-e8` 相对 `rk4v4` 的精度来源：格投影让 4-bit 舍入显著更紧。

**E8 根（`src/ops/kernel/e8_root_codec.cuh`）— 供 `rk2v4-e8` 的 K**
- 240 根 = 112 型 A（`±e_i ± e_j` 排列）+ 128 型 B（`±0.5` 全符号、偶数个负号）。
- `e8_quantize_root_8d(u, v_out)`：把单位 8-d 向量映射到最近根，返回 0..239 code，并输出根方向。
- `e8_encode_cylinder_8d*` / `e8_encode_cylinder_8d_warp`：**柱面分解** = 8-bit 根 code +
  4-bit 对数半径 `rad_idx`（`3·log2(r_rel)+8`，clamp 1..15）+ 4-bit 超八面体轴 `axis_idx`
  （残差最大维 + 符号）。`rad_axis = (rad_idx<<4) | axis_idx`。
  → 每 8-d 子向量 **2 字节**（`c1=root`, `c2=rad_axis`）。64 组 = 8 子向量 = 16 字节；4 组 = 64 字节 = head_dim/4。
- 关键正确性注记（`e8_encode_cylinder_8d_warp` 内 `I6l` 起）：`rad_idx` 由 8-lane 子组的范数推出，
  **同一 warp 的 4 个 8-lane 子组可能各不相同**，因此函数**绝不能在 `rad_idx==0` 时早退**——
  必须让整 warp 32 lane 在每个 `__shfl*_sync` 上都汇聚，否则 divergent shuffle 是 UB、会污染
  RK2V4E8 存的 key code；近零范数子组改在**所有 shuffle 完成后**由受保护的输出写清零。
  （原设计来自 UDPSendToFailed/ninfer-4090，此注释是 port 后的正确性加固。）
- `e8_root_decode_8d_fast`：`vadd4` 硬件 SIMD 把根表 `c_e8_stage1_i8x8[256]`（2KiB `__device__`，`__ldg`）
  与轴表 `c_axis_i8x8[16]`（16 条 `__constant__`）相加，乘 `c_radius_scale[rad_idx]`（16 值，中心 0.5）
  得 8-d int8 重建。
- `__constant__` 表 `c_radius_scale`/`c_axis_i8x8` 与 `__device__` 表 `c_e8_stage1_i8x8` 是硬编码常量。

### 4.2 append kernel 与分发（`append/kernel.cuh` + `append/launch.cu`）

- `kv_cache_append_full_bf16_kernel`：BF16 直拷（int4 向量化）。
- `kv_cache_append_full_fp8_kernel` / `_page_kernel`：每 warp 持一整条 D256 行 →
  `normalized_hadamard_d256_inplace` → warp absmax → E4M3 code + 1 scale。
- `kv_cache_append_full_i8_kernel` / `_page_kernel`：模板
  `<PackedV, RotateK, RotateV, PackedK, E8Lattice, E8Root>`，每 warp 持一个 (token, head, 64 组)，
  每 lane 2 维（`d0=group*64+lane`, `d1=d0+32`）。核心逻辑：
  ```
  RotateK? H64(k0,k1) : - ;  RotateV? H64(v0,v1) : -
  ksh = fp16(warp_absmax(K) / ((PackedK||E8Root) ? 7 : 127))   // 4-bit/根 用 /7，8-bit 用 /127
  vsh = fp16(warp_absmax(V) / (PackedV ? 7 : 127))
  K:  E8Root → e8_encode_cylinder_8d_warp → 2 字节/8-d 写 U8 平面
      PackedK(E8Lattice? 投影: 直舍) → int4 clamp ±8 → pack_i4 2/字节
      否则 int8 clamp ±127 直写
  V:  PackedV → pack_i4 2/字节 ; 否则 int8 直写
  lane0 写 ksh/vsh scale
  ```
  - `tokens>=32`（`launch.cu` 阈值）走 `_page_kernel`（按 8-token tile 绝对调度，CTA 页局部），
    否则走 `_i8_kernel`（按 unit 线性调度）。
- 分发（`launch.cu:RTL-yOp`，`cache.e8_root/e8_lattice/packed_k/packed_v` 优先序）：
  ```
  e8_root     → <true,true,true, false,false,true>
  e8_lattice  → <true,true,true, true, true,false>
  packed_k    → <true,true,true, true, false,false>
  packed_v    → <true,true,true, false,false,false>
  否则        → <false,false,false,false,false,false>
  ```
  （元组 = `PackedV,RotateK,RotateV,PackedK,E8Lattice,E8Root`。）
- 前缀 append（MTP/dflash 用）`kv_cache_append_prefix_*`：**只 BF16 拷贝**（`kKVCacheAppendPrefixHeadDim=128`、
  `Heads=8`、`Window=4096`、`Page=64`），不走量化——量化缓存的可见 key 走主 append 路径。

---

## 5. 解码（读取）：causal softmax attention

消费端是 **paged causal attention**（`src/ops/softmax_attention/dense/causal_cache/`），分 prefill
（`prompt*`）与 short/decode（`small_t*`）两条，各自按 dtype 分 kernel：
`bf16` / `i8` / `fp8`（`prompt_{bf16,i8,fp8}`、`small_t_{bf16,i8,fp8}`）。
其它 attention op（`context` / `packed` / `sliding_window`）走原始 K/V buffer，服务 MTP/DFlash/verify，
不消费量化 paged 缓存（见 `small_t.cu:242-254` 对 cache_dtype 的断言：仅 `BF16/I8/FP8_E4M3FN`）。

**I8 系 prompt kernel**（`prompt_i8.cuh`，`causal_attention_prompt_i8_kernel<...>`）：
- **Q 侧**：每 (row,64 组) 一 warp，`RotateK` 时对 Q 也做 H64（与 K 的旋转对齐），Q→int8 ±127，
  per-group scale `q_scale`。
- **K 侧**（dequant 到 int8，进 smem 后走 **INT8 张量核 `mma_s8`** 做 QK）：
  - `int8-group64`：K 原生 int8，`cp_async` 直读；
  - `packed_k`（rk4v4/rk4v4-e8）：`kv_cache_unpack_i4x16` → int8(±7) 进 smem；
  - `e8_root`（rk2v4-e8）：读 2 字节/8-d → `e8_root_decode_8d_int8` → int8 进 smem；
  - QK 结果 `score += qscale × kscale × (int8 mma)`（FP16/FP32 折进 FMA）。
- **V 侧**：dequant 到 **fp16** 做 PV（`mma_f16`；sm_89 上 64-key tile 先 fp16 累加再折入 fp32 累加器）：
  - int8 V：`causal_prompt_i8_dequant_f16x8`（sm_89 用 byte-permute 走 I2F 快路，bit-identical）；
  - packed V（rk*）：`kv_cache_unpack_i4x16` → dequant。
- **输出逆旋转**：`cache.rotate_v` 时，attention 输出处于旋转域，launch 后跑
  `kv_cache_inverse_rotate_output_kernel`（`int8_g64_codec.cuh`）做 H64 逆变换把输出拉回原域。

**I8 系 small_t kernel**（`small_t_i8.cuh`，`launch_tc_partial_i8<...>`）：同样的
`E8Root/PackedK/PackedV/…` 分支（`small_t.cu:278-309` 的 `NINFER_CAUSAL_SMALL_T_DISPATCH`），
decode 阶段可见 key 走 paged i8 缓存，V 用 `kv_cache_int8_dequant_i8x8_from` 还原（`small_t_i8.cuh:625`）。

**FP8 prompt/small_t**（`prompt_fp8.cuh`/`small_t_fp8.cu`）：K/V 用 E4M3 code + 行 scale 还原。
**BF16** 直接 bf16 matmul。

> 精度合同（AGENTS.md）：每个量化 Op 有**独立 naive FP32/FP64 oracle**，从表示的公共输入算完整公式；
> packed 权重用精确 code+scale 解码；跨 Op 无共享容差。E8 codec 本身有位精确微基准（见 §8）。

---

## 6. 端到端数据流

```
BF16 K/V (新 token)
  └─ kv_cache_append  ──(可选 H64/D256 旋转)──(量化: int8/i4-pair/E4M3/E8格/E8根)
       ├─ K code 平面 (I8 或 U8)      ├─ V code 平面
       └─ K scale 平面 (FP16)         └─ V scale 平面 (FP16)      [PagedKVCache, PageMajor]
                                                        │
  Q (attention) ──(Q int8 量化, 可选 H64)──► causal softmax attention (prompt / small_t)
       └─ K dequant→int8 → mma_s8 (QK) ;  V dequant→fp16 → mma_f16 (PV)
            └─ (rotate_v? 逆 H64) ──► BF16 attention 输出
```

---

## 7. 各入口的接线点（新增 `rk6v4e8` 必须同步的地方）

| 位置 | 作用 |
|---|---|
| `include/ninfer/types.h:31` | `KvCacheStorage` 枚举（追加新值） |
| `src/targets/qwen3_6/impl/runtime/layouts_impl.h:69,783-798` | 枚举→profile(DType/group)、枚举→flags |
| `src/targets/qwen3_6/impl/state/decoder_state.cpp:plan_cache` | 平面 extent/dtype 推导 + 约束校验 |
| `src/ops/kv_cache/append/launch.cu`（分发）+ `kernel.cuh`（`kv_cache_append_full_i8_*_kernel` 模板实例） | 编码分发/新分支 |
| `src/ops/softmax_attention/dense/causal_cache/prompt.cu`、`small_t.cu`（分发宏）+ `prompt_i8.cuh`/`small_t_i8.cuh` | 解码分发/新分支 |
| `apps/cli/options.cpp:56-62`、`apps/cli/main.cpp:97-109`（`format_kv_cache`）、`apps/cli/options.h` | CLI parse + 显示名 |
| `src/serve/serve_options.cpp:50-56`（parse）、`src/serve/request_log.cpp:134-146`（`kv_cache_name`）、`serve_options.h` | serve parse + 日志名 |
| `src/targets/qwen3_6/impl/runtime/program_impl.h:11530`（`memory_summary` 反向映射） | 内存汇总报回枚举 |
| `bench/targets/qwen3_6_27b/ninfer_bench_support.cpp:51-68,862`、`apps/perplexity/main.cpp:102-132` | 基准/困惑度工具 parse + 名称 |
| `docs/serving.md:734`、`docs/cli.md:203`、`README.md:316/382`、`docs/udp-fork-comparison.md` | 文档 |
| `tests/test_serve_options.cpp`、`tests/test_ninfer_bench_support.cpp`、`tests/test_kv_cache.cpp`、`tests/targets/…/test_engine_*_real.cpp` | 测试 |

**关键设计观察**（对 `rk6v4e8` 直接有用）：
- 现网所有 packed/旋转/E8 模式的 K 侧都最终落回 **int8（`mma_s8`）** 进 QK 张量核——`int4`/`E8根`
  只是**存储更密**，读回仍还原成 int8 code。V 侧一律还原成 fp16。
- "6-bit" 若指 K 用 6-bit 码（±31）：现有打包是 **2 i4/字节** 或 **2 字节/8-d**；6-bit 不整字节对齐，
  需新引入码表/位打包方案（例如每 3 code 占 2 字节不可行，需 4 code/3 字节等），或改走"格投影 +
  int8 存储"（即复用 int8 平面、靠 E8 类投影把有效精度压到 ~6 bit），后者最贴合现有 `mma_s8` 读路径。
- 新增枚举后：`switch` 全覆盖（`-Wall` 会提示 missing case）、`memory_summary` 反向映射、
  所有 parse/名称表、CMake 无需改（kernel 是模板，实例由分发决定）。

---

## 8. 校验与微基准

- `tools/test_kv/`：
  - `test_e8_codec.{cu,cuh}`：E8 格投影 + 根 quantize/decode 的**独立 CUDA 微基准**，
    256 head_dim / 8 sub-dim / 32 子空间 / 64-token tile；`E8Packed2BitTile`（64 字节 code + 64 字节 scale）
    与 `E8Packed4BitTile`（128 字节 code + 8 字节 scale）两种布局。生产 codec 对其**位精确**对齐
    （README：cosine 96.155% / 98.678%）。
  - `kv_oracle.py`：KV 数值 oracle（与 AGENTS.md 的"每个浮点 Op 独立 oracle"对应）。
  - `verify_1m_retrieval.cu`：1M 可见 key 的检索验证（UDPSendToFailed fork 引入的上限抬升）。
- 单测：`tests/test_kv_cache.cpp`（布局/平面）、`tests/test_serve_options.cpp`（`rk8v4→RotatedInt8KeyInt4ValueGroup64`、
  `rk4v4-e8→RK4V4E8`、`rk2v4-e8→RK2V4E8`、`fp8→Fp8E4M3Row256`，默认 `BFloat16` 不变）、
  `tests/test_ninfer_bench_support.cpp`（parse + `memory.kv_cache` 名称）。
- 质量/容量脚本：`tools/bench/run_rk8v4_quality.py`（`KV_MODES=("int8","rk8v4")` 匹配 1k 语料）、
  `run_rk8v4_c2_smoke.py`（C2 API 冒烟）、`scripts/oom_diag_4090*.sh`（rk8v4 200k OOM 诊断）。

---

## 9. 容量与选型（实测结论摘录）

- **容量排序**（每 token/head 字节，越小越省）：
  `bf16(1024) > int8(528) ≈ fp8(516) > rk8v4(400) > rk4v4 = rk4v4-e8(272) > rk2v4-e8(208)`。
- `int8` 是推荐默认；`rk8v4` 在 4090 上把 C1 自动 sizing 抬高（`docs/rtx-3090-windows.md:93`）。
- **NVFP4 + 长上下文的硬动机**（`docs/udp-fork-comparison.md`，UDPSendToFailed PR #35 实测）：
  NVFP4 权重下 32GB 机器 `int8` KV 无法到 262,144（预留 13.02GiB > 权重后 11.66GiB），
  而 `rk8v4` 在 31.15GiB 可启动（MTP k=3 + vision）；检索/质量对 `int8` 是统计平手。
  → 旋转/E8 模式的价值 = 用 K/V 精度换容量，而非普遍降质量。
- E8 模式（`rk4v4-e8`/`rk2v4-e8`）在 `rk4v4`/`rk8v4` 的骨架上叠加格/根投影，进一步提升 K 的
  4-bit（或 2 字节/8-d）量化精度，用于更激进的 K 压缩档。
- 20w 上下文 OOM 真凶是 **pinned 主机内存**（context cache host 层），与 KV 模式/容量轴无关
  （AGENTS.md 定案）；host 层现默认 0、`--host-kv-mib`/`--host-state-slots` 显式 opt-in。

---

## 10. 一句话小结（给 `rk6v4e8`）

**一种 KV 模式 = `KvCacheStorage` 新枚举值 + 一组 flag（§2）+ 三处 kernel 分发
（append 编码 §4 / prompt & small_t 解码 §5）+ 全链路 parse/名称/汇总（§7）。**
K 侧无论 i4/8bit/E8 都还原成 int8 走 `mma_s8`，V 侧还原成 fp16 走 `mma_f16`，
scale 恒为 FP16、I8 系 per-64 组、FP8 系 per-256 行；旋转模式对 K/V 做 H64、对输出做逆 H64。
`rk6v4e8` 若要在 K 上引入 6-bit 有效精度，**最省力的路径是复用 I8/U8 code 平面 + `mma_s8` 读路径，
靠 E8 类格/根投影把码值压到 ~6-bit 动态范围**（避免为 6-bit 专门造位打包），再按 §7 清单打通管线。
