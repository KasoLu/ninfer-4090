# HANDOFF — rk6v4e8 4090 卡死：根因已定位，修复待落盘+验证

> 本文档是给"新会话"接手的。目标只有一件：**把 rk6v4-e8 在 4090 上的 GPU kernel 死锁修掉，让 `--kv-dtype rk6v4-e8` 能正常出答案**，然后回到用户最初的需求（跑 benchmark 测 rk6v4e8 vs bf16 的余弦，确认只比 rk8v4 低一点点、≥99%）。

---

## 0. 一句话现状
- **根因已 100% 定位**（运行时证据 + 静态推导双重闭合）：K6 写分支用了**非 uniform 的 warp shuffle mask** `sub_mask = 0x000000FFu << (lane & 0x1FC)` 配 `__shfl_down_sync(sub_mask, c, o)`（o=1..3）。同一 warp 32 lane 传入 4 个不同 mask 操作数，且 shfl_down 偏移 1..3 会让组内非 leader lane 读到**自己 sub_mask 之外**的源 lane = PTX 规格违例 = UB → sm_89（4090）上 warp 挂死。
- **修复方案已定稿**（位精确等价、纯 uniform `__shfl_sync(FullMask, ...)` 显式源 gather），**代码已写好但截至本文档时尚未真正落盘**（多次 edit 回执 success 但 grep 复核发现盘上仍是旧代码——edit 工具的成功回执不可信，**必须用 grep 落锤**）。
- 下一步就是：**把修复真正写进 3 个位置 → grep 复核 → dev docker 增量编译 → 用户在 4090 上 pull + 增量 build + 重跑测试1**。

---

## 1. 硬约束（务必遵守）
- **审批提示已禁用**：任何需要审批的操作一律被自动拒绝，**绝不要设 `sandbox_permissions`**。
- DSH 文件策略 `danger-full-access`，文件读写不受限。
- **4090 GPU 相关操作固定由用户手动执行**（serve 启停、benchmark、profiler、测试）。我（会话）只做**不碰 GPU 的只读远程检查**（查文件/进程/读日志）。绝不在 4090 上起 GPU 任务。
- 提交仅在用户要求时创建（Conventional Commit 格式）。
- 代码风格 `.clang-format` LLVM，100 列、4 空格。
- 远程通道：`python scripts/remote_4090.py run "cmd"`（paramiko，192.168.137.2:22，user `ninfer`，PASS `NinFer-Build2026-x7`）。4090 是 **Windows（cmd 语法）**；进容器用 `docker exec ninfer-4090-kaso-dev <cmd>`（**裸命令，禁止 `bash -c '...'` 嵌套引号、禁止重定向 `2>/dev/null`、禁止 `grep -E 'a|b'` 管道——都会被宿主 cmd 拆碎/吞掉**）。容器内 /tmp 不在宿主 SFTP 路径：要取文件先 `docker cp <容器>:/tmp/x C:\Data\ninfer\x` 再 `python scripts/remote_4090.py get C:\Data\ninfer\x <本地>`。
- 开发机（本机）无 NVIDIA GPU：只能编译不能跑 GPU 测试。构建用 `docker run --rm -v C:\Workspace\codes\ninfer-4090-kaso:/src ninfer-local-build:latest cmake --build /src/build --parallel`（build/ 已配置 Ninja+sm_89+Release）。4090 工作区 = `C:\Data\ninfer\ninfer-4090-kaso`，容器挂载为 `/ninfer-4090-kaso`，模型 `/models/qwen3_8_27b.ninfer`。
- **4090 代码同步协议（2026-09 用户指令，持续有效）**：一律走 **本地 commit+push → 4090 上 `git fetch` + `git reset --hard origin/rk6v4e8`**（pull 属非 GPU 操作，会话可经 `remote_4090.py run 'docker exec ... git ...'` 远程执行）；4090 仓库必须保持与 origin 某提交完全一致的干净状态，**禁止 SFTP put 仓库文件**（2026-09 曾违规被用户纠正）。构建统一走 `docker exec -d ninfer-4090-kaso-dev cmake --build /ninfer-4090-kaso/build --parallel`（`-d` 脱离式，会话中断不丢）。

---

## 2. 根因细节（为什么挂、为什么别的模式没事）
- K6（rk6v4-e8 的 K 侧）4-lane quad 收集：4 个连续 lane 各持一个 6-bit 码，要聚成一个 24-bit 字（3 字节）。原实现：
  ```cpp
  const unsigned sub_mask = 0x000000FFu << (lane & 0x1FC);   // 每 4-lane 组一个 4-bit 子掩码
  std::uint8_t quad0[4]; quad0[0] = c0;
  for (int o = 1; o < 4; ++o)
      quad0[o] = (std::uint8_t)__shfl_down_sync(sub_mask, (int)c0, o);
  if ((lane & 3) == 0) { /* leader 写 pack_i6_quad */ }
  ```
  - `sub_mask` 按 lane 不同而不同（lane 0-7→0xFF，8-15→0xFF00，…），**非 uniform**。
  - `__shfl_down_sync(mask, x, offset)` 的语义是"读 source lane = own_lane + offset"。组内非 leader lane（组内第 1/2/3 号）读到的 source 会**跨出本组**，落在自己 sub_mask 之外 → UB。
  - 对比：`e8_project_8d_warp`（`src/ops/kernel/e8_lattice.cuh:179`，`0xFFu << (lane & 24)`）也是 sub_mask，但用的是 `__shfl_xor_sync`，源 lane = `lane ^ offset`（offset∈{1,2,4}）恒落在同一 8-lane 子组内 → 每个 lane 读取都合法 → **rk4v4-e8 在 4090 实测无恙，印证此判断**。
  - 而既有安全代码全用 `__shfl_down_sync(FullMask, x, 1)` / `__shfl_sync(FullMask, x, src)`（uniform FullMask）。
- 为什么"看起来数据对、纯 hang 无精度错"：leader（lane&3==0）读的 source 全在自己 sub_mask 内，值正确；挂死发生在非 leader lane 的越界读上，所以是纯死锁。
- 为什么 be17531d 的"修复"没修好：那次只是把 shuffle 移出 `if (leader)` 分支（让 32 lane 都执行、解决"FullMask 在 leader 分支里等不到"的第一次死锁），但**mask 本身仍非 uniform** → 第二次 hang。

### 定位证据链（都已存档，供复查）
- `CUDA_LAUNCH_BLOCKING=1 + LD_PRELOAD ktrace`（ktrace v3，钩 `__cudaLaunchKernel`/`cudaLaunchKernelExC`/`cudaStreamSynchronize`/`cudaEventSynchronize`）：最后一条 launch = **`kv_cache_append_full_i8_page_kernel`（grid=9x4x4, block=256, K6 写分支）**，其后 stream sync 无返回 → 自旋者锁定为 **append kernel 的 K6 写分支（prefill 内，64 token prompt 走 page kernel）**。
- 一次性解释所有旧疑点：small_t 四臂（NINFER_K6_BISECT）全卡（只切 decode 路径，没碰 append kernel）；rk4v4-e8 探针正常（PackedK 分支用 uniform FullMask）；compute-sanitizer 空日志 = GPU kernel 自旋（非 host 死循环）；`ctest -R kv_cache`（含 PageMajorK6 几何 {U8,192,2,256}+{U8,128,2,256}+{FP16,4,2,256}×2）在 4090 全过 = 非几何/内存池问题。
- 开发机留档：`tools/test_kv/kt_block.log`（+ `kt_k6.log`/`kt_nograph.log`/`kt_k6b.log`）、`tools/test_kv/ktrace.c`（v3，已提交 c91dfd5f）、`tools/test_kv/kt_addrs.txt`、`bt_match.py`（用法 `python bt_match.py ninfer_disasm.txt ninfer_syms_raw.txt <log>`，三参）、`bt_sym.py`、`ninfer_disasm.txt`/`ninfer_syms_raw.txt`。废弃可删：`t1c.qdstrm`/`probe.qdstrm`。

---

## 3. 修复（位精确等价，纯 uniform FullMask 显式源 gather）
把每处 `sub_mask + __shfl_down_sync(sub_mask, c, o=1..3)` 的 quad 收集，换成**从显式源 lane 做 `__shfl_sync(FullMask, c, qbase+m)`**（m=0..3，`qbase = lane & ~3`）。32 lane 全收敛、单一 uniform FullMask、只读本 quad 4 个源 lane → 与 leader 读到的值逐位一致。

**替换后的新代码（3 处都要）：**
```cpp
// Gather the 4-lane quad with full-mask shuffles from explicit source
// lanes: every lane of the warp converges on each __shfl_sync, the only
// mask form this codebase relies on. The per-quad sub-mask
// __shfl_down_sync variant deadlocked on sm_89 (4090 hang 2026-09-04):
// shfl_down offsets 1..3 make the group's non-leader lanes read source
// lanes outside their sub-mask, which is undefined behavior. The
// explicit-source gather below is bit-identical for the leader reads.
const int c0i = static_cast<int>(c0);
const int c1i = static_cast<int>(c1);
const int qbase = lane & ~3;
std::uint8_t quad0[4];
std::uint8_t quad1[4];
quad0[0] = static_cast<std::uint8_t>(__shfl_sync(FullMask, c0i, qbase));
quad0[1] = static_cast<std::uint8_t>(__shfl_sync(FullMask, c0i, qbase + 1));
quad0[2] = static_cast<std::uint8_t>(__shfl_sync(FullMask, c0i, qbase + 2));
quad0[3] = static_cast<std::uint8_t>(__shfl_sync(FullMask, c0i, qbase + 3));
quad1[0] = static_cast<std::uint8_t>(__shfl_sync(FullMask, c1i, qbase));
quad1[1] = static_cast<std::uint8_t>(__shfl_sync(FullMask, c1i, qbase + 1));
quad1[2] = static_cast<std::uint8_t>(__shfl_sync(FullMask, c1i, qbase + 2));
quad1[3] = static_cast<std::uint8_t>(__shfl_sync(FullMask, c1i, qbase + 3));
```
> 注意：small_t_i8.cuh 那处缩进多一层（在更深的作用域），把上面每行前面多补 4 个空格。`c0`/`c1` 变量名在三个文件里一致；full kernel 用 `page`、page kernel 与 small_t 用 `physical_page`——但新代码里用不到 page 变量，照抄即可。**leader 写逻辑（`if ((lane&3)==0) { quad_off=(lane>>2)*3; kv_cache_pack_i6_quad(quad0, &k_bytes[k_row+quad_off]); kv_cache_pack_i6_quad(quad1, &k_bytes[k_row+24+quad_off]); }`）保持不变，紧跟在新 gather 之后。**

### 要改的 3 处（替换"旧 sub_mask 块"→"新 gather 块"，含其上方的旧注释行）
1. **`src/ops/kv_cache/append/kernel.cuh` — full kernel K6 支**（约 :298 区，变量 `page`）：
   旧块从 `const unsigned sub_mask = 0x000000FFu << (lane & 0x1FC);` 那行开始，到 `}`（for 循环结束，`for (int o = 1; o < 4; ++o)` 的收尾 `}`）为止。上面还有 3 行旧注释（"Every lane of a quad must converge..."）。连同注释一起替换。
2. **`src/ops/kv_cache/append/kernel.cuh` — page kernel K6 支**（约 :479 区，变量 `physical_page`）：同上结构。
3. **`src/ops/softmax_attention/dense/causal_cache/small_t_i8.cuh` — K6 写支**（约 :299 区，缩进更深；在 `k6_bisect` 的 `else`（生产 i6 路径）分支内）：同上结构，多一层缩进。
   - **small_t_i8.cuh 读支不用改**（纯 `kv_cache_unpack_i6x16` 同步 unpack，无 shuffle）。
   - **append kernel 里 V 侧的 `__shfl_down_sync(FullMask, v, 1)` 保持不动**（本来就 uniform，安全）。
   - **`e8_lattice.cuh` 的 sub_mask 保持不动**（shfl_xor 源恒在子组内，已验证安全）。

### 落盘后必须做的复核（grep 落锤，因为 edit 回执不可信）
```
grep -rn "sub_mask = 0x000000FFu" src/   → 应只剩 e8_lattice.cuh 的 0xFFu<<(lane&24)（不是 0x1FC 那个），即 K6 的三处 0x1FC 应全部消失
grep -rn "__shfl_sync(FullMask, c0i" src/ → 应命中 3 处（新代码）
```
若新代码 grep 不到 / 旧 `0x000000FFu` 仍在 → 说明 edit 又没落盘，**重新 edit**（先 `read` 拿新鲜锚点，锚点是 3 字符 HASH，形如 `V4e`）。

---

## 4. 验证流程（严格顺序）
1. **dev 机增量编译**：`docker run --rm -v C:\Workspace\codes\ninfer-4090-kaso:/src ninfer-local-build:latest cmake --build /src/build --parallel` → 应全绿（只重编 kernel.cuh/small_t_i8.cuh 相关 + 链接）。
2. **交付用户 4090 验证**（用户手动执行）：
   - `git pull`（`C:\Data\ninfer\ninfer-4090-kaso`）
   - `docker exec ninfer-4090-kaso-dev bash -lc "cd /ninfer-4090-kaso && cmake --build build --parallel"`（增量）
   - 决定性回归（**测试1**，此前必卡死）：
     `docker exec ninfer-4090-kaso-dev ./build/apps/ninfer /models/qwen3_8_27b.ninfer --kv-dtype rk6v4-e8 --max-context 4096 --kv-capacity 4096 --prompt "1加1等于几？只回答数字" --max-new 32 --greedy`
     → 期望：~30s 出答案（如 "2"），不再卡死。
   - 若仍卡 → 再跑 ktrace（v3 已部署或按 §1 部署）+ `CUDA_LAUNCH_BLOCKING=1` 抓新 spinner 名。
3. **回归对照**（确认没弄坏别的模式）：同命令把 `--kv-dtype` 换成 `rk8v4`、`rk4v4-e8`、`bf16` 各跑一次，应都正常。
4. **回到用户最初需求**：跑余弦 benchmark 比较 rk6v4e8 vs bf16（确认只比 rk8v4 低一点点、≥99%）。可复用 `tools/bench/run_rk6v4e8_quality.py`（KV_MODES 含 bf16/rk8v4/rk6v4-e8，用 NINFER_EXE/NINFER_MODEL/NINFER_QUALITY_OUT 环境变量）或 `tools/test_kv/test_kv6_cosine`（`ninfer_test_kv6_cosine`，4090 有 GPU 才跑得动，无 GPU 退出 77=SKIP）。

---

## 5. 背景速览（为什么有 rk6v4-e8）
- rk6v4-e8 = 新 `KvCacheStorage::RK6V4E8` 模式：K 侧 6-bit 码 + E8 格投影（`e8_project_8d_warp`）+ H64 旋转，V 侧 4-bit + H64（同 rk8v4 的 V）。容量档 336 B/token/kv_head（K 192B + V 128B + k/v scale 16B FP16×4 组），介于 rk8v4(400) 与 rk4v4(272) 之间。K 平面 U8、`k_head_extent=192`（head_dim=256 的 3/4）。码值 value∈[-32,+31]，存 u6=value&0x3F，unpack (code^32)-32。4 code→24-bit 字 w=c0|c1<<6|c2<<12|c3<<18 → 3 字节。新 flag `kv_k6_bit`（`kKvFlagK6Bit=1U<<6`）。
- i6 codec 原语 @ `src/ops/kv_cache/int8_g64_codec.cuh`：`kKVCacheI6HeadExtent=192`；`kv_cache_i6_code_index<Geometry>(page,head,d,page_off)`（d 须 16 倍数）；`kv_cache_i6_code_from_int`/`kv_cache_unpack_i6`/`kv_cache_pack_i6_quad`/`kv_cache_unpack_i6x16`。
- 主体实现已提交：`d4c9763a`（feat）→ `bcecacf2`（`src/core/kv_cache_mode.h` 共享 flag 表，修 e8_lattice 漏项）→ `be17531d`（sub_mask "修复"=本次二次 hang 的根因）→ `c91dfd5f`（ktrace.c）→ `468a5fa7`（NINFER_K6_BISECT 五臂开关，方向搞错、无效但保留）。分支 `rk6v4e8`，origin=`KasoLu/ninfer-4090`。设计文档 `RK6V4E8.md`（仓库根）、梳理文档 `KVMODE.md`（仓库根）。
- 提交建议（修复验证通过后，需用户要求）：`fix(kv): use uniform full-mask gather for K6 quad packing to fix sm_89 hang`。

## 6. 踩坑备忘
- **edit 工具的成功回执不可信**：多次显示 "success" 但盘上未变。任何关键 edit 后必须 `read` 或 `grep` 落锤。
- **上下文触顶**会让回合被截断、工具调用没发出去（表现为"自动停下来"）。大日志（kt_*.log 数十 KB）读一次就涨很多，能少读就少读、读完即压缩。
- ktrace v2 曾误用镜像结构体把 dim3 当指针解引用 → 段错误打崩被跟踪进程（那轮 eager 的 SIGSEGV 是 tracer bug，非 ninfer 的）。v3 已修（结构体逐字节对齐真实 `cudaLaunchConfig_t`）。
- compute-sanitizer（CUDA 13 容器版）只采不析（无 analyzer，日志仅 28B 头）；`--force-blocking-launches` 下 GPU kernel 自旋会 SIGINT 打不动、日志空——空日志本身=GPU 自旋的证据。
- nsys 2025.4 容器版（NCu 捆绑）也是 collector-only，`.qdstrm` 无法 export 成 sqlite（"Invalid version prefix: U3w" 是假报错）。
