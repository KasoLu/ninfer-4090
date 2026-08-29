# Port ledger

The RTX 4090 fork (this repository, branch `rtx4090-port`) and the RTX 5090 fork
([sergiuszm/ninfer-5090](https://github.com/sergiuszm/ninfer-5090), branch
`nuntius-serve`) share almost all of their engine and serve code. Features are
born in one tree and cherry-picked into the other. This ledger records, per
feature, the commit hash in each tree, so coverage stays checkable without
archaeology.

Maintenance rules:

- Cherry-pick with `git cherry-pick -x`, so the destination commit records its
  source hash. Each repository holds the other as a local remote
  (`local-5090` here, `local-4090` there).
- Port in the session that ships the feature. Delayed ports pay a growing
  adaptation cost; `f640b404` on the 5090 side is the receipt.
- When a feature is deliberately not ported, record the decision here instead
  of leaving a silent gap.

## Feature rows

| Feature | 4090 (`rtx4090-port`) | 5090 (`nuntius-serve`) | Notes |
|---|---|---|---|
| `context_window` in `/v1/models` | `0f308358` | `ed606ed5` | |
| Prometheus `/metrics` | (commit series) | `fc582982`, `5fa5ffa1` | |
| Retained depth on idle `/slots` | `b0893e79` | `1a9640f1` | |
| Vision modality in `/v1/models` | `b6172f24` | `61013cf1` | Born on the 5090 side |
| fp16-accumulate PV tiles | part of `ce50e995` | `4483c820` | 4090 folds it into the sm_89 retune |
| 413 body fix + media prompt cap | `85f685a3`, `bde2765c` | `66423552`, `e9093c77` | Born on the 5090 side |
| sws_scale stride pad | `5a08683d` | `060bb320` | Born on the 5090 side |
| Tool content-part arrays | `d78df936` | `b0a0a6fe` | |
| llama.cpp-compatible `timings` | `0f95b32e` | `0834b6cb` | From the shantanusingh16 fork |
| Final-chunk usage param fix | `265011d9` | `51857983` | |
| Slot save/restore to disk | `beaeb70a` | `aeaf3f28` | 5090 needed `f640b404` (KV modes) |
| Session digests + `if_digest` | `8e478945` | `40504615` | |
| Cheapest-lane reuse tie-break | `1614ef54` | `59cb1afb` | |
| `/slots` snapshot publishing | `4a4fa92b` | `93fadf94` | |
| Live llamacpp `/metrics` counters | `656b0df7` | `b38ae92d` | |
| Turn checkpoint ring | `3a2e7f07`, `cba2c1f8`, `2cbe488d`, `3419fe43` | `6826b8b0`, `1ac61caf`, `8b28e502`, `66401303` | Picked with `-x` |
| Auto-save on eviction | `8093c640` | `cad0218e` | Picked with `-x` |
| Causal-tile key-block partition | `694e01f0` | `b5179823` | i8 body re-applied per schedule; bf16 and common taken verbatim |
| E8 codec hardening | `bc569eb8`, `a0e03d37` | not applicable | The 5090 tree carries no E8 code. Third-hand from upstream PR #35 through the sibling fork; authorship preserved |
| Production E8 codec test | `94830b3f` | not applicable | Same reason. Also registers the standalone oracle, which ctest had never run |
| GDN QK norm XOR butterfly | `6e239351` | open | Bit-exact over 6.4M lanes, measures near zero. Port is cheap; value is consistency, not throughput |
| GDN uniform value pack | `c4d09b61` | open | Bit-exact over all 65536 bf16 patterns, measures near zero. Born here, not a port |

## Inbound ports from downstream forks

Nine forks of `sergiuszm/ninfer-4090` now exist. This table records what each
one contributed and what was declined, so that a later sweep does not
re-examine the same commits. Survey date: 2026-08-29.

| Source commit | Here | Decision |
|---|---|---|
| jomcgi `a0d78215`, `chat_template_kwargs` aliases | `6affed2e` | Ported. llama.cpp and vLLM spell the Qwen thinking controls under `chat_template_kwargs`. Existing clients reach the effort knob without a change |
| Don-Chad `db076d67`, remove CUDA forward-compat libraries | `ff925039` | Ported. Our `Dockerfile` uses the same `nvidia/cuda:13.1.2` base and carried the same latent failure. The deployed `ninfer-dev:runtime` image is built by hand, so no rebuild is forced |
| Don-Chad `ccb20680`, qualify Qwen3.8 on SM89 | not applicable | `layouts_impl.h` already gates on `device.sm() != 89`. The upstream form admits 86 or 89 and would loosen our gate |
| jomcgi `1513de5a`, ghcr image workflow | declined | Hardwired to `ghcr.io/jomcgi` and to a `runAsNonRoot` cluster policy. We deploy hand-built local images |
| Don-Chad `7afc8e17`, resident-CTA budget from the runtime SM count | open | Not a cherry-pick. See the note below |

The `7afc8e17` principle applies to us, but its constants do not. That fix
separates per-SM occupancy from the device-wide budget for sm_86, where the
supported range is 82 to 84 SMs. Our `bf16_gdn_gating_proj_plan.cpp` hardcodes
the 128-SM budget of the RTX 4090 and feeds the same constants to a
`static_assert`. The number is correct for the RTX 4090 and wrong for every
other Ada device: the RTX 4080 has 76 SMs and the L40S has 142. On a card with
fewer SMs the budget is overstated. A cooperative launch that does not fit is
then accepted, and the driver rejects it with
`cudaErrorCooperativeLaunchTooLarge`. A port needs `DeviceContext::sm_count()`,
our own per-SM occupancy figures, and a `kMinSupportedSmCount` value for sm_89.

## Deliberate non-ports

| Feature | Lives in | Decision |
|---|---|---|
| sm_89 attention retune (`ce50e995`) | 4090 | Architecture-specific by design |
| E8 lattice KV modes (`c3a6e5c4`, `ec56f922`, series) | 4090 | Declined for the 5090 on 2026-08-19: 32 GB fits the full 262K context on `int8`, so E8 would buy only the decode-at-depth gain. **Revisit if NVFP4 becomes the goal**: upstream PR #35 ports E8 to sm_120a, and NVFP4 cannot reach 262K on `int8` at all. Wait for that PR to merge rather than hand-porting it. See `docs/udp-fork-comparison.md` |
| `--vision-max-tokens` (`0c3d2bee`, `73b42127`) | 4090 | Open: the 5090 fits the legacy 32K scratchpad next to 262K + vision, so nothing forces the port |
| Single-token W8 column-store fix (`68e2d0be`) | 4090 | Not applicable: the 5090 tree's `w8_linear_add_gemm_splitk.cu` is the upstream variant without the vulnerable tail dispatch |
| NVFP4 weights profile | 5090 (upstream) | Ada has no FP4 tensor cores; the 4090 gates the A4 tests off instead |

## Long-term direction

The measured divergence between the trees is about 40 files once in-flight
ports land: roughly half architecture-specific kernels, half platform
configuration. The plan of record is to converge on one repository with two
architecture profiles (`sm_89` and `sm_120a` behind a CMake switch) and retire
the second tree to a deploy configuration. Until then, this ledger is the
source of truth for coverage.
