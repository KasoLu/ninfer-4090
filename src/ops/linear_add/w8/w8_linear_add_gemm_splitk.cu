#include "ops/linear_add/w8/w8_linear_add_kernels.h"

#include "core/device.h"
#include "ops/linear/w8/w8_small_t_mma.cuh"

#include <array>
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace ninfer::ops::detail {
namespace {

constexpr int kRows           = 2048;
constexpr int kRowsPerCta     = 16;
constexpr int kFirstExactCols = 2;
constexpr int kLastExactCols  = 32;
using ProjectionLauncher      = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

template <int Hidden, int ActiveCols>
void launch_active_cols(const Tensor& x, const Weight& weight, Tensor& residual_out,
                        cudaStream_t stream) {
    constexpr int TileCols = ActiveCols <= 8    ? 8
                             : ActiveCols <= 16 ? 16
                             : ActiveCols <= 24 ? 24
                             : ActiveCols <= 32 ? 32
                             : ActiveCols <= 40 ? 40
                                                : 48;
    constexpr int KWarps = ActiveCols <= 32 ? 8 : 4;
    constexpr int MinBlocks = Hidden == 4096 ? (KWarps == 16 ? 1 : 2) : (ActiveCols <= 32 ? 2 : 3);
    constexpr auto ScaleAccess =
        ActiveCols > 4 ? W8SmallTMmaScaleAccess::Shared : W8SmallTMmaScaleAccess::Direct;
    constexpr auto ActivationCache =
        Hidden == 4096 && (ActiveCols == 4 || (ActiveCols >= 27 && ActiveCols <= 40)) ? Cache::cg
                                                                                      : Cache::ca;
    using Geometry = W8LinearGeometry<kRows, Hidden>;
    using Schedule = W8SmallTMmaSchedule<KWarps, TileCols, MinBlocks, ScaleAccess, ActivationCache>;
    static_assert((kRows % kRowsPerCta) == 0);
    auto* residual = static_cast<__nv_bfloat16*>(residual_out.data);
    const W8ContiguousOutput output{residual, kRows};
    w8_small_t_mma_kernel<Geometry, ActiveCols, Schedule, W8ContiguousOutput,
                          W8SmallTMmaResidualEpilogue>
        <<<kRows / kRowsPerCta, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), output, W8SmallTMmaResidualEpilogue{});
}

template <int Hidden, std::size_t... Offsets>
constexpr auto make_projection_launchers(std::index_sequence<Offsets...>) {
    return std::array<ProjectionLauncher, sizeof...(Offsets)>{
        &launch_active_cols<Hidden, kFirstExactCols + static_cast<int>(Offsets)>...};
}

constexpr auto kK4096ProjectionLaunchers = make_projection_launchers<4096>(
    std::make_index_sequence<kLastExactCols - kFirstExactCols + 1>{});
constexpr auto kK6144ProjectionLaunchers = make_projection_launchers<6144>(
    std::make_index_sequence<kLastExactCols - kFirstExactCols + 1>{});

} // namespace

void w8_linear_add_splitk_mma_launch(const Tensor& x, const Weight& weight, Tensor& residual_out,
                                     cudaStream_t stream) {
    if (x.ne[1] < kFirstExactCols || x.ne[1] > 48) {
        throw std::invalid_argument("W8 linear_add split-K MMA requires exact T=2..48");
    }
    if (x.ne[1] > kLastExactCols) {
        std::int32_t offset = 0;
        while (x.ne[1] - offset >= kLastExactCols) {
            const Tensor x_slice = x.slice(1, offset, kLastExactCols);
            Tensor residual_slice = residual_out.slice(1, offset, kLastExactCols);
            w8_linear_add_splitk_mma_launch(x_slice, weight, residual_slice, stream);
            offset += kLastExactCols;
        }
        const std::int32_t tail = x.ne[1] - offset;
        if (tail == 1) {
            const Tensor x_slice = x.slice(1, offset, 1);
            Tensor residual_slice = residual_out.slice(1, offset, 1);
            if (weight.k == 6144) {
                w8_linear_add_decode_r16_launch(x_slice, weight, residual_slice, stream);
            } else {
                w8_linear_add_simt_r8_c4_launch(false, x_slice, weight, residual_slice, stream);
            }
        } else if (tail >= kFirstExactCols) {
            const Tensor x_slice = x.slice(1, offset, tail);
            Tensor residual_slice = residual_out.slice(1, offset, tail);
            w8_linear_add_splitk_mma_launch(x_slice, weight, residual_slice, stream);
        }
        return;
    }
    if (weight.k == 6144) {
        kK6144ProjectionLaunchers[x.ne[1] - kFirstExactCols](x, weight, residual_out, stream);
    } else {
        kK4096ProjectionLaunchers[x.ne[1] - kFirstExactCols](x, weight, residual_out, stream);
    }
    CUDA_CHECK(cudaGetLastError());
}

void w8_linear_add_medium_splitk_launch(const Tensor& x, const Weight& weight, Tensor& residual_out,
                                        cudaStream_t stream) {
    const std::int32_t t = x.ne[1];
    if ((weight.k != 4096 && weight.k != 6144) || t < 49 || t > 128) {
        throw std::invalid_argument("W8 linear_add medium split-K requires T=49..128");
    }
    std::int32_t offset = 0;
    while (offset < t) {
        const std::int32_t count = std::min<std::int32_t>(kLastExactCols, t - offset);
        const Tensor x_slice = x.slice(1, offset, count);
        Tensor residual_slice = residual_out.slice(1, offset, count);
        if (count == 1) {
            if (weight.k == 6144) {
                w8_linear_add_decode_r16_launch(x_slice, weight, residual_slice, stream);
            } else {
                w8_linear_add_simt_r8_c4_launch(false, x_slice, weight, residual_slice, stream);
            }
        } else {
            w8_linear_add_splitk_mma_launch(x_slice, weight, residual_slice, stream);
        }
        offset += count;
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
