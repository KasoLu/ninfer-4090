#include "ops/linear_pair/w8/w8_pair_kernels.h"
#include "ops/linear_pair/w8/w8_pair_plan.h"

#include "core/device.h"
#include "ops/linear/w8/w8_small_t_mma.cuh"

#include <cuda_bf16.h>

#include <array>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace ninfer::ops::detail {
namespace {

constexpr int kRows        = 1024;
constexpr int kHidden      = 2048;
constexpr int kRowsPerCta  = 8;
constexpr int kFirstExactT = 2;
constexpr int kLastExactT  = 32;
using PairLauncher         = void (*)(const Tensor&, const Weight&, const Weight&, Tensor&, Tensor&,
                              cudaStream_t);

struct W8PairExactTRows {
    static constexpr int kOutputRowsPerCta = kRowsPerCta;

    __device__ __forceinline__ int weight_row(int output_row0, int local_row) const {
        return output_row0 + (local_row & (kRowsPerCta - 1)) +
               (local_row >= kRowsPerCta ? kRows : 0);
    }
};

struct W8PairExactTEpilogue {
    __nv_bfloat16* first;
    __nv_bfloat16* second;

    template <int ActiveCols>
    __device__ __forceinline__ void store(int row, float (&projected)[ActiveCols]) const {
        constexpr unsigned kPairMask = 0x0000ffffu;
        const int lane               = static_cast<int>(threadIdx.x) & 31;
        const int output_row         = row - lane + (lane & (kRowsPerCta - 1));
#pragma unroll
        for (int token = 0; token < ActiveCols; ++token) {
            const float second_value =
                __shfl_sync(kPairMask, projected[token], (lane & (kRowsPerCta - 1)) + kRowsPerCta);
            if (lane < kRowsPerCta) {
                const std::int64_t offset = static_cast<std::int64_t>(token) * kRows + output_row;
                first[offset]             = __float2bfloat16_rn(projected[token]);
                second[offset]            = __float2bfloat16_rn(second_value);
            }
        }
    }
};

template <int ActiveCols>
void launch_active_cols(const Tensor& x, const Weight& first_weight, const Weight& second_weight,
                        Tensor& first_out, Tensor& second_out, cudaStream_t stream) {
    constexpr int TileCols =
        ActiveCols <= 8 ? 8 : (ActiveCols <= 16 ? 16 : (ActiveCols <= 24 ? 24 : 32));
    using Geometry           = W8LinearGeometry<2 * kRows, kHidden>;
    using Schedule           = W8SmallTMmaDefaultSchedule<TileCols, ActiveCols>;
    const auto* first_codes  = static_cast<const std::uint8_t*>(first_weight.qdata);
    const auto* first_scales = static_cast<const std::uint8_t*>(first_weight.scales);
    if (static_cast<const std::uint8_t*>(second_weight.qdata) != first_codes + kRows * kHidden ||
        static_cast<const std::uint8_t*>(second_weight.scales) !=
            first_scales + kRows * (kHidden / 32) * 2) {
        throw std::invalid_argument("W8 exact pair requires adjacent K/V row views");
    }

    const W8ContiguousOutput ignored{static_cast<__nv_bfloat16*>(first_out.data), kRows};
    const W8PairExactTEpilogue epilogue{static_cast<__nv_bfloat16*>(first_out.data),
                                        static_cast<__nv_bfloat16*>(second_out.data)};
    w8_small_t_mma_kernel<Geometry, ActiveCols, Schedule, W8ContiguousOutput, W8PairExactTEpilogue,
                          W8PairExactTRows><<<kRows / kRowsPerCta, Schedule::kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), first_codes, first_scales, ignored, epilogue,
        W8PairExactTRows{});
}

template <std::size_t... Offsets>
constexpr auto make_launchers(std::index_sequence<Offsets...>) {
    return std::array<PairLauncher, sizeof...(Offsets)>{
        &launch_active_cols<kFirstExactT + static_cast<int>(Offsets)>...};
}

constexpr auto kLaunchers =
    make_launchers(std::make_index_sequence<kLastExactT - kFirstExactT + 1>{});

} // namespace

void w8_pair_splitk_exact_t_launch(const Tensor& x, const Weight& first_weight,
                                   const Weight& second_weight, Tensor& first_out,
                                   Tensor& second_out, cudaStream_t stream) {
    if (x.ne[0] != kHidden || x.ne[1] < kFirstExactT || x.ne[1] > kLastExactT ||
        first_out.ne[0] != kRows || first_out.ne[1] != x.ne[1] || second_out.ne[0] != kRows ||
        second_out.ne[1] != x.ne[1]) {
        throw std::invalid_argument("W8 exact pair requires [1024,2048] and T=2..32");
    }
    kLaunchers[x.ne[1] - kFirstExactT](x, first_weight, second_weight, first_out, second_out,
                                       stream);
    CUDA_CHECK(cudaGetLastError());
}

void w8_pair_splitk_medium_launch(W8PairScheduleId schedule, const Tensor& x,
                                  const Weight& first_weight, const Weight& second_weight,
                                  Tensor& first_out, Tensor& second_out, cudaStream_t stream) {
    if (x.ne[0] != kHidden || x.ne[1] < 33 || first_out.ne[0] != kRows ||
        first_out.ne[1] != x.ne[1] || second_out.ne[0] != kRows || second_out.ne[1] != x.ne[1]) {
        throw std::invalid_argument("W8 medium pair requires [1024,2048] and T>=33");
    }
    (void)schedule;
    std::int32_t offset = 0;
    while (offset < x.ne[1]) {
        const std::int32_t count = std::min<std::int32_t>(kLastExactT, x.ne[1] - offset);
        const Tensor x_slice = x.slice(1, offset, count);
        Tensor first_slice   = first_out.slice(1, offset, count);
        Tensor second_slice  = second_out.slice(1, offset, count);
        if (count == 1) {
            w8_pair_decode_r16_launch(x_slice, first_weight, second_weight, first_slice,
                                      second_slice, stream);
        } else {
            w8_pair_splitk_exact_t_launch(x_slice, first_weight, second_weight, first_slice,
                                          second_slice, stream);
        }
        offset += count;
    }
}

} // namespace ninfer::ops::detail
