#include "core/device.h"
#include "ops/linear/w8/w8_small_t_mma.cuh"
#include "ops/linear/w8/w8_launch.h"

#include <array>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace ninfer::ops::detail {

void w8_rowsplit_decode_r16_launch(const Tensor& x, const Weight& w, Tensor& out,
                                   cudaStream_t stream);

namespace {

constexpr int kRows           = 2048;
constexpr int kHidden         = 16384;
constexpr int kRowsPerCta     = 16;
constexpr int kFirstExactCols = 2;
constexpr int kLastExactCols  = 48;
using ExactTLauncher          = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

template <int ActiveCols>
void launch_active_cols(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    constexpr int TileCols  = ActiveCols <= 8    ? 8
                              : ActiveCols <= 16 ? 16
                              : ActiveCols <= 24 ? 24
                              : ActiveCols <= 32 ? 32
                              : ActiveCols <= 40 ? 40
                                                 : 48;
    constexpr int KWarps    = 4;
    constexpr int MinBlocks = 2;
    constexpr auto ScaleAccess =
        ActiveCols > 4 ? W8SmallTMmaScaleAccess::Shared : W8SmallTMmaScaleAccess::Direct;
    constexpr auto ActivationCache = ActiveCols <= 36 || ActiveCols == 48 ? Cache::cg : Cache::ca;
    using Geometry                 = W8LinearGeometry<kRows, kHidden>;
    using Schedule = W8SmallTMmaSchedule<KWarps, TileCols, MinBlocks, ScaleAccess, ActivationCache>;
    static_assert((kRows % kRowsPerCta) == 0);
    const W8ContiguousOutput output{static_cast<__nv_bfloat16*>(out.data), kRows};
    w8_small_t_mma_kernel<Geometry, ActiveCols, Schedule>
        <<<kRows / kRowsPerCta, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), output);
}

template <std::size_t... Offsets>
constexpr auto make_launchers(std::index_sequence<Offsets...>) {
    return std::array<ExactTLauncher, sizeof...(Offsets)>{
        &launch_active_cols<kFirstExactCols + static_cast<int>(Offsets)>...};
}

constexpr auto kLaunchers =
    make_launchers(std::make_index_sequence<kLastExactCols - kFirstExactCols + 1>{});

void require_problem(const Tensor& x, const Weight& w, const Tensor& out) {
    if (x.ne[0] != kHidden || out.ne[0] != kRows || out.ne[1] != x.ne[1] || w.n != kRows ||
        w.k != kHidden || w.padded_shape[1] != kHidden) {
        throw std::invalid_argument("W8 exact-T split-K requires [2048,16384]");
    }
}

} // namespace

void launch_w8_exact_t_splitk(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    require_problem(x, w, out);
    if (x.ne[1] < kFirstExactCols || x.ne[1] > kLastExactCols) {
        throw std::invalid_argument("W8 exact-T split-K requires T=2..48");
    }
    kLaunchers[x.ne[1] - kFirstExactCols](x, w, out, stream);
    CUDA_CHECK(cudaGetLastError());
}

void launch_w8_exact_t_composite(const Tensor& x, const Weight& w, Tensor& out,
                                 cudaStream_t stream) {
    require_problem(x, w, out);
    if (x.ne[1] < 33) {
        throw std::invalid_argument("W8 exact-T composite requires T>=33");
    }

    std::int32_t offset = 0;
    while (x.ne[1] - offset >= 32) {
        const Tensor x_slice = x.slice(1, offset, 32);
        Tensor out_slice     = out.slice(1, offset, 32);
        launch_w8_exact_t_splitk(x_slice, w, out_slice, stream);
        offset += 32;
    }
    const std::int32_t tail = x.ne[1] - offset;
    if (tail == 1) {
        const Tensor x_slice = x.slice(1, offset, 1);
        Tensor out_slice     = out.slice(1, offset, 1);
        w8_rowsplit_decode_r16_launch(x_slice, w, out_slice, stream);
    } else if (tail >= 2) {
        const Tensor x_slice = x.slice(1, offset, tail);
        Tensor out_slice     = out.slice(1, offset, tail);
        launch_w8_exact_t_splitk(x_slice, w, out_slice, stream);
    }
}

void launch_w8_dflash_medium(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    require_problem(x, w, out);
    const int t = x.ne[1];
    if (t < 49 || t > 128) {
        throw std::invalid_argument("W8 DFlash medium route requires T=49..128");
    }

    launch_w8_exact_t_composite(x, w, out, stream);
    CUDA_CHECK(cudaGetLastError());
}

void launch_w8_medium_splitk_c144(const Tensor& x, const Weight& w, Tensor& out,
                                  cudaStream_t stream) {
    launch_w8_exact_t_composite(x, w, out, stream);
}

} // namespace ninfer::ops::detail
