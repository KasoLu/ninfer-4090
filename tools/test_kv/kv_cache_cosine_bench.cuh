// Device-side replica of the production rotated KV encode arithmetic
// (src/ops/kv_cache/append/kernel.cuh), shared by the rk6v4e8 cosine
// benchmark (test_kv6_cosine.cu). One warp handles one 64-d group exactly
// like the append kernels: lane holds d0 = lane, d1 = lane + 32 of the
// group row.
#pragma once

#include "ops/common/warp.cuh"
#include "ops/kernel/e8_lattice.cuh"
#include "ops/kv_cache/int8_g64_codec.cuh"

#include <cuda_fp16.h>

#include <cmath>

namespace kv_cache_cosine_bench {
using ninfer::ops::e8_project_8d_warp;
using ninfer::ops::kv_cache_hadamard64;
using ninfer::ops::kv_cache_i4_quant_code;
using ninfer::ops::kv_cache_i6_code_from_int;
using ninfer::ops::kv_cache_unpack_i6;
using ninfer::ops::kv_cache_int8_quant_code;
using ninfer::ops::warp_max;
using ninfer::ops::warp_sum;

struct KeyMode {
    const char* name;
    int k_scale_div;  // 127 (rk8v4), 7 (rk4v4 / rk4v4-e8); 31 is selected by k6
    bool e8_lattice;  // E8 projection before rintf (rk4v4-e8 / rk6v4-e8)
    bool k6;          // 6-bit E8 lattice key codes (rk6v4-e8)
};

// K-side reconstruction: identical arithmetic to kv_cache_append_full_i8_kernel.
__device__ __forceinline__ void key_reconstruct(const float x0, const float x1, int lane, float k_abs,
                                                const KeyMode& mode, float& r0, float& r1) {
    if (mode.k6) {
        const __half ksh = __float2half_rn(k_abs > 0.0f ? k_abs / 31.0f : 0.0f);
        const float ks = __half2float(ksh);
        const float kinv = ks > 0.0f ? 1.0f / ks : 0.0f;
        float a0 = x0 * kinv;
        float a1 = x1 * kinv;
        e8_project_8d_warp(a0, a1, lane);
        // Same half-coset approximation as the production K6 branch: rintf
        // collapses the D8+0.5 coset; i6 codes store values in [-32, +31].
        const int q0 =
            static_cast<int>(kv_cache_unpack_i6(kv_cache_i6_code_from_int(static_cast<int>(rintf(a0)))));
        const int q1 =
            static_cast<int>(kv_cache_unpack_i6(kv_cache_i6_code_from_int(static_cast<int>(rintf(a1)))));
        r0 = static_cast<float>(q0) * ks;
        r1 = static_cast<float>(q1) * ks;
    } else if (mode.e8_lattice) {
        const __half ksh = __float2half_rn(k_abs > 0.0f ? k_abs / 7.0f : 0.0f);
        const float ks = __half2float(ksh);
        const float kinv = ks > 0.0f ? 1.0f / ks : 0.0f;
        float a0 = x0 * kinv;
        float a1 = x1 * kinv;
        e8_project_8d_warp(a0, a1, lane);
        // Production: rintf + clamp to [-8, +7] + packed i4 code.
        const int q0 = max(-8, min(7, static_cast<int>(rintf(a0))));
        const int q1 = max(-8, min(7, static_cast<int>(rintf(a1))));
        r0 = static_cast<float>(q0) * ks;
        r1 = static_cast<float>(q1) * ks;
    } else if (mode.k_scale_div == 7) {
        const __half ksh = __float2half_rn(k_abs > 0.0f ? k_abs / 7.0f : 0.0f);
        const float ks = __half2float(ksh);
        const float kinv = ks > 0.0f ? 1.0f / ks : 0.0f;
        r0 = static_cast<float>(kv_cache_i4_quant_code(x0, kinv)) * ks;
        r1 = static_cast<float>(kv_cache_i4_quant_code(x1, kinv)) * ks;
    } else {
        const __half ksh = __float2half_rn(k_abs > 0.0f ? k_abs / 127.0f : 0.0f);
        const float ks = __half2float(ksh);
        const float kinv = ks > 0.0f ? 1.0f / ks : 0.0f;
        r0 = static_cast<float>(kv_cache_int8_quant_code(x0, kinv)) * ks;
        r1 = static_cast<float>(kv_cache_int8_quant_code(x1, kinv)) * ks;
    }
}

// V-side reconstruction for the rk* family: packed i4 on a /7 group scale.
__device__ __forceinline__ void value_reconstruct(const float x0, const float x1, float v_abs, float& r0,
                                                  float& r1) {
    const __half vsh = __float2half_rn(v_abs > 0.0f ? v_abs / 7.0f : 0.0f);
    const float vs = __half2float(vsh);
    const float vinv = vs > 0.0f ? 1.0f / vs : 0.0f;
    r0 = static_cast<float>(kv_cache_i4_quant_code(x0, vinv)) * vs;
    r1 = static_cast<float>(kv_cache_i4_quant_code(x1, vinv)) * vs;
}

template <typename Reconstruct>
__device__ __forceinline__ void group_cosine(const float* src, int group, int lane, int tid,
                                             float* cos_out, const Reconstruct& reconstruct) {
    const float k0 = src[group * 64 + lane];
    const float k1 = src[group * 64 + lane + 32];
    float x0 = k0;
    float x1 = k1;
    kv_cache_hadamard64(x0, x1);  // production rotates per 64-d group

    const float abs_ = warp_max(fmaxf(fabsf(x0), fabsf(x1)));
    float r0 = 0.0f, r1 = 0.0f;
    reconstruct(x0, x1, lane, abs_, r0, r1);

    // Cosine in the rotated space equals the pre-rotation cosine because the
    // H64 rotation is orthogonal.
    const float dot = warp_sum(x0 * r0 + x1 * r1);
    const float n1 = warp_sum(x0 * x0 + x1 * x1);
    const float n2 = warp_sum(r0 * r0 + r1 * r1);
    if (lane == 0) {
        const float denom = sqrtf(n1) * sqrtf(n2);
        cos_out[tid / 32] = denom > 0.0f ? dot / denom : 1.0f;
    }
}

__global__ void bench_key_mode_kernel(const float* __restrict__ src, float* __restrict__ cos_out,
                                      int groups, KeyMode mode) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= groups * 32) { return; }
    const int lane = tid & 31;
    group_cosine(src, tid / 32, lane, tid, cos_out,
                 [&](const float x0, const float x1, int ln, float abs_, float& r0, float& r1) {
                     key_reconstruct(x0, x1, ln, abs_, mode, r0, r1);
                 });
}

__global__ void bench_value_mode_kernel(const float* __restrict__ src, float* __restrict__ cos_out,
                                        int groups) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= groups * 32) { return; }
    const int lane = tid & 31;
    group_cosine(src, tid / 32, lane, tid, cos_out,
                 [&](const float x0, const float x1, int, float abs_, float& r0, float& r1) {
                     value_reconstruct(x0, x1, abs_, r0, r1);
                 });
}

}  // namespace kv_cache_cosine_bench
