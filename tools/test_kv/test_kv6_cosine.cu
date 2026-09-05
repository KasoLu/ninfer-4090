// rk6v4e8 key-quantization cosine benchmark (codec level, GPU).
//
// Reproduces the exact production K/V encode arithmetic from
// src/ops/kv_cache/append/kernel.cuh (H64 rotation, FP16-RNE group scale,
// E8 lattice projection + rintf half-coset approximation, i4/i6 codes) on
// synthetic standard-normal groups, and reports the cosine similarity
// between the rotated original and the codec reconstruction for each KV
// mode. Cosine is invariant under the orthogonal H64 rotation, so these
// numbers bound the per-group fidelity of the shipped codec.
//
// Modes (K side; V side is packed i4 in every rk* mode):
//   rk8v4    : K i8   codes, scale = FP16-RNE(absmax/127)
//   rk4v4    : K i4   codes, scale = FP16-RNE(absmax/7)
//   rk4v4e8  : K E8-lattice projection + rintf -> i4 codes, /7 scale
//   rk6v4e8  : K E8-lattice projection + rintf -> i6 codes, /31 scale
//
// Exit code 0 = ran (no pass/fail threshold; inspect printed numbers).
// Exit code 77 = no CUDA device (test harness SKIP convention).

#include "kv_cache_cosine_bench.cuh"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>

#include <cuda_fp16.h>

namespace {

constexpr int kGroups  = 4096;  // 32 warps per block, 128 blocks
constexpr int kThreads = 256;

struct GroupStats {
    float mean_cos;
    float min_cos;
    float p99_cos;
};

using kv_cache_cosine_bench::KeyMode;

const KeyMode kKeyModes[] = {
    {"rk8v4 (i8 K)", 127, false, false},
    {"rk4v4 (i4 K)", 7, false, false},
    {"rk4v4-e8 (E8 i4 K)", 7, true, false},
    {"rk6v4-e8 (E8 i6 K)", 31, true, true},
};

void check_cuda(const char* what) {
    const auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(err));
        std::exit(1);
    }
}

GroupStats collect_stats(float* d_cos, int groups) {
    GroupStats stats{};
    float* h_cos = new float[groups];
    cudaMemcpy(h_cos, d_cos, groups * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaGetLastError() != cudaSuccess) {
        std::fprintf(stderr, "cudaMemcpy d2h failed: %s\n", cudaGetErrorString(cudaGetLastError()));
        std::exit(1);
    }
    float sum = 0.0f;
    float minv = 1.0f;
    for (int i = 0; i < groups; ++i) {
        sum += h_cos[i];
        if (h_cos[i] < minv) { minv = h_cos[i]; }
    }
    std::sort(h_cos, h_cos + groups);
    stats.mean_cos = sum / groups;
    stats.min_cos = minv;
    stats.p99_cos = h_cos[static_cast<int>(0.99f * (groups - 1))];
    delete[] h_cos;
    return stats;
}

}  // namespace

int main() {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::fprintf(stderr, "no CUDA device available (run with --gpus all)\n");
        return 77;
    }
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    std::printf("device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    std::printf("groups: %d (standard-normal 64-d vectors, H64 rotation, production codec "
                "arithmetic)\n",
                kGroups);
    std::printf("%-22s %12s %12s %12s   %12s %12s %12s\n", "mode", "K mean", "K min", "K p99",
                "V mean", "V min", "V p99");

    // Host-side standard-normal source, deterministic seed.
    const int dims = kGroups * 64;
    std::mt19937 rng(20260207u);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    float* h_src = new float[dims];
    for (int i = 0; i < dims; ++i) { h_src[i] = normal(rng); }
    float* d_src = nullptr;
    float* d_cos = nullptr;
    cudaMalloc(&d_src, dims * sizeof(float));
    cudaMalloc(&d_cos, kGroups * sizeof(float));
    cudaMemcpy(d_src, h_src, dims * sizeof(float), cudaMemcpyHostToDevice);
    check_cuda("cudaMemcpy h2d");

    const int blocks = kGroups / 32;
    for (const KeyMode& mode : kKeyModes) {
        kv_cache_cosine_bench::bench_key_mode_kernel<<<blocks, kThreads>>>(d_src, d_cos, kGroups,
                                                                           mode);
        if (cudaDeviceSynchronize() != cudaSuccess) {
            std::fprintf(stderr, "key kernel failed: %s\n", cudaGetErrorString(cudaGetLastError()));
            std::exit(1);
        }
        GroupStats ks = collect_stats(d_cos, kGroups);
        kv_cache_cosine_bench::bench_value_mode_kernel<<<blocks, kThreads>>>(d_src, d_cos, kGroups);
        if (cudaDeviceSynchronize() != cudaSuccess) {
            std::fprintf(stderr, "value kernel failed: %s\n",
                         cudaGetErrorString(cudaGetLastError()));
            std::exit(1);
        }
        GroupStats vs = collect_stats(d_cos, kGroups);
        std::printf("%-22s %12.6f %12.6f %12.6f   %12.6f %12.6f %12.6f\n", mode.name, ks.mean_cos,
                    ks.min_cos, ks.p99_cos, vs.mean_cos, vs.min_cos, vs.p99_cos);
    }

    std::printf("\nnote: bf16 K/V is the reference (cosine 1.0 by construction).\n");
    cudaFree(d_src);
    cudaFree(d_cos);
    delete[] h_src;
    return 0;
}
