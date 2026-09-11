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
// Exit code 0 = passed the production i6 pack/unpack round-trip regression and
// ran the bench (no cosine threshold; inspect printed numbers); non-zero =
// round-trip mismatch or CUDA failure.
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

// ---------------------------------------------------------------------------
// Production i6 codec round-trip regression (pack quad -> unpack i6x16).
// Hard-fail gate before the cosine bench: catches byte-layout slips in
// kv_cache_unpack_i6x16. The 2026-09 bug took quad2's high byte from byte11
// instead of byte8, corrupting dims 10/11 of every 16-dim block and
// degrading rk6v4e8 PPL ~10x vs rk4v4-e8; the cosine bench (which decodes
// with its own reference arithmetic) could not see it.
// ---------------------------------------------------------------------------
__global__ void kv6_roundtrip_check_kernel(const std::uint8_t* d_codes, std::uint8_t* d_bad,
                                           int blocks) {
    const int b = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (b >= blocks) { return; }
    const std::uint8_t* src = d_codes + static_cast<std::size_t>(b) * 16;
    std::uint8_t packed[12];
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        ninfer::ops::kv_cache_pack_i6_quad(&src[4 * j], &packed[3 * j]);
    }
    std::int8_t dec[16];
    ninfer::ops::kv_cache_unpack_i6x16(packed, dec);
    int bad = 0;
    #pragma unroll
    for (int m = 0; m < 16; ++m) {
        const int expected = (static_cast<int>(src[m]) ^ 32) - 32;
        if (static_cast<int>(dec[m]) != expected) { ++bad; }
    }
    d_bad[b] = static_cast<std::uint8_t>(bad);
}

bool run_kv6_roundtrip_check() {
    constexpr int kSweepBlocks = 16 * 64;  // every code value at every position
    constexpr int kRandBlocks  = 4096;
    const int blocks = kSweepBlocks + kRandBlocks;
    std::uint8_t* h_codes = new std::uint8_t[static_cast<std::size_t>(blocks) * 16];
    for (int p = 0; p < 16; ++p) {
        for (int v = 0; v < 64; ++v) {
            std::uint8_t* row = h_codes + static_cast<std::size_t>(p * 64 + v) * 16;
            for (int m = 0; m < 16; ++m) {
                row[m] = static_cast<std::uint8_t>((v + m) & 0x3Fu);
            }
        }
    }
    std::mt19937 rng(0x616e31u);
    std::uniform_int_distribution<int> dist(0, 63);
    for (int b = kSweepBlocks; b < blocks; ++b) {
        std::uint8_t* row = h_codes + static_cast<std::size_t>(b) * 16;
        for (int m = 0; m < 16; ++m) { row[m] = static_cast<std::uint8_t>(dist(rng)); }
    }
    std::uint8_t* d_codes = nullptr;
    std::uint8_t* d_bad = nullptr;
    if (cudaMalloc(&d_codes, static_cast<std::size_t>(blocks) * 16) != cudaSuccess ||
        cudaMalloc(&d_bad, blocks) != cudaSuccess) {
        std::fprintf(stderr, "kv6 round-trip: cudaMalloc failed: %s\n",
                     cudaGetErrorString(cudaGetLastError()));
        return false;
    }
    if (cudaMemcpy(d_codes, h_codes, static_cast<std::size_t>(blocks) * 16,
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        std::fprintf(stderr, "kv6 round-trip h2d failed: %s\n",
                     cudaGetErrorString(cudaGetLastError()));
        return false;
    }
    const int threads = 128;
    kv6_roundtrip_check_kernel<<<(blocks + threads - 1) / threads, threads>>>(d_codes, d_bad,
                                                                               blocks);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::fprintf(stderr, "kv6 round-trip kernel failed: %s\n",
                     cudaGetErrorString(cudaGetLastError()));
        return false;
    }
    std::uint8_t* h_bad = new std::uint8_t[blocks];
    if (cudaMemcpy(h_bad, d_bad, blocks, cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::fprintf(stderr, "kv6 round-trip d2h failed: %s\n",
                     cudaGetErrorString(cudaGetLastError()));
        return false;
    }
    int bad_blocks = 0;
    int bad_codes = 0;
    int first_block = -1;
    for (int b = 0; b < blocks; ++b) {
        if (h_bad[b] != 0) {
            if (first_block < 0) { first_block = b; }
            ++bad_blocks;
            bad_codes += h_bad[b];
        }
    }
    cudaFree(d_codes);
    cudaFree(d_bad);
    delete[] h_bad;
    delete[] h_codes;
    if (bad_blocks > 0) {
        std::fprintf(stderr,
                    "FAIL: kv6 i6 pack/unpack round-trip mismatch: %d/%d blocks, %d codes "
                    "(first block %d); kv_cache_unpack_i6x16 byte layout is broken\n",
                    bad_blocks, blocks, bad_codes, first_block);
        return false;
    }
    std::printf("i6 pack/unpack round-trip: %d blocks, 0 mismatches (sweep + random)\n", blocks);
    return true;
}

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

    if (!run_kv6_roundtrip_check()) { return 1; }

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
