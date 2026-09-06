#pragma once
//
// gpu_specs.h — per-GPU hardware reference constants for the L1 op benchmarks.
//
// The benches report throughput as a percentage of a hardware roofline. A
// roofline depends on the specific card, so instead of hard-coding one card's
// numbers (the benches were originally calibrated on an RTX 5090), the specs
// are looked up by the runtime device name (cudaDeviceProp::name).
//
// This fork targets the RTX 4090 (sm_89). The RTX 5090 entry is kept so the
// same binaries can be re-pointed at either card for cross-GPU comparison.
//
// The 4090 sustained-read bandwidth is a placeholder (0.0) until
// tools/hbm_bandwidth_probe measures it on a 4090 and backfills the value.

#include <cuda_runtime.h>

#include <string_view>

namespace ninfer::bench {

struct GpuSpecs {
    const char* name            = "unknown";
    double dram_gbs             = 0.0;  // DRAM bandwidth roofline (GB/s)
    double sustained_read_gbs   = 0.0;  // measured sustained read bandwidth (GB/s)
    double dense_bf16_tflops    = 0.0;  // dense BF16 tensor-core throughput
    double fp8_fp16_acc_tflops  = 0.0;  // FP8 (e4m3) tensor-core, FP16 accumulate
    double fp8_fp32_acc_tflops  = 0.0;  // FP8 (e4m3) tensor-core, FP32 accumulate
};

// Match specs by a substring of the device name. Unknown cards fall back to
// the RTX 4090, which is this fork's build target.
inline const GpuSpecs& gpu_specs(std::string_view device_name) {
    // RTX 5090 (sm_120a, Blackwell): upstream measured values, kept for comparison.
    static const GpuSpecs kRtx5090{"RTX 5090", 1792.0, 1674.5, 209.5, 838.0, 419.0};
    // RTX 4090 (sm_89, Ada): DRAM 1008 GB/s, dense BF16 165.2 TFLOPs, dense FP8
    // 330.3 TFLOPs. Ada quotes a single dense FP8 tensor figure, so both
    // accumulator variants share it. sustained_read_gbs stays 0.0 until the
    // hbm_bandwidth_probe measurement on a 4090 is backfilled.
    static const GpuSpecs kRtx4090{"RTX 4090", 1008.0, 0.0, 165.2, 330.3, 330.3};
    if (device_name.find("5090") != std::string_view::npos) {
        return kRtx5090;
    }
    return kRtx4090;
}

// Specs for the current CUDA device, resolved once and cached for the process.
// Falls back to the RTX 4090 if the device cannot be queried (no GPU present).
inline const GpuSpecs& active_gpu_specs() {
    static const GpuSpecs resolved = [] {
        int device             = 0;
        cudaDeviceProp props{};
        if (cudaGetDevice(&device) == cudaSuccess &&
            cudaGetDeviceProperties(&props, device) == cudaSuccess) {
            return gpu_specs(props.name);
        }
        return gpu_specs("RTX 4090");
    }();
    return resolved;
}

}  // namespace ninfer::bench
