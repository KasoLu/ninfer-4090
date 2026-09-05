// ktrace.c — LD_PRELOAD kernel-launch tracer for the 4090 rk6v4-e8 hang.
//
// ninfer is linked against libcudart.so.13 and its `>>>` launches resolve to
// the *internal* entry `__cudaLaunchKernel` (confirmed via `nm -D`), plus
// `cudaLaunchKernelExC` for kernels with large parameter buffers. Those exact
// symbol names are interposed here (C linkage); sync sites (stream and event
// sync) are logged so that "S..-IN with no S..-OUT" pins the spinning kernel.
// Each launch also dumps a dladdr-symbolicated backtrace so the host call
// site is visible even from a stripped binary. Line-buffered => a SIGKILL of a
// hung run still keeps the log.
//
// Build (inside the GPU container, CPU-only):
//   g++ -shared -fPIC -O2 -o /tmp/ktrace.so ktrace.c -ldl
// Run:
//   LD_PRELOAD=/tmp/ktrace.so NINFER_KTRACE=/tmp/kt_k6.log timeout -s KILL 60
//     ./build/apps/ninfer <model> --kv-dtype rk6v4-e8 ... --greedy

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <dlfcn.h>
#include <execinfo.h>
#include <cstdlib>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <unistd.h>

typedef int cudaError_t;
typedef void* cudaStream_t;
typedef void* cudaEvent_t;
struct dim3_t {
    unsigned x;
    unsigned y;
    unsigned z;
};
// cudaLaunchConfig_t mirrors: {void* gridDim; void* blockDim; void* dynamicSmemBytes;
// cudaStream_t stream; void* attrs; size_t numAttrs;}
struct launch_config_t {
    void* gridDim;
    void* blockDim;
    void* dynamicSmemBytes;
    cudaStream_t stream;
    void* attrs;
    size_t numAttrs;
};

static uint64_t g_seq = 0;
static FILE* g_out = NULL;

__attribute__((constructor)) static void ktrace_init(void) {
    const char* path = getenv("NINFER_KTRACE");
    g_out = path ? fopen(path, "w") : stderr;
    if (!g_out) { g_out = stderr; }
    setvbuf(g_out, NULL, _IOLBF, 0);
    fprintf(g_out, "== ktrace init pid=%d\n", (int)getpid());
}

static void dump_bt(void) {
    void* bt[12];
    int n = backtrace(bt, 12);
    for (int i = 1; i < n; ++i) {
        Dl_info info;
        if (dladdr(bt[i], &info) && info.dli_sname) {
            uintptr_t off = (uintptr_t)bt[i] - (uintptr_t)info.dli_saddr;
            fprintf(g_out, "    +%05zx %s  (%s)\n", off, info.dli_sname,
                    info.dli_fname ? info.dli_fname : "?");
        } else {
            fprintf(g_out, "    ?%p\n", bt[i]);
        }
    }
}

static void log_launch(const char* what, void* func, const dim3_t* grid, const dim3_t* block,
                       size_t smem, void* stream) {
    uint64_t seq = __atomic_add_fetch(&g_seq, 1, __ATOMIC_SEQ_CST);
    const dim3_t g = grid ? *grid : (dim3_t){0, 0, 0};
    const dim3_t b = block ? *block : (dim3_t){0, 0, 0};
    fprintf(g_out, "L%04llu %s func=%p grid=%ux%ux%u block=%ux%ux%u smem=%zu stream=%p\n",
            (unsigned long long)seq, what, func, g.x, g.y, g.z, b.x, b.y, b.z, smem, stream);
    dump_bt();
    fflush(g_out);
}

static void log_sync_in(const char* what, void* h) {
    uint64_t seq = __atomic_add_fetch(&g_seq, 1, __ATOMIC_SEQ_CST);
    fprintf(g_out, "S%04llu %s-IN h=%p\n", (unsigned long long)seq, what, h);
    fflush(g_out);
}

static void log_sync_out(const char* what, void* h, int rc) {
    fprintf(g_out, "S %s-OUT h=%p rc=%d\n", what, h, rc);
    fflush(g_out);
}

// ---- interposed entry points (exact symbol names from `nm -D ninfer`) -----

extern "C" {

// Internal `>>>` launch entry used by nvcc-generated code (CUDA 13).
cudaError_t __cudaLaunchKernel(const void* func, dim3_t grid, dim3_t block, void** args,
                               size_t smem, cudaStream_t s) {
    static cudaError_t (*real)(const void*, dim3_t, dim3_t, void**, size_t, cudaStream_t) = NULL;
    if (!real) { real = (decltype(real))dlsym(RTLD_NEXT, "__cudaLaunchKernel"); }
    if (!real) {
        fprintf(g_out, "FATAL: dlsym(__cudaLaunchKernel) failed; trace disabled\n");
        fflush(g_out);
        return 0;
    }
    log_launch("__cudaLaunchKernel", (void*)func, &grid, &block, smem, s);
    (void)args;
    return real(func, grid, block, args, smem, s);
}

cudaError_t cudaLaunchKernelExC(const struct launch_config_t* config, const void* func,
                                void** args) {
    static cudaError_t (*real)(const struct launch_config_t*, const void*, void**) = NULL;
    if (!real) { real = (decltype(real))dlsym(RTLD_NEXT, "cudaLaunchKernelExC"); }
    if (!real) {
        fprintf(g_out, "FATAL: dlsym(cudaLaunchKernelExC) failed; trace disabled\n");
        fflush(g_out);
        return 0;
    }
    size_t smem = 0;
    if (config && config->dynamicSmemBytes) { smem = *(size_t*)config->dynamicSmemBytes; }
    log_launch("cudaLaunchKernelExC", (void*)func, config ? (dim3_t*)config->gridDim : NULL,
               config ? (dim3_t*)config->blockDim : NULL, smem,
               config ? (void*)config->stream : NULL);
    (void)args;
    return real(config, func, args);
}

cudaError_t cudaStreamSynchronize(cudaStream_t s) {
    static cudaError_t (*real)(cudaStream_t) = NULL;
    if (!real) { real = (decltype(real))dlsym(RTLD_NEXT, "cudaStreamSynchronize"); }
    if (!real) {
        fprintf(g_out, "FATAL: dlsym(cudaStreamSynchronize) failed; trace disabled\n");
        fflush(g_out);
        return 0;
    }
    log_sync_in("cudaStreamSynchronize", s);
    int rc = real(s);
    log_sync_out("cudaStreamSynchronize", s, rc);
    return rc;
}

cudaError_t cudaEventSynchronize(cudaEvent_t e) {
    static cudaError_t (*real)(cudaEvent_t) = NULL;
    if (!real) { real = (decltype(real))dlsym(RTLD_NEXT, "cudaEventSynchronize"); }
    if (!real) {
        fprintf(g_out, "FATAL: dlsym(cudaEventSynchronize) failed; trace disabled\n");
        fflush(g_out);
        return 0;
    }
    log_sync_in("cudaEventSynchronize", (void*)e);
    int rc = real(e);
    log_sync_out("cudaEventSynchronize", (void*)e, rc);
    return rc;
}

} // extern "C"
