// Regression test for the KvCacheStorage -> PagedKVCache mode-flag table
// (src/core/kv_cache_mode.h), the same table the qwen3_6 layout planner
// (layouts_impl.h) feeds into decoder_state.cpp::plan_cache. A mode whose
// derived flags violate the plan_cache / append / attention validator
// invariants (e.g. k6_bit without e8_lattice, which surfaced as the runtime
// throw "K6-bit keys require packed E8 lattice keys and packed rotated V")
// is caught here at build time instead of on first use.
//
// Host-only: no CUDA and no GPU required.

#include "core/kv_cache_mode.h"
#include <ninfer/types.h>

#include <cstdio>

namespace {

using Flags = ninfer::kv_cache_mode::Flags;

int g_failures = 0;

void fail(const char* mode, const char* what) {
    std::printf("FAIL %s: %s\n", mode, what);
    ++g_failures;
}

void expect(const char* mode, bool ok, const char* what) {
    if (!ok) fail(mode, what);
}

// Every storage mode with a full expected table. The switch has no default so
// a new KvCacheStorage value missing here is a -Wswitch build error.
[[nodiscard]] Flags expected_flags(ninfer::KvCacheStorage storage) {
    switch (storage) {
        case ninfer::KvCacheStorage::BFloat16:
        case ninfer::KvCacheStorage::Int8Group64:
        case ninfer::KvCacheStorage::Fp8E4M3Row256:
            return {};
        case ninfer::KvCacheStorage::RotatedInt8KeyInt4ValueGroup64: {
            Flags f;
            f.packed_v = f.rotate_k = f.rotate_v = true;
            return f;
        }
        case ninfer::KvCacheStorage::RotatedInt4KeyInt4ValueGroup64: {
            Flags f;
            f.packed_v = f.rotate_k = f.rotate_v = f.packed_k = true;
            return f;
        }
        case ninfer::KvCacheStorage::RK4V4E8: {
            Flags f;
            f.packed_v = f.rotate_k = f.rotate_v = f.packed_k = f.e8_lattice = true;
            return f;
        }
        case ninfer::KvCacheStorage::RK2V4E8: {
            Flags f;
            f.packed_v = f.rotate_k = f.rotate_v = true;
            f.e8_root = true;
            return f;
        }
        case ninfer::KvCacheStorage::RK6V4E8: {
            Flags f;
            f.packed_v = f.rotate_k = f.rotate_v = f.packed_k = f.e8_lattice = f.k6_bit = true;
            return f;
        }
    }
    return {};  // Unreachable: the switch covers every enumerator.
}

// Invariants plan_cache (decoder_state.cpp) and the append / attention
// validators (kv_cache_append.cpp, causal_softmax_attention.cpp) enforce;
// derived flags that break them throw at first engine use.
void check_invariants(const char* mode, const Flags& f) {
    expect(mode, !(f.rotate_v && !f.packed_v), "rotated V without packed V");
    expect(mode, !(f.e8_lattice && !f.packed_k), "E8 lattice without packed K");
    expect(mode,
           !(f.k6_bit && !(f.packed_k && f.e8_lattice && f.packed_v && f.rotate_k && f.rotate_v &&
                           !f.e8_root)),
           "K6-bit keys without packed E8 lattice keys and packed rotated V");
    expect(mode, !(f.e8_root && (f.k6_bit || f.e8_lattice)), "E8 root combined with lattice/K6");
    expect(mode, !(f.packed_k && f.e8_root), "packed K combined with E8 root");
    expect(mode,
           !((f.packed_k || f.e8_root || f.k6_bit) && !f.rotate_k),
           "packed key plane without key rotation");
}

const struct {
    ninfer::KvCacheStorage storage;
    const char* name;
} kModes[] = {
    {ninfer::KvCacheStorage::BFloat16, "bf16"},
    {ninfer::KvCacheStorage::Int8Group64, "int8"},
    {ninfer::KvCacheStorage::RotatedInt8KeyInt4ValueGroup64, "rk8v4"},
    {ninfer::KvCacheStorage::RotatedInt4KeyInt4ValueGroup64, "rk4v4"},
    {ninfer::KvCacheStorage::RK4V4E8, "rk4v4-e8"},
    {ninfer::KvCacheStorage::RK2V4E8, "rk2v4-e8"},
    {ninfer::KvCacheStorage::RK6V4E8, "rk6v4-e8"},
    {ninfer::KvCacheStorage::Fp8E4M3Row256, "fp8"},
};

}  // namespace

int main() {
    for (const auto& mode : kModes) {
        const Flags actual = ninfer::kv_cache_mode::flags_for(mode.storage);
        const Flags want = expected_flags(mode.storage);
        expect(mode.name, actual.packed_v == want.packed_v, "packed_v mismatch");
        expect(mode.name, actual.rotate_k == want.rotate_k, "rotate_k mismatch");
        expect(mode.name, actual.rotate_v == want.rotate_v, "rotate_v mismatch");
        expect(mode.name, actual.packed_k == want.packed_k, "packed_k mismatch");
        expect(mode.name, actual.e8_lattice == want.e8_lattice, "e8_lattice mismatch");
        expect(mode.name, actual.e8_root == want.e8_root, "e8_root mismatch");
        expect(mode.name, actual.k6_bit == want.k6_bit, "k6_bit mismatch");
        check_invariants(mode.name, actual);
    }
    if (g_failures != 0) {
        std::printf("kv_mode: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("kv_mode: all %zu mode tables and invariants OK\n", sizeof(kModes) / sizeof(kModes[0]));
    return 0;
}
