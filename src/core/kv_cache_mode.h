#pragma once
// Host-only single source of truth mapping KvCacheStorage to the PagedKVCache
// mode flags consumed by the qwen3_6 layout planner, the append kernels, and
// the attention decode path. Production code (layouts_impl.h) and the
// ninfer_kv_mode_test regression test both read this table, so a mode added to
// KvCacheStorage that misses a flag here fails the test at build time.

#include <ninfer/types.h>

namespace ninfer::kv_cache_mode {

struct Flags {
    bool packed_v   = false;  // V plane stores packed 4-bit codes (U8).
    bool rotate_k   = false;  // Keys are H64-rotated before quantization.
    bool rotate_v   = false;  // Values are H64-rotated; output is inverse-rotated.
    bool packed_k   = false;  // Key plane stores packed 4-bit codes (U8).
    bool e8_lattice = false;  // 4-bit key codes come from an E8 lattice projection.
    bool e8_root    = false;  // 2-bit E8-root key codes (240 roots, 2 bytes per 8-d).
    bool k6_bit     = false;  // 6-bit E8 lattice key codes, four packed per 24-bit word.
};

[[nodiscard]] inline Flags flags_for(KvCacheStorage storage) {
    Flags f;
    switch (storage) {
        case KvCacheStorage::BFloat16:
        case KvCacheStorage::Int8Group64:
        case KvCacheStorage::Fp8E4M3Row256:
            // Unpacked modes: no mode flags.
            break;
        case KvCacheStorage::RotatedInt8KeyInt4ValueGroup64:  // rk8v4
            f.packed_v = f.rotate_k = f.rotate_v = true;
            break;
        case KvCacheStorage::RotatedInt4KeyInt4ValueGroup64:  // rk4v4
            f.packed_v = f.rotate_k = f.rotate_v = f.packed_k = true;
            break;
        case KvCacheStorage::RK4V4E8:
            f.packed_v = f.rotate_k = f.rotate_v = f.packed_k = f.e8_lattice = true;
            break;
        case KvCacheStorage::RK2V4E8:
            f.packed_v = f.rotate_k = f.rotate_v = true;
            f.e8_root = true;
            break;
        case KvCacheStorage::RK6V4E8:
            f.packed_v = f.rotate_k = f.rotate_v = f.packed_k = f.e8_lattice = f.k6_bit = true;
            break;
    }
    return f;
}

}  // namespace ninfer::kv_cache_mode
