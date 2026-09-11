#pragma once

// Opt-in full-chain tracing for the KV-cache path: planned mode geometry,
// append/prompt/decode dispatch selection, and snapshot kv_flags save/restore.
//
//   NINFER_KV_TRACE non-empty enables every "[kv-trace] <point>: key=value"
//   line on stderr (each line flushed before returning). When unset or empty
//   the call sites reduce to one static branch and print nothing.
//
// kv_trace_once emits each distinct 64-bit key at most once per process, so
// per-step call sites (append/prompt/decode dispatch) print one line the
// first time a configuration is seen and are a static branch afterwards.

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <unordered_set>

namespace ninfer {

[[nodiscard]] inline bool kv_trace_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_KV_TRACE");
        return value != nullptr && value[0] != '\0';
    }();
    return enabled;
}

inline void kv_trace(const char* point, const char* format, ...) {
    if (!kv_trace_enabled()) { return; }
    std::fprintf(stderr, "[kv-trace] %s: ", point);
    std::va_list args;
    va_start(args, format);
    std::vfprintf(stderr, format, args);
    va_end(args);
    std::fputc('\n', stderr);
    std::fflush(stderr);
}

[[nodiscard]] inline bool kv_trace_once(std::uint64_t key) {
    if (!kv_trace_enabled()) { return false; }
    static std::unordered_set<std::uint64_t> seen;
    if (seen.size() >= 128) { seen.clear(); }
    return seen.insert(key).second;
}

} // namespace ninfer
