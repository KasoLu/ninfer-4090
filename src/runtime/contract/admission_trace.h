#pragma once

// V3 R-V3-3: opt-in full-chain admission tracing.
//
//   NINFER_ADMISSION_TRACE  non-empty enables every "[admission-trace] <point>: key=value"
//                           line on stderr (each line flushed before returning). When unset
//                           or empty the call sites reduce to one static branch and print
//                           nothing.
//   NINFER_PREFILL_TRIPWIRE non-empty enables the prefill-session tripwire in
//                           advance_prefill (program_impl.h): every new prefill session
//                           (staged.cursor == staged.base) prints a "TRIPWIRE:" line, and
//                           two consecutive FULL re-prefills (base==0 both times) with
//                           different prompts abort the run. A single base==0 round (e.g.
//                           the legal post-compression full round) never triggers.

#include <cstdarg>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

namespace ninfer::runtime {

[[nodiscard]] inline bool admission_trace_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_ADMISSION_TRACE");
        return value != nullptr && value[0] != '\0';
    }();
    return enabled;
}

[[nodiscard]] inline bool prefill_tripwire_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_PREFILL_TRIPWIRE");
        return value != nullptr && value[0] != '\0';
    }();
    return enabled;
}

inline void admission_trace(const char* point, const char* format, ...) {
    if (!admission_trace_enabled()) { return; }
    std::fprintf(stderr, "[admission-trace] %s: ", point);
    std::va_list args;
    va_start(args, format);
    std::vfprintf(stderr, format, args);
    va_end(args);
    std::fputc('\n', stderr);
    std::fflush(stderr);
}

[[nodiscard]] inline const char* prefix_reuse_path_name(ninfer::PrefixReusePath path) {
    switch (path) {
        case ninfer::PrefixReusePath::Root:
            return "root";
        case ninfer::PrefixReusePath::PrivateEndpoint:
            return "private_endpoint";
        case ninfer::PrefixReusePath::PrivateTurnClosure:
            return "private_turn_closure";
        case ninfer::PrefixReusePath::PrivateResponseReplay:
            return "private_response_replay";
        case ninfer::PrefixReusePath::PrivateLongAnchor:
            return "private_long_anchor";
        case ninfer::PrefixReusePath::SharedStablePrefix:
            return "shared_stable_prefix";
    }
    return "unknown";
}

} // namespace ninfer::runtime
