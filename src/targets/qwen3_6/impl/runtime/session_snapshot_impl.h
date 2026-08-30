#include "targets/qwen3_6/impl/runtime/instance.h"
#include "targets/qwen3_6/impl/runtime/program.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

// Retained-session snapshot format (target-private).
//
// A snapshot is the complete host image of one catalogued continuation: the resident prefix
// (ledger + identity), the paged Text/backend KV payload in logical page order, and the
// linear-attention state image at the endpoint. Byte order is the host's (x86 little-endian);
// a snapshot binds to the exact weights identity and KV configuration, so cross-endian
// portability is intentionally out of scope.
//
// Versions 1 and 2 described the pre-reconciliation lane-retained format (v2 appended the
// turn-checkpoint ring). Both are rejected by this build: the physical KV layout and the
// state model changed with the upstream reconciliation, so old files cannot be re-landed.
// Version 3 is the continuation-catalog format. Its serialization is PENDING: the save and
// restore entry points below throw until the reimplementation on the upstream state-image +
// logical-KV model lands. The parked reader/writer helpers stay for that work.

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {
namespace {

constexpr char kSessionSnapshotMagic[8]         = {'N', 'I', 'N', 'F', 'S', 'E', 'S', '1'};
constexpr std::uint32_t kSessionSnapshotVersion = 3;

constexpr std::uint32_t kKvFlagPackedV   = 1U << 0;
constexpr std::uint32_t kKvFlagRotateK   = 1U << 1;
constexpr std::uint32_t kKvFlagRotateV   = 1U << 2;
constexpr std::uint32_t kKvFlagPackedK   = 1U << 3;
constexpr std::uint32_t kKvFlagE8Lattice = 1U << 4;
constexpr std::uint32_t kKvFlagE8Root    = 1U << 5;

class SnapshotWriter {
public:
    explicit SnapshotWriter(std::vector<std::uint8_t>& out) : out_(out) {}

    void bytes(const void* data, std::size_t count) {
        const auto* begin = static_cast<const std::uint8_t*>(data);
        out_.insert(out_.end(), begin, begin + count);
    }

    template <class T>
    void pod(T value) {
        static_assert(std::is_trivially_copyable_v<T>);
        bytes(&value, sizeof(T));
    }

    // Reserves a device-payload region and returns its offset; the caller fills it with
    // cudaMemcpyAsync once the full host image is sized (the vector no longer reallocates).
    std::size_t reserve_payload(std::size_t count) {
        const std::size_t offset = out_.size();
        out_.resize(out_.size() + count);
        return offset;
    }

private:
    std::vector<std::uint8_t>& out_;
};

class SnapshotReader {
public:
    explicit SnapshotReader(std::span<const std::uint8_t> data) : data_(data) {}

    void bytes(void* out, std::size_t count) {
        if (count > data_.size() - cursor_) {
            throw std::invalid_argument("session snapshot is truncated");
        }
        std::memcpy(out, data_.data() + cursor_, count);
        cursor_ += count;
    }

    template <class T>
    [[nodiscard]] T pod() {
        static_assert(std::is_trivially_copyable_v<T>);
        T value{};
        bytes(&value, sizeof(T));
        return value;
    }

    // Borrows a device-payload region without copying; valid for the snapshot's lifetime.
    [[nodiscard]] const std::uint8_t* payload(std::size_t count) {
        if (count > data_.size() - cursor_) {
            throw std::invalid_argument("session snapshot is truncated");
        }
        const std::uint8_t* region = data_.data() + cursor_;
        cursor_ += count;
        return region;
    }

    [[nodiscard]] std::size_t remaining() const noexcept { return data_.size() - cursor_; }

private:
    std::span<const std::uint8_t> data_;
    std::size_t cursor_ = 0;
};

template <class T>
[[maybe_unused]] void write_vector(SnapshotWriter& writer, const std::vector<T>& values) {
    static_assert(std::is_trivially_copyable_v<T>);
    writer.pod<std::uint64_t>(values.size());
    writer.bytes(values.data(), values.size() * sizeof(T));
}

template <class T>
[[maybe_unused]] std::vector<T> read_vector(SnapshotReader& reader, std::size_t maximum_count,
                                            const char* label) {
    const std::uint64_t count = reader.pod<std::uint64_t>();
    if (count > maximum_count) {
        throw std::invalid_argument(std::string("session snapshot ") + label +
                                    " count is out of range");
    }
    std::vector<T> values(static_cast<std::size_t>(count));
    reader.bytes(values.data(), values.size() * sizeof(T));
    return values;
}

// Session identity: FNV-1a 64 over the resident ledger's token bytes, rendered as 16 hex
// chars. Deterministic across processes on one endianness, which snapshot compatibility
// already requires. The shared prefix form lives in program_impl.h so checkpoint digests
// hash identically.
std::string ledger_digest(const std::vector<TokenId>& ledger) {
    return ledger_prefix_digest(std::span<const TokenId>(ledger.data(), ledger.size()));
}

constexpr const char* kPersistencePendingMessage =
    "session persistence is pending reimplementation on the upstream context-cache model "
    "(state images + logical KV); save/restore is unavailable in this build";

} // namespace

std::uint32_t
ProgramImplCore::continuation_depth(const ContinuationHandle& continuation) const noexcept {
    if (!valid_continuation(continuation)) { return 0; }
    const SequenceState& sequence = continuation_states[ContractAccess::index(continuation)];
    return static_cast<std::uint32_t>(sequence.ledger.size());
}

std::string ProgramImplCore::continuation_digest(const ContinuationHandle& continuation) const {
    if (!valid_continuation(continuation)) { return {}; }
    const SequenceState& sequence = continuation_states[ContractAccess::index(continuation)];
    return ledger_digest(sequence.ledger);
}

std::vector<SlotCheckpoint>
ProgramImplCore::continuation_checkpoints(const ContinuationHandle& continuation) const {
    if (!valid_continuation(continuation)) { return {}; }
    const SequenceState& sequence = continuation_states[ContractAccess::index(continuation)];
    const std::uint32_t depth     = static_cast<std::uint32_t>(sequence.ledger.size());

    std::vector<std::uint32_t> frontiers;
    frontiers.reserve(sequence.long_anchors.size() + 2U);
    for (const LongAnchorCheckpoint& anchor : sequence.long_anchors) {
        frontiers.push_back(anchor.frontier);
    }
    if (sequence.rewrite_checkpoint.valid) {
        frontiers.push_back(sequence.rewrite_checkpoint.frontier);
    }
    if (sequence.endpoint_valid) { frontiers.push_back(sequence.ledger_frontier); }
    std::sort(frontiers.begin(), frontiers.end());
    frontiers.erase(std::unique(frontiers.begin(), frontiers.end()), frontiers.end());

    std::vector<SlotCheckpoint> out;
    out.reserve(frontiers.size());
    for (const std::uint32_t frontier : frontiers) {
        if (frontier == 0 || frontier > depth) { continue; }
        out.push_back(SlotCheckpoint{
            frontier, ledger_prefix_digest(
                          std::span<const TokenId>(sequence.ledger.data(), frontier))});
    }
    return out;
}

qwen3_6::RetainedSessionSnapshot
ProgramImplCore::save_continuation(const ContinuationHandle& continuation,
                                   std::string_view model_binding) {
    if (!valid_continuation(continuation)) {
        throw std::invalid_argument("continuation holds no retained session");
    }
    if (model_binding.size() > 4096) {
        throw std::invalid_argument("session snapshot model binding is too long");
    }
    throw std::logic_error(kPersistencePendingMessage);
}

ContinuationHandle
ProgramImplCore::restore_continuation(std::span<const std::uint8_t> snapshot,
                                      std::string_view model_binding) {
    SnapshotReader reader(snapshot);
    char magic[sizeof(kSessionSnapshotMagic)] = {};
    reader.bytes(magic, sizeof(magic));
    if (std::memcmp(magic, kSessionSnapshotMagic, sizeof(magic)) != 0) {
        throw std::invalid_argument("file is not a session snapshot");
    }
    const std::uint32_t version = reader.pod<std::uint32_t>();
    if (version < kSessionSnapshotVersion) {
        throw std::invalid_argument(
            "session snapshot predates the context-cache reconciliation and cannot be restored");
    }
    if (version != kSessionSnapshotVersion) {
        throw std::invalid_argument("session snapshot version is unsupported");
    }
    (void)model_binding;
    throw std::logic_error(kPersistencePendingMessage);
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS
