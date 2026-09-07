#pragma once

#include "ninfer/types.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::targets::qwen3_6::frontend_internal {

// Qwen's tool syntax carries each top-level argument as text between parameter tags. This
// contract records only whether an explicit JSON Schema type admits a string or requires JSON
// decoding. It intentionally does not perform full Schema validation.
struct ToolArgumentTypeContracts {
    enum class Encoding : std::uint8_t {
        Json,
        String,
    };

    struct Parameter {
        std::string name;
        Encoding encoding = Encoding::Json;
    };

    struct Tool {
        std::string name;
        std::vector<Parameter> parameters;
        bool unambiguous = true;
    };

    std::vector<Tool> tools;
    bool enforce_declared_names = false;
};

struct ToolCallOutputContract {
    ToolArgumentTypeContracts argument_types;
};

struct ParsedToolCallOutput {
    bool is_tool_call_response = false;
    std::string content;
    // On a successful response, bytes of the tool region that could not be attributed to a valid
    // tool block (narrative between or after blocks, or a whole malformed block) are restored
    // here instead of forcing a wholesale text fallback. Empty when every region byte belongs to
    // a valid call or is formatting whitespace.
    std::string residual;
    std::vector<GeneratedToolCall> tool_calls;
};

[[nodiscard]] std::shared_ptr<const ToolCallOutputContract>
build_tool_call_output_contract(std::span<const std::string> tool_jsons, bool enabled);

[[nodiscard]] ParsedToolCallOutput
parse_qwen_tool_call_output(const std::string& text, std::size_t max_tool_name_length,
                            const ToolArgumentTypeContracts& contracts);

// Incrementally publishes bytes that are provably outside a possible terminal Qwen tool-call
// suffix. At terminal time, valid calls are retained structurally; narrative that cannot be
// attributed to a valid block is restored as content, and wholly malformed output is restored
// verbatim.
class ToolCallOutputDecoder {
public:
    struct Terminal {
        std::string content;
        std::vector<GeneratedToolCall> tool_calls;
    };

    ToolCallOutputDecoder(std::shared_ptr<const ToolCallOutputContract> contract,
                          std::size_t max_tool_name_length);

    [[nodiscard]] std::string feed(std::string_view text);
    [[nodiscard]] Terminal finish();

private:
    std::shared_ptr<const ToolCallOutputContract> contract_;
    std::string trailing_whitespace_;
    std::string tool_region_;
    std::size_t marker_prefix_bytes_  = 0;
    std::size_t max_tool_name_length_ = 0;
    bool saw_tool_marker_             = false;
    bool finished_                    = false;
};

} // namespace ninfer::targets::qwen3_6::frontend_internal
