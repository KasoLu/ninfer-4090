#include "targets/qwen3_6/impl/frontend/tool_call_parser.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace ninfer::targets::qwen3_6::frontend_internal {
#ifdef NINFER_TOOL_CALL_TRACE
// Diagnostic hook: a host tool that compiles this translation unit with NINFER_TOOL_CALL_TRACE
// defined provides this sink. Every NINFER_TC_TRACE call expands to a no-op otherwise.
void tool_call_trace(const char* stage, std::string_view detail);
#define NINFER_TC_TRACE(stage, ...)                                                                \
    ::ninfer::targets::qwen3_6::frontend_internal::tool_call_trace(stage, __VA_ARGS__)
#else
#define NINFER_TC_TRACE(...) ((void)0)
#endif

namespace {

using Json = nlohmann::json;

#ifdef NINFER_TOOL_CALL_TRACE
const char* encoding_name(const ToolArgumentTypeContracts::Parameter* contract) {
    if (contract == nullptr) { return "legacy"; }
    return contract->encoding == ToolArgumentTypeContracts::Encoding::String ? "String" : "Json";
}

std::string trace_snippet(std::string_view text, std::size_t max_length) {
    const std::size_t limit = std::min(max_length, text.size());
    std::string result;
    result.reserve(limit + 4);
    for (std::size_t index = 0; index < limit; ++index) {
        const char byte = text[index];
        if (byte == '\n') {
            result += "\\n";
        } else if (byte == '\r') {
            result += "\\r";
        } else if (byte == '\t') {
            result += "\\t";
        } else {
            result += byte;
        }
    }
    if (text.size() > limit) { result += "..."; }
    return result;
}

std::string trace_json_error(const std::string& text) {
    try {
        [[maybe_unused]] const Json parsed = Json::parse(text);
        return "parse ok";
    } catch (const std::exception& error) {
        return error.what();
    }
}
#endif

std::string trim_ascii(std::string_view text) {
    std::size_t begin = 0;
    while (begin < text.size() && std::isspace(static_cast<unsigned char>(text[begin])) != 0) {
        ++begin;
    }
    std::size_t end = text.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1])) != 0) { --end; }
    return std::string(text.substr(begin, end - begin));
}

std::string rtrim_ascii(std::string_view text) {
    std::size_t end = text.size();
    while (end != 0 && std::isspace(static_cast<unsigned char>(text[end - 1])) != 0) { --end; }
    return std::string(text.substr(0, end));
}

void skip_ws(std::string_view text, std::size_t& pos) {
    while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos])) != 0) { ++pos; }
}

bool starts_with_at(std::string_view text, std::size_t pos, std::string_view prefix) {
    return pos <= text.size() && text.substr(pos, prefix.size()) == prefix;
}

bool has_non_whitespace(std::string_view text) {
    for (const unsigned char c : text) {
        if (std::isspace(c) == 0) { return true; }
    }
    return false;
}

// Qwen's closing tags are line-anchored: the template requires tags to start a line. Prefer the
// first close tag that starts a line so literal tag text embedded inside a parameter value does
// not terminate the value early, and fall back to the first plain occurrence for inputs that do
// not follow the line contract. Returns the close tag's offset, or npos.
std::size_t find_closing_tag(std::string_view text, std::size_t pos, std::string_view close) {
    std::string anchored;
    anchored.reserve(close.size() + 1);
    anchored.push_back('\n');
    anchored.append(close);
    const std::size_t line_anchored = text.find(anchored, pos);
    if (line_anchored != std::string_view::npos) { return line_anchored + 1; }
    return text.find(close, pos);
}

bool valid_function_name(std::string_view name, std::size_t max_name_length) {
    if (name.empty() || name.size() > max_name_length) { return false; }
    for (const unsigned char c : name) {
        if (std::isalnum(c) == 0 && c != '_' && c != '-') { return false; }
    }
    return true;
}

bool is_json_schema_type(std::string_view type) {
    return type == "string" || type == "integer" || type == "number" || type == "boolean" ||
           type == "object" || type == "array" || type == "null";
}

bool explicit_parameter_encoding(const Json& property,
                                 ToolArgumentTypeContracts::Encoding& encoding) {
    if (!property.is_object()) { return false; }
    const auto type = property.find("type");
    if (type == property.end()) { return false; }

    if (type->is_string()) {
        const std::string& name = type->get_ref<const std::string&>();
        if (!is_json_schema_type(name)) { return false; }
        encoding = name == "string" ? ToolArgumentTypeContracts::Encoding::String
                                    : ToolArgumentTypeContracts::Encoding::Json;
        return true;
    }
    if (!type->is_array() || type->empty()) { return false; }
    bool admits_string = false;
    for (const Json& member : *type) {
        if (!member.is_string()) { return false; }
        const std::string& name = member.get_ref<const std::string&>();
        if (!is_json_schema_type(name)) { return false; }
        admits_string |= name == "string";
    }
    encoding = admits_string ? ToolArgumentTypeContracts::Encoding::String
                             : ToolArgumentTypeContracts::Encoding::Json;
    return true;
}

ToolArgumentTypeContracts::Tool compile_tool_contract(const Json& definition) {
    ToolArgumentTypeContracts::Tool contract;
    if (!definition.is_object()) { return contract; }
    const auto function = definition.find("function");
    if (function == definition.end() || !function->is_object()) { return contract; }
    const auto name = function->find("name");
    if (name == function->end() || !name->is_string()) { return contract; }
    contract.name = name->get<std::string>();

    const auto schema = function->find("parameters");
    if (schema == function->end() || !schema->is_object()) { return contract; }
    const auto properties = schema->find("properties");
    if (properties == schema->end() || !properties->is_object()) { return contract; }

    for (const auto& [parameter_name, property] : properties->items()) {
        ToolArgumentTypeContracts::Encoding encoding;
        if (explicit_parameter_encoding(property, encoding)) {
            contract.parameters.push_back({parameter_name, encoding});
        }
    }
    return contract;
}

bool same_contract(const ToolArgumentTypeContracts::Tool& lhs,
                   const ToolArgumentTypeContracts::Tool& rhs) {
    if (lhs.parameters.size() != rhs.parameters.size()) { return false; }
    for (std::size_t i = 0; i < lhs.parameters.size(); ++i) {
        if (lhs.parameters[i].name != rhs.parameters[i].name ||
            lhs.parameters[i].encoding != rhs.parameters[i].encoding) {
            return false;
        }
    }
    return true;
}

void append_tool_contract(ToolArgumentTypeContracts& contracts, const Json& definition) {
    ToolArgumentTypeContracts::Tool compiled = compile_tool_contract(definition);
    if (compiled.name.empty()) { return; }
    const auto existing =
        std::find_if(contracts.tools.begin(), contracts.tools.end(),
                     [&](const auto& tool) { return tool.name == compiled.name; });
    if (existing == contracts.tools.end()) {
        contracts.tools.push_back(std::move(compiled));
        return;
    }
    if (existing->unambiguous && !same_contract(*existing, compiled)) {
        existing->parameters.clear();
        existing->unambiguous = false;
    }
}

const ToolArgumentTypeContracts::Parameter*
find_parameter_contract(const ToolArgumentTypeContracts& contracts, std::string_view tool_name,
                        std::string_view parameter_name) {
    const auto tool =
        std::find_if(contracts.tools.begin(), contracts.tools.end(),
                     [&](const auto& candidate) { return candidate.name == tool_name; });
    if (tool == contracts.tools.end() || !tool->unambiguous) { return nullptr; }
    const auto parameter =
        std::find_if(tool->parameters.begin(), tool->parameters.end(),
                     [&](const auto& candidate) { return candidate.name == parameter_name; });
    return parameter == tool->parameters.end() ? nullptr : &*parameter;
}

bool declares_tool(const ToolArgumentTypeContracts& contracts, std::string_view tool_name) {
    if (!contracts.enforce_declared_names) { return true; }
    return std::any_of(contracts.tools.begin(), contracts.tools.end(),
                       [&](const auto& tool) { return tool.name == tool_name; });
}

std::string_view remove_parameter_framing_newlines(std::string_view text) {
    std::size_t begin = 0;
    std::size_t end   = text.size();
    if (text.starts_with("\r\n")) {
        begin = 2;
    } else if (text.starts_with('\n')) {
        begin = 1;
    }
    if (end >= begin + 2 && text.substr(end - 2, 2) == "\r\n") {
        end -= 2;
    } else if (end > begin && text[end - 1] == '\n') {
        --end;
    }
    return text.substr(begin, end - begin);
}

bool parse_parameter(std::string_view inner, std::size_t& pos, Json& args,
                     std::string_view tool_name, const ToolArgumentTypeContracts& contracts) {
    constexpr std::string_view kParamOpen  = "<parameter=";
    constexpr std::string_view kParamClose = "</parameter>";
    if (!starts_with_at(inner, pos, kParamOpen)) {
        NINFER_TC_TRACE("parameter", "FAIL: expected '<parameter=' at offset " +
                                         std::to_string(pos) + " of " +
                                         std::to_string(inner.size()) + ", found '" +
                                         trace_snippet(inner.substr(pos, 32), 32) + "'");
        return false;
    }
    const std::size_t name_begin = pos + kParamOpen.size();
    const std::size_t name_end   = inner.find('>', name_begin);
    if (name_end == std::string_view::npos || name_end == name_begin) {
        NINFER_TC_TRACE("parameter", "FAIL: '<parameter=' without a key or '>' at offset " +
                                         std::to_string(pos) + ", found '" +
                                         trace_snippet(inner.substr(pos, 32), 32) + "'");
        return false;
    }
    const std::string key       = std::string(inner.substr(name_begin, name_end - name_begin));
    pos                         = name_end + 1;
    const std::size_t value_end = find_closing_tag(inner, pos, kParamClose);
    if (value_end == std::string_view::npos) {
        NINFER_TC_TRACE("parameter", "FAIL: missing '</parameter>' for key '" + key +
                                         "' at offset " + std::to_string(pos));
        return false;
    }
    const std::string_view encoded_value = inner.substr(pos, value_end - pos);
    const ToolArgumentTypeContracts::Parameter* contract =
        find_parameter_contract(contracts, tool_name, key);
    NINFER_TC_TRACE("parameter", "key '" + key + "' value [" + std::to_string(pos) + ", " +
                                     std::to_string(value_end) + ") len " +
                                     std::to_string(encoded_value.size()) + ", encoding " +
                                     encoding_name(contract));
    if (contract == nullptr) {
        const std::string legacy_value = trim_ascii(encoded_value);
        Json parsed                    = Json::parse(legacy_value, nullptr, false);
        NINFER_TC_TRACE("parameter", "key '" + key + "': legacy trimmed len " +
                                         std::to_string(legacy_value.size()) + ", JSON " +
                                         (parsed.is_discarded() ? "discarded -> raw string"
                                                                : "decoded"));
        args[key] = parsed.is_discarded() ? Json(legacy_value) : std::move(parsed);
    } else {
        const std::string value(remove_parameter_framing_newlines(encoded_value));
        if (contract->encoding == ToolArgumentTypeContracts::Encoding::String) {
            NINFER_TC_TRACE("parameter",
                            "key '" + key + "': String len " + std::to_string(value.size()) +
                                " (framing strip removed " +
                                std::to_string(encoded_value.size() - value.size()) + " bytes)");
            args[key] = value;
        } else {
            Json parsed = Json::parse(value, nullptr, false);
            if (parsed.is_discarded()) {
                // Best-effort: a value that is not valid JSON degrades to its raw text instead of
                // rejecting the call, so the client harness can inspect or reject it.
                NINFER_TC_TRACE("parameter", "key '" + key +
                                                 "': JSON decode discarded -> raw string (len " +
                                                 std::to_string(value.size()) + "): " +
                                                 trace_json_error(value) + "; value '" +
                                                 trace_snippet(value, 160) + "'");
                args[key] = value;
            } else {
                NINFER_TC_TRACE("parameter", "key '" + key + "': Json decoded (len " +
                                                 std::to_string(value.size()) + ")");
                args[key] = std::move(parsed);
            }
        }
    }
    pos = value_end + kParamClose.size();
    return true;
}

bool parse_one_tool_call(std::string_view block, std::size_t max_name_length,
                         const ToolArgumentTypeContracts& contracts, GeneratedToolCall& out) {
    constexpr std::string_view kFunctionOpen  = "<function=";
    constexpr std::string_view kFunctionClose = "</function>";
    std::size_t pos                           = 0;
    skip_ws(block, pos);
    if (!starts_with_at(block, pos, kFunctionOpen)) {
        NINFER_TC_TRACE("function", "FAIL: expected '<function=' at offset " +
                                        std::to_string(pos) + " of " +
                                        std::to_string(block.size()) + ", found '" +
                                        trace_snippet(block.substr(pos, 32), 32) + "'");
        return false;
    }
    const std::size_t name_begin = pos + kFunctionOpen.size();
    const std::size_t name_end   = block.find('>', name_begin);
    if (name_end == std::string_view::npos || name_end == name_begin) {
        NINFER_TC_TRACE("function", "FAIL: '<function=' without a name or '>' at offset " +
                                        std::to_string(pos));
        return false;
    }
    const std::string name = std::string(block.substr(name_begin, name_end - name_begin));
    const bool valid_name  = valid_function_name(name, max_name_length);
    const bool declared    = declares_tool(contracts, name);
    NINFER_TC_TRACE("function", "name '" + name + "' (len " + std::to_string(name.size()) +
                                    ") valid=" + (valid_name ? "1" : "0") + " declared=" +
                                    (declared ? "1" : "0"));
    if (!valid_name || !declared) { return false; }
    pos = name_end + 1;

    const std::size_t function_end = find_closing_tag(block, pos, kFunctionClose);
    if (function_end == std::string_view::npos) {
        NINFER_TC_TRACE("function", "FAIL: missing '</function>' after offset " +
                                        std::to_string(pos));
        return false;
    }
    const std::string_view params = block.substr(pos, function_end - pos);
    NINFER_TC_TRACE("function", "body [" + std::to_string(pos) + ", " +
                                    std::to_string(function_end) + ") len " +
                                    std::to_string(params.size()));
    Json args             = Json::object();
    std::size_t param_pos = 0;
    for (;;) {
        skip_ws(params, param_pos);
        if (param_pos >= params.size()) { break; }
        if (!parse_parameter(params, param_pos, args, name, contracts)) { return false; }
    }

    pos = function_end + kFunctionClose.size();
    skip_ws(block, pos);
    if (pos != block.size()) {
        NINFER_TC_TRACE("function", "FAIL: trailing non-whitespace after '</function>' at " +
                                        std::to_string(pos) + ": '" +
                                        trace_snippet(block.substr(pos, 32), 32) + "'");
        return false;
    }

    out.name           = name;
    out.arguments_json = args.dump();
    NINFER_TC_TRACE("function", "OK: '" + name + "' arguments '" +
                                    trace_snippet(out.arguments_json, 160) + "'");
    return true;
}

ParsedToolCallOutput fallback(const std::string& text) {
    ParsedToolCallOutput out;
    out.content = text;
    return out;
}

} // namespace

std::shared_ptr<const ToolCallOutputContract>
build_tool_call_output_contract(std::span<const std::string> tool_jsons, bool enabled) {
    if (!enabled) {
        NINFER_TC_TRACE("contract", "disabled");
        return {};
    }
    auto contract                                   = std::make_shared<ToolCallOutputContract>();
    contract->argument_types.enforce_declared_names = true;
    contract->argument_types.tools.reserve(tool_jsons.size());
    for (const std::string& tool_json : tool_jsons) {
        const Json definition = Json::parse(tool_json, nullptr, false);
        NINFER_TC_TRACE("contract", std::string("tool_json ") +
                                        (definition.is_discarded() ? "discarded" : "parsed"));
        if (!definition.is_discarded()) {
            append_tool_contract(contract->argument_types, definition);
        }
    }
#ifdef NINFER_TOOL_CALL_TRACE
    for (const auto& tool : contract->argument_types.tools) {
        std::string parameters;
        for (const auto& parameter : tool.parameters) {
            if (!parameters.empty()) { parameters += ", "; }
            parameters += parameter.name + "=" + encoding_name(&parameter);
        }
        NINFER_TC_TRACE("contract", "tool '" + tool.name + "' unambiguous=" +
                                        (tool.unambiguous ? "1" : "0") + " params=[" +
                                        parameters + "]");
    }
#endif
    return contract;
}

ParsedToolCallOutput parse_qwen_tool_call_output(const std::string& text,
                                                 std::size_t max_tool_name_length,
                                                 const ToolArgumentTypeContracts& contracts) {
    constexpr std::string_view kToolOpen  = "<tool_call>";
    constexpr std::string_view kToolClose = "</tool_call>";

    const std::size_t first = text.find(kToolOpen);
    if (first == std::string::npos) {
        NINFER_TC_TRACE("output", "FAIL: no '<tool_call>' marker in " +
                                      std::to_string(text.size()) + " bytes -> fallback");
        return fallback(text);
    }
    NINFER_TC_TRACE("output", "first '<tool_call>' at offset " + std::to_string(first) + " of " +
                                  std::to_string(text.size()));

    ParsedToolCallOutput out;
    out.content = rtrim_ascii(std::string_view(text).substr(0, first));
    NINFER_TC_TRACE("output", "content prefix len " + std::to_string(out.content.size()));

    std::size_t pos = first;
    while (pos < text.size()) {
        const std::size_t open = text.find(kToolOpen, pos);
        if (open == std::string::npos) {
            // Ordinary text after the last block stays content.
            const std::string_view tail = std::string_view(text).substr(pos);
            if (has_non_whitespace(tail)) { out.content.append(tail); }
            break;
        }
        if (open > pos) {
            // Text between blocks is preserved as content instead of discarding every parsed call.
            const std::string_view gap = std::string_view(text).substr(pos, open - pos);
            if (has_non_whitespace(gap)) { out.content.append(gap); }
        }
        const std::size_t inner_begin = open + kToolOpen.size();
        const std::size_t close       = find_closing_tag(text, inner_begin, kToolClose);
        if (close == std::string::npos) {
            // An unterminated block cannot be parsed; keep its raw text as content.
            out.content.append(text, open, text.size() - open);
            NINFER_TC_TRACE("output", "unterminated block at offset " + std::to_string(open) +
                                          " -> degraded to content");
            break;
        }
        NINFER_TC_TRACE("output", "block #" + std::to_string(out.tool_calls.size() + 1) +
                                      " span [" + std::to_string(open) + ", " +
                                      std::to_string(close) + "), inner len " +
                                      std::to_string(close - inner_begin));
        GeneratedToolCall call;
        if (parse_one_tool_call(std::string_view(text).substr(inner_begin, close - inner_begin),
                                max_tool_name_length, contracts, call)) {
            out.tool_calls.push_back(std::move(call));
        } else {
            // Per-block isolation: a rejected block degrades to its raw text while the remaining
            // blocks still produce tool calls.
            out.content.append(text, open, close + kToolClose.size() - open);
            NINFER_TC_TRACE("output", "block at offset " + std::to_string(open) +
                                          " rejected -> degraded to content");
        }
        pos = close + kToolClose.size();
    }

    if (out.tool_calls.empty()) { return fallback(text); }
    out.is_tool_call_response = true;
    NINFER_TC_TRACE("output", "OK: " + std::to_string(out.tool_calls.size()) +
                                  " tool call(s), content len " +
                                  std::to_string(out.content.size()));
    return out;
}

ToolCallOutputDecoder::ToolCallOutputDecoder(std::shared_ptr<const ToolCallOutputContract> contract,
                                             std::size_t max_tool_name_length)
    : contract_(std::move(contract)), max_tool_name_length_(max_tool_name_length) {}

std::string ToolCallOutputDecoder::feed(std::string_view text) {
    if (finished_) { throw std::logic_error("tool-call output decoder is already finished"); }
    if (text.empty()) { return {}; }
    if (!contract_) { return std::string(text); }
    if (saw_tool_marker_) {
        tool_region_.append(text);
        return {};
    }

    constexpr std::string_view kToolOpen = "<tool_call>";
    std::string visible;
    for (std::size_t index = 0; index < text.size(); ++index) {
        const char byte = text[index];
        if (marker_prefix_bytes_ != 0) {
            if (byte == kToolOpen[marker_prefix_bytes_]) {
                ++marker_prefix_bytes_;
                if (marker_prefix_bytes_ == kToolOpen.size()) {
                    tool_region_ = std::move(trailing_whitespace_);
                    trailing_whitespace_.clear();
                    tool_region_.append(kToolOpen);
                    tool_region_.append(text.substr(index + 1));
                    marker_prefix_bytes_ = 0;
                    saw_tool_marker_     = true;
                    break;
                }
                continue;
            }
            visible.append(trailing_whitespace_);
            trailing_whitespace_.clear();
            visible.append(kToolOpen.substr(0, marker_prefix_bytes_));
            marker_prefix_bytes_ = 0;
        }

        if (byte == kToolOpen.front()) {
            marker_prefix_bytes_ = 1;
        } else if (std::isspace(static_cast<unsigned char>(byte)) != 0) {
            trailing_whitespace_.push_back(byte);
        } else {
            visible.append(trailing_whitespace_);
            trailing_whitespace_.clear();
            visible.push_back(byte);
        }
    }
    return visible;
}

ToolCallOutputDecoder::Terminal ToolCallOutputDecoder::finish() {
    if (finished_) { throw std::logic_error("tool-call output decoder is already finished"); }
    finished_ = true;
    if (!contract_) { return {}; }

    ParsedToolCallOutput parsed =
        parse_qwen_tool_call_output(tool_region_, max_tool_name_length_, contract_->argument_types);
    if (saw_tool_marker_ && parsed.is_tool_call_response) {
        trailing_whitespace_.clear();
        tool_region_.clear();
        marker_prefix_bytes_ = 0;
        // Content carries any block that degraded to raw text; the remaining blocks stay calls.
        return Terminal{.content    = std::move(parsed.content),
                        .tool_calls = std::move(parsed.tool_calls)};
    }

    constexpr std::string_view kToolOpen = "<tool_call>";
    std::string tail                     = std::move(trailing_whitespace_);
    tail.append(kToolOpen.substr(0, marker_prefix_bytes_));
    marker_prefix_bytes_ = 0;
    tail += tool_region_;
    tool_region_.clear();
    return Terminal{.content = std::move(tail), .tool_calls = {}};
}

} // namespace ninfer::targets::qwen3_6::frontend_internal
