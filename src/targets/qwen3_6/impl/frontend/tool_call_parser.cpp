#include "targets/qwen3_6/impl/frontend/tool_call_parser.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace ninfer::targets::qwen3_6::frontend_internal {
namespace {

using Json = nlohmann::json;

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

// Model-generated JSON frequently contains bare line-break bytes inside string literals (a common
// slip on long values: some lines are escaped as \n, others are emitted as raw newlines). A strict
// Json::parse rejects raw control bytes inside strings, which used to drop the whole tool call.
// This rewrites ONLY raw control bytes inside string literals to their standard escapes; structure
// outside strings and every other byte are preserved verbatim.
std::string escape_raw_controls_in_strings(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    bool in_string = false;
    std::size_t i  = 0;
    while (i < text.size()) {
        const char c = text[i];
        if (!in_string) {
            out.push_back(c);
            if (c == '"') { in_string = true; }
            ++i;
            continue;
        }
        if (c == '\\') {
            out.push_back(c);
            // An escaped byte (\", \\, ...) is copied verbatim - but only when the escaped byte is
            // not itself a raw control character (a backslash does not license a bare newline).
            if (i + 1 < text.size() &&
                static_cast<unsigned char>(text[i + 1]) >= 0x20) {
                out.push_back(text[i + 1]);
                i += 2;
                continue;
            }
            ++i;
            continue;
        }
        if (c == '"') {
            out.push_back(c);
            in_string = false;
            ++i;
            continue;
        }
        const unsigned char byte = static_cast<unsigned char>(c);
        if (byte < 0x20) { // raw control byte inside a string literal
            switch (c) {
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            default: {
                static constexpr char kHex[] = "0123456789abcdef";
                out += "\\u00";
                out.push_back(kHex[(byte >> 4) & 0xF]);
                out.push_back(kHex[byte & 0xF]);
                break;
            }
            }
            ++i;
            continue;
        }
        out.push_back(c);
        ++i;
    }
    return out;
}

// Second common slip: the model drops the closing brace(s) of the innermost object(s) when the
// value ends inside an array - e.g. {"insert_after": {...}} closed as "}]" instead of "}}]".
// Inserts the missing '}' closers before a ']' whenever the innermost open container is an object.
// Bounded: only synthesizes object closers ahead of a bracket; anything else that still fails to
// parse is left to the caller's drop path.
std::string close_unclosed_objects_before_brackets(std::string_view text) {
    std::string out;
    out.reserve(text.size() + 4);
    std::vector<char> containers;
    bool in_string = false;
    std::size_t i  = 0;
    while (i < text.size()) {
        const char c = text[i];
        if (in_string) {
            out.push_back(c);
            if (c == '\\') {
                if (i + 1 < text.size()) { out.push_back(text[i + 1]); }
                i += 2;
                continue;
            }
            if (c == '"') { in_string = false; }
            ++i;
            continue;
        }
        if (c == '"') {
            in_string = true;
            out.push_back(c);
            ++i;
            continue;
        }
        if (c == '{' || c == '[') {
            containers.push_back(c);
            out.push_back(c);
            ++i;
            continue;
        }
        if (c == '}' || c == ']') {
            while (!containers.empty() && containers.back() == '{' && c == ']') {
                out.push_back('}'); // the model closed the array but skipped the object closer(s)
                containers.pop_back();
            }
            if (!containers.empty() &&
                (containers.back() == '{' ? c == '}' : c == ']')) {
                containers.pop_back();
                out.push_back(c);
                ++i;
                continue;
            }
            if (containers.empty() && c == '}') { // stray close outside any container
                out.push_back(c);
                ++i;
                continue;
            }
            return out; // mismatched closer that is not repairable: keep what we have
        }
        out.push_back(c);
        ++i;
    }
    return out;
}

bool parse_parameter(std::string_view inner, std::size_t& pos, Json& args,
                     std::string_view tool_name, const ToolArgumentTypeContracts& contracts) {
    constexpr std::string_view kParamOpen  = "<parameter=";
    constexpr std::string_view kParamClose = "</parameter>";
    if (!starts_with_at(inner, pos, kParamOpen)) { return false; }
    const std::size_t name_begin = pos + kParamOpen.size();
    const std::size_t name_end   = inner.find('>', name_begin);
    if (name_end == std::string_view::npos || name_end == name_begin) { return false; }
    const std::string key       = std::string(inner.substr(name_begin, name_end - name_begin));
    pos                         = name_end + 1;
    const std::size_t value_end = inner.find(kParamClose, pos);
    if (value_end == std::string_view::npos) { return false; }
    const std::string_view encoded_value = inner.substr(pos, value_end - pos);
    const ToolArgumentTypeContracts::Parameter* contract =
        find_parameter_contract(contracts, tool_name, key);
    if (contract == nullptr) {
        const std::string legacy_value = trim_ascii(encoded_value);
        Json parsed                    = Json::parse(legacy_value, nullptr, false);
        args[key] = parsed.is_discarded() ? Json(legacy_value) : std::move(parsed);
    } else {
        std::string value(remove_parameter_framing_newlines(encoded_value));
        if (contract->encoding == ToolArgumentTypeContracts::Encoding::String) {
            args[key] = std::move(value);
        } else {
            Json parsed = Json::parse(value, nullptr, false);
            if (parsed.is_discarded()) {
                // Retry with bounded repairs for the two slips observed in model output: raw
                // control bytes inside string literals, and object closers dropped ahead of a
                // closing bracket. Keeps the declared JSON contract while tolerating both.
                value  = escape_raw_controls_in_strings(value);
                parsed = Json::parse(value, nullptr, false);
                if (parsed.is_discarded()) {
                    value  = close_unclosed_objects_before_brackets(value);
                    parsed = Json::parse(value, nullptr, false);
                }
                if (parsed.is_discarded()) { return false; }
            }
            args[key] = std::move(parsed);
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
    if (!starts_with_at(block, pos, kFunctionOpen)) { return false; }
    const std::size_t name_begin = pos + kFunctionOpen.size();
    const std::size_t name_end   = block.find('>', name_begin);
    if (name_end == std::string_view::npos || name_end == name_begin) { return false; }
    const std::string name = std::string(block.substr(name_begin, name_end - name_begin));
    if (!valid_function_name(name, max_name_length) || !declares_tool(contracts, name)) {
        return false;
    }
    pos = name_end + 1;

    const std::size_t function_end = block.find(kFunctionClose, pos);
    if (function_end == std::string_view::npos) { return false; }
    const std::string_view params = block.substr(pos, function_end - pos);
    Json args                     = Json::object();
    std::size_t param_pos         = 0;
    for (;;) {
        skip_ws(params, param_pos);
        if (param_pos >= params.size()) { break; }
        if (!parse_parameter(params, param_pos, args, name, contracts)) { return false; }
    }

    pos = function_end + kFunctionClose.size();
    skip_ws(block, pos);
    if (pos != block.size()) { return false; }

    out.name           = name;
    out.arguments_json = args.dump();
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
    if (!enabled) { return {}; }
    auto contract                                   = std::make_shared<ToolCallOutputContract>();
    contract->argument_types.enforce_declared_names = true;
    contract->argument_types.tools.reserve(tool_jsons.size());
    for (const std::string& tool_json : tool_jsons) {
        const Json definition = Json::parse(tool_json, nullptr, false);
        if (!definition.is_discarded()) {
            append_tool_contract(contract->argument_types, definition);
        }
    }
    return contract;
}

ParsedToolCallOutput parse_qwen_tool_call_output(const std::string& text,
                                                 std::size_t max_tool_name_length,
                                                 const ToolArgumentTypeContracts& contracts) {
    constexpr std::string_view kToolOpen  = "<tool_call>";
    constexpr std::string_view kToolClose = "</tool_call>";

    const std::size_t first = text.find(kToolOpen);
    if (first == std::string::npos) { return fallback(text); }

    ParsedToolCallOutput out;
    out.content = rtrim_ascii(std::string_view(text).substr(0, first));

    // Salvage loop: every well-formed block is retained structurally; bytes that cannot be
    // attributed to a valid block (narrative between or after blocks, or a whole malformed
    // block) are restored as residual content instead of discarding the good calls.
    const auto not_space = [](unsigned char byte) { return std::isspace(byte) == 0; };
    std::string residual;
    std::size_t pos = first;
    for (;;) {
        const std::size_t open = text.find(kToolOpen, pos);
        if (open == std::string::npos) {
            const std::string_view trailing = std::string_view(text).substr(pos);
            if (std::none_of(trailing.begin(), trailing.end(), not_space)) { break; }
            residual.append(trailing);
            break;
        }
        const std::string_view between = std::string_view(text).substr(pos, open - pos);
        if (std::any_of(between.begin(), between.end(), not_space)) { residual.append(between); }
        const std::size_t inner_begin = open + kToolOpen.size();
        const std::size_t close       = text.find(kToolClose, inner_begin);
        if (close == std::string::npos) {
            residual.append(text.substr(open));
            break;
        }
        GeneratedToolCall call;
        if (parse_one_tool_call(std::string_view(text).substr(inner_begin, close - inner_begin),
                                max_tool_name_length, contracts, call)) {
            out.tool_calls.push_back(std::move(call));
        } else {
            residual.append(text.substr(open, close + kToolClose.size() - open));
        }
        pos = close + kToolClose.size();
    }

    if (out.tool_calls.empty()) { return fallback(text); }
    out.is_tool_call_response = true;
    out.residual              = std::move(residual);
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
        // Narrative that could not be attributed to a valid block is restored as content; the
        // retained calls are published structurally.
        return Terminal{.content    = std::move(parsed.residual),
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
