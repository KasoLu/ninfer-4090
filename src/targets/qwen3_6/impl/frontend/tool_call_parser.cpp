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

ToolArgumentTypeContracts::Schema compile_schema(const Json& property) {
    ToolArgumentTypeContracts::Schema schema;
    if (!property.is_object()) { return schema; }
    const auto type = property.find("type");
    if (type == property.end()) { return schema; }

    std::string name;
    if (type->is_string()) {
        name = type->get<std::string>();
    } else if (type->is_array()) {
        for (const Json& member : *type) {
            if (!member.is_string()) { continue; }
            const std::string& candidate = member.get_ref<const std::string&>();
            if (candidate != "null" && is_json_schema_type(candidate)) {
                name = candidate;
                break;
            }
        }
    }

    if (name == "array") {
        schema.kind      = ToolArgumentTypeContracts::Schema::Kind::Array;
        const auto items = property.find("items");
        schema.element.push_back(items != property.end() && items->is_object()
                                     ? compile_schema(*items)
                                     : ToolArgumentTypeContracts::Schema{});
    } else if (name == "object") {
        schema.kind           = ToolArgumentTypeContracts::Schema::Kind::Object;
        const auto properties = property.find("properties");
        if (properties != property.end() && properties->is_object()) {
            for (const auto& [member_name, member] : properties->items()) {
                schema.properties.emplace_back(member_name, compile_schema(member));
            }
        }
    } else if (name == "integer") {
        schema.kind = ToolArgumentTypeContracts::Schema::Kind::Integer;
    } else if (name == "number") {
        schema.kind = ToolArgumentTypeContracts::Schema::Kind::Number;
    } else if (name == "boolean") {
        schema.kind = ToolArgumentTypeContracts::Schema::Kind::Boolean;
    }
    return schema;
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
            contract.parameters.push_back({parameter_name, encoding, compile_schema(property)});
        }
    }
    return contract;
}

bool same_schema(const ToolArgumentTypeContracts::Schema& lhs,
                 const ToolArgumentTypeContracts::Schema& rhs) {
    if (lhs.kind != rhs.kind || lhs.element.size() != rhs.element.size() ||
        lhs.properties.size() != rhs.properties.size()) {
        return false;
    }
    for (std::size_t i = 0; i < lhs.element.size(); ++i) {
        if (!same_schema(lhs.element[i], rhs.element[i])) { return false; }
    }
    for (std::size_t i = 0; i < lhs.properties.size(); ++i) {
        if (lhs.properties[i].first != rhs.properties[i].first ||
            !same_schema(lhs.properties[i].second, rhs.properties[i].second)) {
            return false;
        }
    }
    return true;
}

bool same_contract(const ToolArgumentTypeContracts::Tool& lhs,
                   const ToolArgumentTypeContracts::Tool& rhs) {
    if (lhs.parameters.size() != rhs.parameters.size()) { return false; }
    for (std::size_t i = 0; i < lhs.parameters.size(); ++i) {
        if (lhs.parameters[i].name != rhs.parameters[i].name ||
            lhs.parameters[i].encoding != rhs.parameters[i].encoding ||
            !same_schema(lhs.parameters[i].schema, rhs.parameters[i].schema)) {
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

// ---------------------------------------------------------------------------
// Nested tag value form (v22.4 template): a declared array/object value may be written with tags
// instead of JSON. <array> holds <item> elements, an object holds one tag per property named
// exactly like the property, and a leaf carries raw text taken verbatim. The declared schema types
// the leaves; structural errors degrade the value to raw text like any other undecodable JSON
// value.
// ---------------------------------------------------------------------------
struct NestedValueParse {
    std::string_view text;
    std::size_t pos = 0;
    std::string error;
    bool failed = false;
};

void nested_fail(NestedValueParse& state, std::string message) {
    if (!state.failed) {
        state.failed = true;
        state.error  = std::move(message);
    }
}

bool read_nested_tag_name(NestedValueParse& state, std::string& name) {
    const std::size_t begin = state.pos;
    while (state.pos < state.text.size()) {
        const unsigned char character = static_cast<unsigned char>(state.text[state.pos]);
        if (std::isalnum(character) != 0 || character == '_' || character == '-' ||
            character == '.') {
            ++state.pos;
        } else {
            break;
        }
    }
    if (state.pos == begin) {
        nested_fail(state, "empty tag name at offset " + std::to_string(state.pos));
        return false;
    }
    name.assign(state.text.substr(begin, state.pos - begin));
    return true;
}

const ToolArgumentTypeContracts::Schema*
find_schema_property(const ToolArgumentTypeContracts::Schema& schema, std::string_view name) {
    for (const auto& [property_name, property] : schema.properties) {
        if (property_name == name) { return &property; }
    }
    return nullptr;
}

// Reads a leaf as the raw bytes up to `close` (or to the end of the span when `close` is empty),
// removing the framing newlines the format writes around multi-line leaves. The nested form writes
// element and property closes inline (e.g. "<item>value</item>"), so the first occurrence ends the
// leaf; the line-anchored preference of the outer tags would skip an inline close whenever a later
// line starts one.
bool parse_nested_leaf(NestedValueParse& state, std::string_view close, Json& out) {
    const std::size_t end =
        close.empty() ? state.text.size() : state.text.find(close, state.pos);
    if (end == std::string_view::npos) {
        nested_fail(state, "missing '" + std::string(close) + "' at offset " +
                               std::to_string(state.pos));
        return false;
    }
    out = std::string(
        remove_parameter_framing_newlines(state.text.substr(state.pos, end - state.pos)));
    state.pos = end;
    return true;
}

bool parse_nested_value(NestedValueParse& state, std::string_view close,
                        const ToolArgumentTypeContracts::Schema& schema, std::size_t depth,
                        Json& out) {
    using Schema = ToolArgumentTypeContracts::Schema;
    constexpr std::string_view kArrayOpen  = "<array>";
    constexpr std::string_view kArrayClose = "</array>";
    constexpr std::string_view kItemOpen   = "<item>";
    constexpr std::string_view kItemClose  = "</item>";
    if (state.failed) { return false; }
    if (depth > 32) {
        nested_fail(state, "nesting deeper than 32");
        return false;
    }
    switch (schema.kind) {
    case Schema::Kind::Array: {
        skip_ws(state.text, state.pos);
        if (!starts_with_at(state.text, state.pos, kArrayOpen)) {
            nested_fail(state, "expected '<array>' at offset " + std::to_string(state.pos));
            return false;
        }
        state.pos += kArrayOpen.size();
        const Schema& element = schema.element.empty() ? Schema{} : schema.element.front();
        Json array            = Json::array();
        for (;;) {
            skip_ws(state.text, state.pos);
            if (starts_with_at(state.text, state.pos, kArrayClose)) {
                state.pos += kArrayClose.size();
                break;
            }
            if (!starts_with_at(state.text, state.pos, kItemOpen)) {
                nested_fail(state, "expected '<item>' or '</array>' at offset " +
                                       std::to_string(state.pos));
                return false;
            }
            state.pos += kItemOpen.size();
            Json item;
            if (!parse_nested_value(state, kItemClose, element, depth + 1, item)) { return false; }
            skip_ws(state.text, state.pos);
            if (!starts_with_at(state.text, state.pos, kItemClose)) {
                nested_fail(state, "expected '</item>' at offset " + std::to_string(state.pos));
                return false;
            }
            state.pos += kItemClose.size();
            array.push_back(std::move(item));
        }
        out = std::move(array);
        return true;
    }
    case Schema::Kind::Object: {
        Json object = Json::object();
        for (;;) {
            skip_ws(state.text, state.pos);
            if (close.empty() ? state.pos >= state.text.size()
                              : starts_with_at(state.text, state.pos, close)) {
                break;
            }
            if (!starts_with_at(state.text, state.pos, "<")) {
                nested_fail(state, "expected '<' at offset " + std::to_string(state.pos));
                return false;
            }
            ++state.pos;
            std::string name;
            if (!read_nested_tag_name(state, name)) { return false; }
            if (!starts_with_at(state.text, state.pos, ">")) {
                nested_fail(state, "expected '>' at offset " + std::to_string(state.pos));
                return false;
            }
            ++state.pos;
            const std::string member_close = "</" + name + ">";
            const Schema* property         = find_schema_property(schema, name);
            Json member;
            if (!parse_nested_value(state, member_close,
                                    property == nullptr ? Schema{} : *property, depth + 1,
                                    member)) {
                return false;
            }
            skip_ws(state.text, state.pos);
            if (!starts_with_at(state.text, state.pos, member_close)) {
                nested_fail(state, "expected '" + member_close + "' at offset " +
                                       std::to_string(state.pos));
                return false;
            }
            state.pos += member_close.size();
            object[name] = std::move(member);
        }
        out = std::move(object);
        return true;
    }
    case Schema::Kind::String:
        return parse_nested_leaf(state, close, out);
    case Schema::Kind::Integer:
    case Schema::Kind::Number:
    case Schema::Kind::Boolean: {
        Json leaf;
        if (!parse_nested_leaf(state, close, leaf)) { return false; }
        const Json scalar = Json::parse(leaf.get<std::string>(), nullptr, false);
        const bool matches =
            !scalar.is_discarded() &&
            ((schema.kind == Schema::Kind::Integer && scalar.is_number_integer()) ||
             (schema.kind == Schema::Kind::Number && scalar.is_number()) ||
             (schema.kind == Schema::Kind::Boolean && scalar.is_boolean()));
        // A leaf that does not match its declared scalar kind keeps its raw text: the engine does
        // not coerce, and the client owns schema validation.
        if (matches) {
            out = scalar;
        } else {
            out = std::move(leaf);
        }
        return true;
    }
    }
    nested_fail(state, "unsupported schema kind");
    return false;
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
            if (!parsed.is_discarded()) {
                NINFER_TC_TRACE("parameter", "key '" + key + "': Json decoded (len " +
                                                 std::to_string(value.size()) + ")");
                args[key] = std::move(parsed);
            } else {
                // The v22.4 template teaches the nested tag form for declared array/object
                // values. Try that form before degrading; a value that decodes as neither keeps
                // its raw text so the client harness can inspect or reject it.
                NestedValueParse nested_state{value, 0, "value is neither JSON nor a tag-led form",
                                              false};
                skip_ws(value, nested_state.pos);
                bool nested_ok = false;
                Json nested;
                if (starts_with_at(value, nested_state.pos, "<")) {
                    nested_state.error.clear();
                    nested_ok = parse_nested_value(nested_state, {}, contract->schema, 0, nested);
                    if (nested_ok) {
                        skip_ws(value, nested_state.pos);
                        if (nested_state.pos != value.size()) {
                            nested_ok = false;
                            nested_fail(nested_state, "trailing bytes after the value at offset " +
                                                          std::to_string(nested_state.pos));
                        }
                    }
                }
                if (nested_ok) {
                    NINFER_TC_TRACE("parameter",
                                    "key '" + key + "': nested tag form decoded (len " +
                                        std::to_string(value.size()) + ")");
                    args[key] = std::move(nested);
                } else {
                    NINFER_TC_TRACE("parameter", "key '" + key +
                                                     "': decode discarded -> raw string (len " +
                                                     std::to_string(value.size()) + "): " +
                                                     nested_state.error + " [json: " +
                                                     trace_json_error(value) + "]; value '" +
                                                     trace_snippet(value, 160) + "'");
                    args[key] = value;
                }
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
