// Real-model adherence probe for the v22.4 nested-XML tool-call format.
//
// The engine's tool-call parser does not consume the <array>/<item> value form yet, so this
// test does not run the product parser at all. It loads a registered 27B artifact (opt-in
// through NINFER_QWEN3_8_27B_NVFP4_WEIGHTS or a sibling 27B artifact variable), asks the model
// to call preset hypothetical tools whose schemas force nested arrays and objects, reads the
// RAW output (RequestOptions.output.raw), and validates the emitted structure with an
// independent checker that implements the format the v22.4 template teaches:
//
//   value  := array | object | text
//   array  := "<array>" item* "</array>"
//   item   := value
//   object := ( "<" property ">" value "</" property ">" )*
//   text   := raw bytes, no escaping, ending at the enclosing closing tag
//
// One ADHERENCE line is printed per sample (sample, thinking mode, verdict, parsed shape or the
// first structural error); a failing sample also prints its raw output between RAW BEGIN/RAW
// END. The test fails only when fewer than NINFER_XML_ADHERENCE_MIN_VALID samples (default 1)
// produce a structurally valid call, so a partial rate still passes and reports the real number.

#include "ninfer/engine.h"

#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// Expected parameter schema: the shape the nested format must reproduce.
// ---------------------------------------------------------------------------
struct Schema {
    enum class Kind : std::uint8_t { Text, Array, Object };

    Kind kind = Kind::Text;
    std::vector<std::pair<std::string, Schema>> properties; // Object members.
    std::vector<Schema> element;                            // Array: exactly one entry.
};

Schema text_schema() { return Schema{}; }

Schema array_schema(Schema element) {
    Schema schema;
    schema.kind = Schema::Kind::Array;
    schema.element.push_back(std::move(element));
    return schema;
}

Schema object_schema(std::vector<std::pair<std::string, Schema>> properties) {
    Schema schema;
    schema.kind       = Schema::Kind::Object;
    schema.properties = std::move(properties);
    return schema;
}

struct Validation {
    bool ok     = false;
    std::size_t calls = 0;
    std::string shape;
    std::string error;
};

struct ParseState {
    std::string_view text;
    std::size_t pos = 0;
    std::string shape;
    std::string error;
    bool failed = false;
};

void fail(ParseState& state, std::string message) {
    if (!state.failed) {
        state.failed = true;
        state.error  = std::move(message);
    }
}

void skip_separators(ParseState& state) {
    while (state.pos < state.text.size()) {
        const char character = state.text[state.pos];
        if (character == ' ' || character == '\t' || character == '\n' || character == '\r') {
            ++state.pos;
        } else {
            break;
        }
    }
}

bool starts_with(const ParseState& state, std::string_view token) {
    return state.text.substr(state.pos).starts_with(token);
}

bool expect(ParseState& state, std::string_view token) {
    if (starts_with(state, token)) {
        state.pos += token.size();
        return true;
    }
    fail(state, "expected '" + std::string(token) + "' at offset " + std::to_string(state.pos));
    return false;
}

bool read_name(ParseState& state, std::string& name) {
    const std::size_t begin = state.pos;
    while (state.pos < state.text.size()) {
        const char character = state.text[state.pos];
        if (std::isalnum(static_cast<unsigned char>(character)) || character == '_' ||
            character == '-' || character == '.') {
            ++state.pos;
        } else {
            break;
        }
    }
    if (state.pos == begin) {
        fail(state, "empty name at offset " + std::to_string(state.pos));
        return false;
    }
    name.assign(state.text.substr(begin, state.pos - begin));
    return true;
}

// The closing tag of a value: prefer an occurrence at the start of a line, which is how the
// format writes multi-line leaves; fall back to the first occurrence for inline leaves.
std::size_t find_close(std::string_view text, std::size_t pos, std::string_view close) {
    const std::string anchored = "\n" + std::string(close);
    const std::size_t line     = text.find(anchored, pos);
    if (line != std::string_view::npos) { return line + 1; }
    return text.find(close, pos);
}

std::string_view trim_framing(std::string_view value) {
    if (!value.empty() && value.front() == '\r') { value.remove_prefix(1); }
    if (!value.empty() && value.front() == '\n') { value.remove_prefix(1); }
    if (!value.empty() && value.back() == '\n') { value.remove_suffix(1); }
    if (!value.empty() && value.back() == '\r') { value.remove_suffix(1); }
    return value;
}

const Schema* find_property(const Schema& schema, std::string_view name) {
    for (const auto& [property_name, property] : schema.properties) {
        if (property_name == name) { return &property; }
    }
    return nullptr;
}

void parse_value(ParseState& state, std::string_view close, const Schema& schema,
                 std::size_t depth) {
    if (state.failed) { return; }
    if (depth > 32) {
        fail(state, "nesting deeper than 32");
        return;
    }
    switch (schema.kind) {
    case Schema::Kind::Array: {
        if (!expect(state, "<array>")) { return; }
        std::string outer = std::move(state.shape);
        state.shape.clear();
        std::size_t count = 0;
        for (;;) {
            skip_separators(state);
            if (state.failed) { return; }
            if (starts_with(state, "</array>")) {
                state.pos += std::string_view("</array>").size();
                break;
            }
            if (!expect(state, "<item>")) { return; }
            if (count != 0) { state.shape += ", "; }
            parse_value(state, "</item>", schema.element.front(), depth + 1);
            if (state.failed) { return; }
            if (!expect(state, "</item>")) { return; }
            ++count;
        }
        std::string elements = std::move(state.shape);
        state.shape          = std::move(outer);
        state.shape += "array[" + std::to_string(count) + "]{" + elements + "}";
        break;
    }
    case Schema::Kind::Object: {
        std::string outer = std::move(state.shape);
        state.shape.clear();
        bool first = true;
        for (;;) {
            skip_separators(state);
            if (state.failed) { return; }
            if (starts_with(state, close)) { break; }
            if (!expect(state, "<")) { return; }
            std::string name;
            if (!read_name(state, name)) { return; }
            if (!expect(state, ">")) { return; }
            const Schema* property = find_property(schema, name);
            if (property == nullptr) {
                fail(state, "unknown property '" + name + "'");
                return;
            }
            if (!first) { state.shape += ", "; }
            state.shape += name + ": ";
            parse_value(state, "</" + name + ">", *property, depth + 1);
            if (state.failed) { return; }
            if (!expect(state, "</" + name + ">")) { return; }
            first = false;
        }
        std::string members = std::move(state.shape);
        state.shape         = std::move(outer);
        state.shape += "object{" + members + "}";
        break;
    }
    case Schema::Kind::Text: {
        const std::size_t close_pos = find_close(state.text, state.pos, close);
        if (close_pos == std::string_view::npos) {
            fail(state, "missing '" + std::string(close) + "'");
            return;
        }
        const std::string_view raw =
            trim_framing(state.text.substr(state.pos, close_pos - state.pos));
        if (raw.empty()) {
            fail(state, "empty text leaf");
            return;
        }
        state.pos = close_pos;
        state.shape += "text";
        break;
    }
    }
}

Validation validate_output(std::string_view output, std::string_view expected_tool,
                           const std::vector<std::pair<std::string, Schema>>& parameters,
                           const std::vector<std::string>& required) {
    Validation result;
    std::string_view text = output;
    // Raw mode with thinking enabled may still carry the closed thinking block; skip it so a
    // format discussion inside it cannot be mistaken for the emitted call.
    const std::size_t think_open = text.find("<think>");
    const std::size_t think_shut = text.find("</think>");
    if (think_open != std::string_view::npos && think_shut != std::string_view::npos &&
        think_open < think_shut) {
        text = text.substr(think_shut + std::string_view("</think>").size());
    }

    const std::size_t call = text.find("<tool_call>");
    if (call == std::string_view::npos) {
        result.error = "no <tool_call> block";
        return result;
    }
    result.calls = 1;

    ParseState state{text, call, {}, {}, false};
    if (!expect(state, "<tool_call>")) {
        result.error = state.error;
        return result;
    }
    skip_separators(state);
    if (!expect(state, "<function=")) {
        result.error = state.error;
        return result;
    }
    std::string function_name;
    if (!read_name(state, function_name) || !expect(state, ">")) {
        result.error = state.error;
        return result;
    }
    if (function_name != expected_tool) {
        result.error =
            "function '" + function_name + "', expected '" + std::string(expected_tool) + "'";
        return result;
    }

    std::vector<std::string> seen;
    for (;;) {
        skip_separators(state);
        if (state.failed) { break; }
        if (starts_with(state, "</function>")) {
            state.pos += std::string_view("</function>").size();
            break;
        }
        if (!expect(state, "<parameter=")) { break; }
        std::string parameter_name;
        if (!read_name(state, parameter_name) || !expect(state, ">")) { break; }
        const Schema* schema = nullptr;
        for (const auto& [name, parameter] : parameters) {
            if (name == parameter_name) { schema = &parameter; }
        }
        if (schema == nullptr) {
            fail(state, "undeclared parameter '" + parameter_name + "'");
            break;
        }
        state.shape += parameter_name + ": ";
        parse_value(state, "</parameter>", *schema, 0);
        if (state.failed) { break; }
        if (!expect(state, "</parameter>")) { break; }
        seen.push_back(parameter_name);
    }
    if (!state.failed) {
        skip_separators(state);
        if (!expect(state, "</tool_call>")) {
            result.error = state.error;
            return result;
        }
        for (const std::string& name : required) {
            bool present = false;
            for (const std::string& seen_name : seen) { present = present || seen_name == name; }
            if (!present) {
                fail(state, "missing required parameter '" + name + "'");
                break;
            }
        }
    }

    result.ok    = !state.failed;
    result.shape = state.shape;
    result.error = state.error;
    return result;
}

// ---------------------------------------------------------------------------
// Hypothetical tools and preset prompts.
// ---------------------------------------------------------------------------
struct Sample {
    std::string name;
    std::string tool_name;
    std::string tool_json;
    std::string prompt;
    std::vector<std::pair<std::string, Schema>> parameters;
    std::vector<std::string> required;
};

std::vector<Sample> build_samples() {
    std::vector<Sample> samples;

    samples.push_back(Sample{
        .name       = "edit",
        .tool_name  = "edit",
        .tool_json  =
            R"({"type":"function","function":{"name":"edit","description":"Apply exact text replacements to a file.","parameters":{"type":"object","properties":{"path":{"type":"string","description":"Path of the file to edit."},"edits":{"type":"array","description":"Replacements, each written as [start_id, end_id, replacement_text].","items":{"type":"array","items":{"type":"string"}}}},"required":["path","edits"]}}})",
        .prompt     =
            R"TXT(Use the edit tool to apply two replacements to src/service.py.

Replacement 1: replace the block with start id "a1" and end id "a1" with exactly this Python code:

def parse(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("[") or text.startswith("{"):
        return {"kind": "json", "value": text, "note": "quotes \" stay literal"}
    return {"kind": "text", "value": text}

Replacement 2: replace the block with start id "b2" and end id "b2" with exactly this text:

return {"ok": True, "paths": [left, right]}

Call the tool now.)TXT",
        .parameters = {{"path", text_schema()},
                       {"edits", array_schema(array_schema(text_schema()))}},
        .required   = {"path", "edits"},
    });

    samples.push_back(Sample{
        .name       = "todo_write",
        .tool_name  = "todo_write",
        .tool_json  =
            R"({"type":"function","function":{"name":"todo_write","description":"Record the current task list.","parameters":{"type":"object","properties":{"todos":{"type":"array","description":"The full task list.","items":{"type":"object","properties":{"content":{"type":"string"},"status":{"type":"string"},"activeForm":{"type":"string"}},"required":["content","status","activeForm"]}}},"required":["todos"]}}})",
        .prompt     =
            R"TXT(Record the current task list with the todo_write tool, using exactly these three items:

1. content "Fix the reconnect leak", status "in_progress", activeForm "Fixing the reconnect leak"
2. content "Add the retry test", status "pending", activeForm "Adding the retry test"
3. content "Update DOING.md", status "pending", activeForm "Updating DOING.md"

Call the tool now.)TXT",
        .parameters = {{"todos", array_schema(object_schema({{"content", text_schema()},
                                                             {"status", text_schema()},
                                                             {"activeForm", text_schema()}}))}},
        .required   = {"todos"},
    });

    samples.push_back(Sample{
        .name       = "deploy",
        .tool_name  = "deploy",
        .tool_json  =
            R"({"type":"function","function":{"name":"deploy","description":"Deploy a service with an explicit rollout config.","parameters":{"type":"object","properties":{"service":{"type":"string"},"config":{"type":"object","properties":{"replicas":{"type":"integer"},"labels":{"type":"array","items":{"type":"string"}},"rollout":{"type":"object","properties":{"batch_size":{"type":"integer"},"paused":{"type":"boolean"}}}}},"required":["service","config"]}}})",
        .prompt     =
            R"TXT(Deploy the service "checkout" with the deploy tool. Config: replicas 4, labels ["blue", "stable"], rollout with batch_size 2 and paused false.

Call the tool now.)TXT",
        .parameters = {{"service", text_schema()},
                       {"config",
                        object_schema({{"replicas", text_schema()},
                                       {"labels", array_schema(text_schema())},
                                       {"rollout",
                                        object_schema({{"batch_size", text_schema()},
                                                       {"paused", text_schema()}})}})}},
        .required   = {"service", "config"},
    });

    samples.push_back(Sample{
        .name       = "pwsh",
        .tool_name  = "pwsh",
        .tool_json  =
            R"({"type":"function","function":{"name":"pwsh","description":"Run a PowerShell command.","parameters":{"type":"object","properties":{"command":{"type":"string"},"description":{"type":"string"},"run_in_background":{"type":"boolean"}},"required":["command","description"]}}})",
        .prompt     =
            R"TXT(Run this command in the background with the pwsh tool:

Get-ChildItem -Recurse -Filter *.log | Select-String -Pattern "error"

Use the description "Scan logs for errors". Call the tool now.)TXT",
        .parameters = {{"command", text_schema()},
                       {"description", text_schema()},
                       {"run_in_background", text_schema()}},
        .required   = {"command", "description"},
    });

    return samples;
}

std::size_t run_sample(ninfer::Engine& engine, const Sample& sample, bool thinking,
                       std::size_t index) {
    ninfer::PromptInput input;
    ninfer::ChatMessage user;
    user.role = ninfer::ChatRole::User;
    user.parts.push_back(ninfer::MessagePart{
        .kind = ninfer::MessagePartKind::Text, .text = sample.prompt, .media = {}});
    input.messages.push_back(std::move(user));
    input.options.enable_thinking = thinking;
    input.options.tool_jsons.push_back(sample.tool_json);

    ninfer::RequestOptions request;
    request.execution.requested_output_tokens = 2048;
    request.execution.sampling.temperature    = 0.0F;
    request.execution.allow_prefix_reuse      = false;
    request.output.raw                        = true;

    const ninfer::GenerationResult result = engine.generate(engine.prepare(input), request);
    const Validation validation =
        validate_output(result.content, sample.tool_name, sample.parameters, sample.required);

    std::cout << "ADHERENCE sample=" << index << " name=" << sample.name
              << " thinking=" << (thinking ? 1 : 0) << " valid=" << (validation.ok ? 1 : 0)
              << " tool_calls=" << validation.calls;
    if (validation.ok) {
        std::cout << " shape=" << validation.shape;
    } else {
        std::cout << " error=" << validation.error;
    }
    std::cout << '\n';
    if (!validation.ok) {
        std::cout << "RAW BEGIN\n" << result.content << "\nRAW END\n";
    }
    return validation.ok ? 1 : 0;
}

} // namespace

int main() {
    const char* artifact = nullptr;
    const char* candidates[] = {"NINFER_QWEN3_8_27B_NVFP4_WEIGHTS", "NINFER_QWEN3_8_27B_WEIGHTS",
                                "NINFER_QWEN3_6_27B_NVFP4_WEIGHTS", "NINFER_QWEN3_6_27B_WEIGHTS"};
    for (const char* candidate : candidates) {
        const char* value = std::getenv(candidate);
        if (value != nullptr && value[0] != '\0') {
            artifact = value;
            break;
        }
    }
    if (artifact == nullptr) {
        std::cout << "skip: set NINFER_QWEN3_8_27B_NVFP4_WEIGHTS (or a sibling 27B artifact "
                     "variable) to run the nested-XML tool-call adherence probe\n";
        return 77;
    }
    std::cout << "artifact: " << artifact << '\n';

    ninfer::EngineOptions options;
    options.artifact_path        = artifact;
    options.max_context          = 4096;
    options.kv_capacity          = ninfer::KvCapacityPolicy::explicit_capacity(4096);
    options.prefill_chunk        = 1024;
    options.max_concurrency      = 1;
    options.max_pending_requests = 1;
    ninfer::Engine engine(std::move(options));

    const std::vector<Sample> samples = build_samples();
    std::size_t valid                 = 0;
    std::size_t total                 = 0;
    for (std::size_t index = 0; index < samples.size(); ++index) {
        for (const bool thinking : {false, true}) {
            valid += run_sample(engine, samples[index], thinking, index);
            ++total;
        }
    }

    std::cout << "ADHERENCE SUMMARY valid=" << valid << "/" << total << '\n';
    const char* minimum_env   = std::getenv("NINFER_XML_ADHERENCE_MIN_VALID");
    const std::size_t minimum = minimum_env == nullptr ? 1 : std::strtoul(minimum_env, nullptr, 10);
    if (valid < minimum) {
        std::cerr << "nested-XML tool-call adherence below the minimum: " << valid << " < "
                  << minimum << '\n';
        return 1;
    }
    return 0;
}
