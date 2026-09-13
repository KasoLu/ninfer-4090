// Real-model adherence probe for the v22.4 nested-XML tool-call format.
//
// The test loads a registered 27B artifact (opt-in through NINFER_QWEN3_8_27B_NVFP4_WEIGHTS or a
// sibling 27B artifact variable), asks the model to call preset hypothetical tools whose schemas
// force nested arrays and objects, and then runs the PRODUCT parser exactly as the serving path
// does: RequestOptions.output.raw stays off, so the engine parses the model output with the
// contract compiled from the tool JSONs, and the test inspects the resulting GeneratedToolCall
// list. The model must emit the nested tag form the v22.4 template teaches:
//
//   value  := array | object | text
//   array  := "<array>" item* "</array>"
//   item   := value
//   object := ( "<" property ">" value "</" property ">" )*
//   text   := raw bytes, no escaping, ending at the enclosing closing tag
//
// One ADHERENCE line is printed per sample (sample, thinking mode, verdict, the parsed argument
// shape, or the first error). A valid sample also prints the parsed arguments; a failing sample
// prints the engine's text content and the parsed calls between CONTENT BEGIN/CONTENT END. The
// test fails only when fewer than NINFER_XML_ADHERENCE_MIN_VALID samples (default 1) produce a
// valid call, so a partial rate still passes and reports the real number.

#include "ninfer/engine.h"

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using Json = nlohmann::json;

// ---------------------------------------------------------------------------
// Expected parameter schema: the shape the nested format must reproduce after the engine parsed
// it. Leaf kinds mirror the JSON Schema types declared in the tool JSONs; every expected property
// must be present because the prompts ask for all of them.
// ---------------------------------------------------------------------------
struct Schema {
    enum class Kind : std::uint8_t { Text, Integer, Number, Boolean, Array, Object };

    Kind kind = Kind::Text;
    std::vector<std::pair<std::string, Schema>> properties; // Object members.
    std::vector<Schema> element;                            // Array: exactly one entry.
};

Schema text_schema() { return Schema{}; }

Schema integer_schema() { return Schema{.kind = Schema::Kind::Integer}; }

Schema boolean_schema() { return Schema{.kind = Schema::Kind::Boolean}; }

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
    bool ok           = false;
    std::size_t calls = 0;
    std::string shape;
    std::string error;
};

bool check_value(const Json& value, const Schema& schema, std::string& shape,
                 std::string& error) {
    switch (schema.kind) {
    case Schema::Kind::Text:
        if (!value.is_string() || value.get_ref<const std::string&>().empty()) {
            error = "expected non-empty text, got " + value.dump(0);
            return false;
        }
        shape += "text";
        return true;
    case Schema::Kind::Integer:
        if (!value.is_number_integer()) {
            error = "expected integer, got " + value.dump(0);
            return false;
        }
        shape += "integer";
        return true;
    case Schema::Kind::Number:
        if (!value.is_number()) {
            error = "expected number, got " + value.dump(0);
            return false;
        }
        shape += "number";
        return true;
    case Schema::Kind::Boolean:
        if (!value.is_boolean()) {
            error = "expected boolean, got " + value.dump(0);
            return false;
        }
        shape += "boolean";
        return true;
    case Schema::Kind::Array: {
        if (!value.is_array()) {
            error = "expected array, got " + std::string(value.type_name());
            return false;
        }
        shape += "array[" + std::to_string(value.size()) + "]{";
        bool ok = true;
        for (std::size_t i = 0; i < value.size(); ++i) {
            if (i != 0) { shape += ", "; }
            ok = check_value(value.at(i), schema.element.front(), shape, error) && ok;
        }
        shape += "}";
        return ok;
    }
    case Schema::Kind::Object: {
        if (!value.is_object()) {
            error = "expected object, got " + std::string(value.type_name());
            return false;
        }
        shape += "object{";
        bool ok    = true;
        bool first = true;
        for (const auto& [name, property] : schema.properties) {
            if (!value.contains(name)) {
                error = "missing property '" + name + "'";
                return false;
            }
            if (!first) { shape += ", "; }
            shape += name + ": ";
            ok    = check_value(value.at(name), property, shape, error) && ok;
            first = false;
        }
        for (auto member = value.begin(); member != value.end(); ++member) {
            bool declared = false;
            for (const auto& [name, property] : schema.properties) {
                declared = declared || name == member.key();
            }
            if (!declared) {
                error = "unknown property '" + member.key() + "'";
                return false;
            }
        }
        shape += "}";
        return ok;
    }
    }
    error = "unsupported schema kind";
    return false;
}

Validation validate_calls(const std::vector<ninfer::GeneratedToolCall>& calls,
                          std::string_view expected_tool,
                          const std::vector<std::pair<std::string, Schema>>& parameters) {
    Validation result;
    result.calls = calls.size();
    if (calls.size() != 1) {
        result.error = "expected exactly 1 tool call, got " + std::to_string(calls.size());
        return result;
    }
    const ninfer::GeneratedToolCall& call = calls.front();
    if (call.name != expected_tool) {
        result.error =
            "call name '" + call.name + "', expected '" + std::string(expected_tool) + "'";
        return result;
    }
    const Json arguments = Json::parse(call.arguments_json, nullptr, false);
    if (arguments.is_discarded() || !arguments.is_object()) {
        result.error = "arguments are not a JSON object: " + call.arguments_json;
        return result;
    }
    Schema object;
    object.kind       = Schema::Kind::Object;
    object.properties = parameters;
    std::string error;
    result.ok    = check_value(arguments, object, result.shape, error);
    result.error = error;
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
};

std::vector<Sample> build_samples() {
    std::vector<Sample> samples;

    samples.push_back(Sample{
        .name      = "edit",
        .tool_name = "edit",
        .tool_json =
            R"({"type":"function","function":{"name":"edit","description":"Apply exact text replacements to a file.","parameters":{"type":"object","properties":{"path":{"type":"string","description":"Path of the file to edit."},"edits":{"type":"array","description":"Replacements, each written as [start_id, end_id, replacement_text].","items":{"type":"array","items":{"type":"string"}}}},"required":["path","edits"]}}})",
        .prompt    =
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
    });

    samples.push_back(Sample{
        .name      = "todo_write",
        .tool_name = "todo_write",
        .tool_json =
            R"({"type":"function","function":{"name":"todo_write","description":"Record the current task list.","parameters":{"type":"object","properties":{"todos":{"type":"array","description":"The full task list.","items":{"type":"object","properties":{"content":{"type":"string"},"status":{"type":"string"},"activeForm":{"type":"string"}},"required":["content","status","activeForm"]}}},"required":["todos"]}}})",
        .prompt    =
            R"TXT(Record the current task list with the todo_write tool, using exactly these three items:

1. content "Fix the reconnect leak", status "in_progress", activeForm "Fixing the reconnect leak"
2. content "Add the retry test", status "pending", activeForm "Adding the retry test"
3. content "Update DOING.md", status "pending", activeForm "Updating DOING.md"

Call the tool now.)TXT",
        .parameters = {{"todos", array_schema(object_schema({{"content", text_schema()},
                                                             {"status", text_schema()},
                                                             {"activeForm", text_schema()}}))}},
    });

    samples.push_back(Sample{
        .name      = "deploy",
        .tool_name = "deploy",
        .tool_json =
            R"({"type":"function","function":{"name":"deploy","description":"Deploy a service with an explicit rollout config.","parameters":{"type":"object","properties":{"service":{"type":"string"},"config":{"type":"object","properties":{"replicas":{"type":"integer"},"labels":{"type":"array","items":{"type":"string"}},"rollout":{"type":"object","properties":{"batch_size":{"type":"integer"},"paused":{"type":"boolean"}}}}},"required":["service","config"]}}})",
        .prompt    =
            R"TXT(Deploy the service "checkout" with the deploy tool. Config: replicas 4, labels ["blue", "stable"], rollout with batch_size 2 and paused false.

Call the tool now.)TXT",
        .parameters = {{"service", text_schema()},
                       {"config",
                        object_schema({{"replicas", integer_schema()},
                                       {"labels", array_schema(text_schema())},
                                       {"rollout",
                                        object_schema({{"batch_size", integer_schema()},
                                                       {"paused", boolean_schema()}})}})}},
    });

    samples.push_back(Sample{
        .name      = "pwsh",
        .tool_name = "pwsh",
        .tool_json =
            R"({"type":"function","function":{"name":"pwsh","description":"Run a PowerShell command.","parameters":{"type":"object","properties":{"command":{"type":"string"},"description":{"type":"string"},"run_in_background":{"type":"boolean"}},"required":["command","description"]}}})",
        .prompt    =
            R"TXT(Run this command in the background with the pwsh tool:

Get-ChildItem -Recurse -Filter *.log | Select-String -Pattern "error"

Use the description "Scan logs for errors". Call the tool now.)TXT",
        .parameters = {{"command", text_schema()},
                       {"description", text_schema()},
                       {"run_in_background", boolean_schema()}},
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

    const ninfer::GenerationResult result = engine.generate(engine.prepare(input), request);
    const Validation validation = validate_calls(result.tool_calls, sample.tool_name,
                                                 sample.parameters);

    std::cout << "ADHERENCE sample=" << index << " name=" << sample.name
              << " thinking=" << (thinking ? 1 : 0) << " valid=" << (validation.ok ? 1 : 0)
              << " tool_calls=" << validation.calls;
    if (validation.ok) {
        std::cout << " shape=" << validation.shape;
    } else {
        std::cout << " error=" << validation.error;
    }
    std::cout << '\n';
    if (validation.ok) {
        std::cout << "ARGS sample=" << index << " json=" << result.tool_calls.front().arguments_json
                  << '\n';
    } else {
        std::cout << "CONTENT BEGIN\n" << result.content << "\nCONTENT END\n";
        for (const ninfer::GeneratedToolCall& call : result.tool_calls) {
            std::cout << "CALL name=" << call.name << " args=" << call.arguments_json << '\n';
        }
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
