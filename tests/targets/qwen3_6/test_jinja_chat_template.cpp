#include <ninfer/targets/qwen3_6/frontend.h>
#include <ninfer/targets/qwen3_6/frontend_resources.h>

#include "targets/qwen3_6/impl/frontend/chat_template.h"
#include "targets/qwen3_6/impl/frontend/test_access.h"
#include "targets/qwen3_6/impl/frontend/tokenizer.h"
#include "text/unicode.h"

#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <limits>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

namespace fi            = ninfer::targets::qwen3_6::frontend_internal;
using Frontend          = ninfer::targets::qwen3_6::Frontend;
using FrontendOptions   = ninfer::targets::qwen3_6::FrontendOptions;
using FrontendFactory   = ninfer::targets::qwen3_6::FrontendTestAccess;
using FrontendResources = ninfer::targets::qwen3_6::FrontendResources;


int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

fi::ChatMessage text_message(ninfer::ChatRole role, std::string text) {
    fi::ChatMessage message;
    message.role = role;
    message.parts.push_back(fi::ChatPart::text_part(std::move(text)));
    return message;
}

std::string byte_level_symbol(std::uint8_t target) {
    std::uint32_t next = 256;
    for (int value = 0; value <= 255; ++value) {
        const bool visible = (value >= 33 && value <= 126) || (value >= 161 && value <= 172) ||
                             (value >= 174 && value <= 255);
        const std::uint32_t codepoint = visible ? static_cast<std::uint32_t>(value) : next++;
        if (value == target) {
            return ninfer::text::unicode_internal::codepoint_to_utf8(
                static_cast<std::int32_t>(codepoint));
        }
    }
    throw std::logic_error("byte-level test symbol is outside one byte");
}

nlohmann::json added(int id, std::string content, bool special = false) {
    return nlohmann::json{{"id", id},
                          {"content", std::move(content)},
                          {"single_word", false},
                          {"lstrip", false},
                          {"rstrip", false},
                          {"normalized", false},
                          {"special", special}};
}

FrontendResources synthetic_resources(std::string chat_template) {
    FrontendResources result;
    result.chat_template_jinja = std::move(chat_template);
    const nlohmann::json tokens =
        nlohmann::json::array({added(6, "<eos>", true)});
    nlohmann::json vocab = nlohmann::json::object();
    for (int value = 0; value < 256; ++value) {
        vocab[byte_level_symbol(static_cast<std::uint8_t>(value))] = 128 + value;
    }
    vocab["x"] = 0;
    result.tokenizer_json =
        nlohmann::json{{"model",
                        {{"type", "BPE"}, {"vocab", std::move(vocab)},
                         {"merges", nlohmann::json::array()}}},
                       {"added_tokens", tokens}}
            .dump();
    nlohmann::json decoder = nlohmann::json::object();
    for (const nlohmann::json& token : tokens) {
        nlohmann::json value = token;
        const std::string id = std::to_string(value.at("id").get<int>());
        value.erase("id");
        decoder[id] = std::move(value);
    }
    result.tokenizer_config_json =
        nlohmann::json{{"add_bos_token", false},
                       {"add_prefix_space", false},
                       {"pad_token", "<|endoftext|>"},
                       {"chat_template", result.chat_template_jinja},
                       {"added_tokens_decoder", std::move(decoder)}}
            .dump();
    result.generation_config_json = R"({"eos_token_id":[6]})";
    result.preprocessor_config_json =
        R"({"patch_size":16,"temporal_patch_size":2,"merge_size":2,"image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5],"size":{"shortest_edge":4096,"longest_edge":16777216}})";
    result.video_preprocessor_config_json =
        R"({"patch_size":16,"temporal_patch_size":2,"merge_size":2,"image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5],"size":{"shortest_edge":4096,"longest_edge":25165824}})";
    return result;
}

int test_custom_template_cache_markers() {
    // Regression (4090 serve run): a custom Jinja template exposes no per-message or
    // cache-marker byte frontier. prepare() must degrade every requested marker to
    // "no independent boundary" and keep only the EngineObserved full-prompt shared
    // prefix, instead of tripping the rendered-marker size guard.
    const std::string source =
        "{%- for message in messages -%}"
        "{{- message.role ~ ': ' ~ message.content -}}"
        "{%- endfor -%}"
        "{{- 'assistant:' -}}";
    FrontendOptions options;
    options.max_context = std::numeric_limits<std::uint32_t>::max();
    const std::filesystem::path template_file =
        std::filesystem::temp_directory_path() / "ninfer_jinja_marker_template_test.jinja";
    {
        std::ofstream out(template_file, std::ios::binary);
        out << source;
    }
    options.chat_template_path = template_file;
    const Frontend frontend =
        FrontendFactory::create_component(synthetic_resources(source), options);

    ninfer::ChatMessage system;
    system.role = ninfer::ChatRole::System;
    system.parts.push_back(ninfer::MessagePart{
        .kind = ninfer::MessagePartKind::Text, .text = "stable system", .media = {}});
    ninfer::ChatMessage user;
    user.role = ninfer::ChatRole::User;
    user.parts.push_back(ninfer::MessagePart{
        .kind = ninfer::MessagePartKind::Text, .text = "hello world", .media = {}});

    ninfer::PromptInput input;
    input.messages.push_back(std::move(system));
    input.messages.push_back(std::move(user));
    input.options.tool_jsons.push_back(
        R"({"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}})");
    input.context_cache.markers.push_back(ninfer::PromptCacheMarker{
        .after_message_count = 2,
        .kind                = ninfer::PromptCacheMarkerKind::PrivateLongAnchor,
        .location            = ninfer::PromptCacheMarkerLocation::MessageBoundary,
    });

    const auto prepared = frontend.prepare(std::move(input));
    const auto& data    = FrontendFactory::inspect(prepared);

    const auto& opportunities = data.context_cache.opportunities;
    return check(opportunities.size() == 1 &&
                     opportunities[0].kind == ninfer::PromptCacheMarkerKind::SharedStablePrefix &&
                     opportunities[0].evidence == ninfer::SharedCandidateEvidence::EngineObserved &&
                     opportunities[0].frontier == data.token_ids.size(),
                 "custom template cache markers must degrade to the full-prompt prefix only");
}

} // namespace


int main() {
    constexpr std::string_view source = R"JINJA(
{%- macro render_content(content) -%}
    {%- if content is string -%}
        {{- content -}}
    {%- else -%}
        {%- for item in content -%}
            {%- if item.type == 'text' -%}{{- item.text -}}
            {%- elif item.type == 'image' -%}{{- '<image>' -}}
            {%- endif -%}
        {%- endfor -%}
    {%- endif -%}
{%- endmacro -%}
{%- set state = namespace(matched=false) -%}
{%- set chained = '<think>\nreason\n</think>'.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n') -%}
{%- for message in messages -%}
    {%- set content = render_content(message.content).lstrip().rstrip() -%}
    {%- if message.role == 'user' and content.startswith('hello') and content.endswith('world') -%}
        {%- set state.matched = true -%}
    {%- endif -%}
    {{- message.role ~ ':' ~ content ~ '/' ~ (content.split(' ') | length) ~ ';' -}}
    {%- if message.tool_calls -%}
        {{- 'args=' ~ (message.tool_calls[0].function.arguments | tojson) ~ ';' -}}
    {%- endif -%}
{%- endfor -%}
{{- 'matched=' ~ state.matched ~ ';tools=' ~ (tools | length) ~ ';preserve=' ~ chat_template_kwargs.preserve_thinking ~ ';chain=' ~ chained ~ ';effort=' ~ reasoning_effort -}}
)JINJA";

    int failures = 0;
    const fi::CompiledChatTemplate chat_template =
        fi::CompiledChatTemplate::compile_jinja(std::string(source), "test-template");
    const ninfer::PromptCapabilities capabilities = chat_template.capabilities();
    failures += check(capabilities.enable_thinking,
                      "custom Jinja template did not expose thinking control");
    failures += check(capabilities.reasoning_effort.low && capabilities.reasoning_effort.medium &&
                          capabilities.reasoning_effort.xhigh &&
                          capabilities.reasoning_effort.default_effort ==
                              ninfer::ReasoningEffort::Medium,
                      "custom Jinja template did not expose reasoning-effort presets");

    fi::ChatMessage user;
    user.role = ninfer::ChatRole::User;
    user.parts.push_back(fi::ChatPart::text_part(" hello "));
    user.parts.push_back(fi::ChatPart::image({}));
    user.parts.push_back(fi::ChatPart::text_part("world "));

    fi::ChatMessage assistant = text_message(ninfer::ChatRole::Assistant, "done");
    assistant.tool_calls.push_back(
        fi::ToolCall{.id = "call-1", .name = "lookup", .arguments_json = R"({"query":"qwen"})"});

    fi::ChatRenderOptions options;
    options.reasoning_effort = ninfer::ReasoningEffort::Medium;
    options.preserve_thinking = true;
    options.tool_jsons.push_back(R"({"type":"function","function":{"name":"lookup"}})");
    const fi::RenderedChat rendered = chat_template.render({std::move(user), std::move(assistant)},
                                                            std::move(options));
    const std::string expected =
        "user:hello <image>world/2;assistant:done/1;args={\"query\": \"qwen\"};"
        "matched=True;tools=1;preserve=True;chain=reason;effort=medium";
    failures += check(rendered.text == expected,
                      ("custom Jinja template context rendered unexpected prompt text: " +
                       rendered.text)
                          .c_str());
    failures += check(rendered.rewrite_checkpoint.has_value() &&
                          rendered.rewrite_checkpoint->kind ==
                              ninfer::targets::qwen3_6::RewriteCheckpointKind::ResponseReplay &&
                          rendered.rewrite_checkpoint->offset == rendered.text.size(),
                      "preserved-thinking closed-turn render did not expose the probed "
                      "response-replay checkpoint at the generation boundary");

    // P1 fix (upstream PR #42 review): the starting channel is derived from the
    // rendered prompt's reasoning markers, not from the enable_thinking option alone.
    failures += check(fi::prompt_starts_in_reasoning("<|im_start|>assistant\n<think>\n"),
                      "open <think> prologue did not classify the turn as starting in reasoning");
    failures += check(!fi::prompt_starts_in_reasoning("<|im_start|>assistant\n<think>\n\n</think>\n\n"),
                      "closed think block was classified as starting in reasoning");
    failures += check(!fi::prompt_starts_in_reasoning("user:hello/assistant:"),
                      "prompt without reasoning markers was classified as starting in reasoning");
    failures += check(fi::prompt_starts_in_reasoning("model\n<|think|>\n"),
                      "open <|think|> prologue did not classify the turn as starting in reasoning");
    failures += check(!fi::prompt_starts_in_reasoning("model\n</thinking>\n"),
                      "closed </thinking> block was classified as starting in reasoning");

    // P2 fix (upstream PR #42 review): a token mentioned only in a {# #} comment must not
    // advertise reasoning-effort support.
    const fi::CompiledChatTemplate commented_template = fi::CompiledChatTemplate::compile_jinja(
        "{# reasoning_effort is intentionally unsupported here #}\n"
        "{% for message in messages %}{{ message.content }}{% endfor %}",
        "block-comment-template");
    failures += check(!commented_template.capabilities().reasoning_effort.low,
                      "block comment mentioning reasoning_effort advertised effort support");

    bool malformed_rejected = false;
    try {
        (void)fi::CompiledChatTemplate::compile_jinja("{% if", "malformed-template");
    } catch (const std::invalid_argument&) { malformed_rejected = true; }
    failures += check(malformed_rejected, "malformed Jinja template was accepted");

    const fi::CompiledChatTemplate no_effort_template = fi::CompiledChatTemplate::compile_jinja(
        "{% for message in messages %}{{ message.content }}{% endfor %}", "no-effort-template");
    bool effort_rejected = false;
    try {
        fi::ChatRenderOptions effort_options;
        effort_options.reasoning_effort = ninfer::ReasoningEffort::Low;
        (void)no_effort_template.render({text_message(ninfer::ChatRole::User, "hello")},
                                        std::move(effort_options));
    } catch (const std::invalid_argument&) { effort_rejected = true; }
    failures += check(effort_rejected,
                      "Jinja template without reasoning_effort accepted reasoning effort");

    // A closed turn without preserved thinking publishes a TurnClosure checkpoint probed
    // at the end of the last real user query: a successor that branches with a new user
    // message rewrites everything rendered after that query, so the stable history before
    // it stays reusable instead of the whole prefix being lost.
    const std::string_view branch_template = R"JINJA({%- for m in messages -%}{%- if m.role == 'user' -%}U{{ m.content }}{%- elif m.role == 'assistant' -%}A{% if m.reasoning_content %}[{{ m.reasoning_content }}]{% endif %}{{ m.content }}{%- elif m.role == 'tool' -%}T{{ m.content }}{%- endif -%}{%- endfor -%}{%- if add_generation_prompt -%}G{%- endif -%})JINJA";
    const fi::CompiledChatTemplate branch_template_compiled =
        fi::CompiledChatTemplate::compile_jinja(std::string(branch_template), "branch-template");
    fi::ChatMessage query = text_message(ninfer::ChatRole::User, "hello");
    fi::ChatMessage tool_turn = text_message(ninfer::ChatRole::Assistant, "c1");
    tool_turn.reasoning_content = "r1";
    fi::ChatMessage tool_result = text_message(ninfer::ChatRole::Tool, "t1");
    const fi::RenderedChat branch_rendered =
        branch_template_compiled.render({query, tool_turn, tool_result}, {});
    const std::string stable_history = "Uhello";
    failures += check(branch_rendered.rewrite_checkpoint.has_value() &&
                          branch_rendered.rewrite_checkpoint->kind ==
                              ninfer::targets::qwen3_6::RewriteCheckpointKind::TurnClosure &&
                          branch_rendered.rewrite_checkpoint->offset == stable_history.size() &&
                          branch_rendered.text.compare(0, stable_history.size(), stable_history) ==
                              0,
                      "closed custom Jinja turn did not probe the turn-closure checkpoint at "
                      "the last real user query");
    const fi::RenderedChat branch_no_prompt =
        branch_template_compiled.render({text_message(ninfer::ChatRole::User, "hello")},
                                        fi::ChatRenderOptions{.add_generation_prompt = false});
    failures += check(!branch_no_prompt.rewrite_checkpoint.has_value(),
                      "custom Jinja render without a generation prompt exposed a checkpoint");
    fi::ChatRenderOptions continuation_options;
    continuation_options.continuation = ninfer::PromptContinuationMode::ContinueFinalAssistant;
    const fi::RenderedChat branch_continuation = branch_template_compiled.render(
        {text_message(ninfer::ChatRole::User, "hello"),
         text_message(ninfer::ChatRole::Assistant, "done")},
        continuation_options);
    failures += check(!branch_continuation.rewrite_checkpoint.has_value(),
                      "custom Jinja assistant continuation exposed a checkpoint");

    // A template whose output depends on later messages (here the total message count)
    // must not expose a checkpoint at a dishonest frontier: the probe text is not a byte
    // prefix of the full render, so the checkpoint stays unset.
    const std::string_view length_dependent_template =
        R"JINJA({%- for m in messages -%}{{ messages|length }}:{{ m.role }};{%- endfor -%})JINJA";
    const fi::CompiledChatTemplate length_dependent_compiled = fi::CompiledChatTemplate::compile_jinja(
        std::string(length_dependent_template), "length-dependent-template");
    const fi::RenderedChat length_dependent_rendered = length_dependent_compiled.render(
        {text_message(ninfer::ChatRole::User, "hello"),
         text_message(ninfer::ChatRole::Assistant, "done")},
        {});
    failures += check(!length_dependent_rendered.rewrite_checkpoint.has_value(),
                      "length-dependent custom Jinja template exposed a dishonest checkpoint");

    failures += test_custom_template_cache_markers();
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}