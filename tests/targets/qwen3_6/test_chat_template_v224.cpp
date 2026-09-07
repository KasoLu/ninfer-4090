#include "targets/qwen3_6/impl/frontend/chat_template.h"

#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

namespace fi = ninfer::targets::qwen3_6::frontend_internal;

constexpr std::string_view kLowInstructions =
    "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to "
    "the conclusion without unnecessary elaboration.";
constexpr std::string_view kXHighInstructions =
    "Reasoning effort is set to xhigh. Please think carefully through the task, validate key "
    "assumptions, consider plausible alternatives, and prioritize correctness, consistency, and "
    "clarity in the final answer.";

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

const fi::CompiledChatTemplate& v224_template() {
    static const fi::CompiledChatTemplate value =
        fi::CompiledChatTemplate::resolve_registered(fi::kRegisteredChatTemplateV224);
    return value;
}

fi::ChatMessage chat_message(ninfer::ChatRole role, std::string content) {
    fi::ChatMessage message;
    message.role = role;
    message.parts.push_back(fi::ChatPart::text_part(std::move(content)));
    return message;
}

template <class Callable>
bool throws_invalid_argument(Callable&& callable) {
    try {
        callable();
    } catch (const std::invalid_argument&) { return true; }
    return false;
}

fi::ChatMessage tool_calling_assistant(std::string reasoning, std::string content,
                                       std::string name, std::string arguments_json) {
    fi::ChatMessage message = chat_message(ninfer::ChatRole::Assistant, std::move(content));
    message.reasoning_content = std::move(reasoning);
    message.tool_calls.push_back(
        {.id = "call_1", .name = std::move(name), .arguments_json = std::move(arguments_json)});
    return message;
}

int test_registered_template_resolution() {
    int failures = check(v224_template().capabilities().enable_thinking &&
                             v224_template().capabilities().reasoning_effort.low &&
                             v224_template().capabilities().reasoning_effort.medium &&
                             v224_template().capabilities().reasoning_effort.xhigh &&
                             v224_template().capabilities().reasoning_effort.default_effort ==
                                 ninfer::ReasoningEffort::Medium,
                         "v22_4 did not advertise its complete capability set");
    failures +=
        check(throws_invalid_argument(
                  [] { (void)fi::CompiledChatTemplate::resolve_registered("not-a-template"); }),
              "an unknown registered chat template name was accepted");
    bool names_enumerated = false;
    try {
        (void)fi::CompiledChatTemplate::resolve_registered("not-a-template");
    } catch (const std::invalid_argument& error) {
        names_enumerated = std::string(error.what()).find("v22_4") != std::string::npos;
    }
    failures += check(names_enumerated,
                      "unknown registered chat template error did not list supported names");
    failures += check(throws_invalid_argument([] {
                          (void)v224_template().render({});
                      }),
                      "v22_4 accepted an empty message list");
    return failures;
}

int test_default_render() {
    int failures                   = 0;
    const std::string default_text =
        v224_template().render({chat_message(ninfer::ChatRole::User, "hello")}).text;
    failures += check(
        default_text == "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n<think>\n",
        "v22_4 default render differs from the jinja medium default");
    return failures;
}

int test_reasoning_instructions() {
    int failures                 = 0;
    fi::ChatRenderOptions options;
    options.reasoning_effort = ninfer::ReasoningEffort::Low;
    failures += check(
        v224_template()
                .render({chat_message(ninfer::ChatRole::User, "hello")}, options)
                .text == "<|im_start|>system\n" + std::string(kLowInstructions) +
                             "<|im_end|>\n<|im_start|>user\nhello<|im_end|>\n"
                             "<|im_start|>assistant\n<think>\n",
        "low effort did not render the standalone instruction turn");
    options.reasoning_effort = ninfer::ReasoningEffort::XHigh;
    failures += check(
        v224_template()
                .render({chat_message(ninfer::ChatRole::System, "<|think_xhigh|>Be brief."),
                         chat_message(ninfer::ChatRole::User, "hi")},
                        options)
                .text == "<|im_start|>system\n" + std::string(kXHighInstructions) +
                             "\n\nBe brief.<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n"
                             "<|im_start|>assistant\n<think>\n",
        "xhigh instructions did not merge with the leading system message");
    return failures;
}

int test_think_control_tags() {
    const fi::ChatRenderOptions default_options;
    int failures = 0;
    failures += check(
        v224_template()
                .render({chat_message(ninfer::ChatRole::User, "<|think_off|>hello")},
                        default_options)
                .text == "<|im_start|>user\nhello<|im_end|>\n"
                         "<|im_start|>assistant\n<think>\n\n</think>\n\n",
        "the <|think_off|> tag did not disable the generation opener");
    failures += check(
        v224_template()
                .render({chat_message(ninfer::ChatRole::System, "<|think_off|>Be concise."),
                         chat_message(ninfer::ChatRole::User, "hi")},
                        default_options)
                .text == "<|im_start|>system\nBe concise.<|im_end|>\n"
                         "<|im_start|>user\nhi<|im_end|>\n"
                         "<|im_start|>assistant\n<think>\n\n</think>\n\n",
        "a system think-off tag was not stripped before merging the head message");
    return failures;
}

int test_head_system_merge() {
    fi::ChatRenderOptions no_generation;
    int failures = 0;
    no_generation.add_generation_prompt = false;
    failures += check(
        v224_template()
                .render({chat_message(ninfer::ChatRole::System, "first"),
                         chat_message(ninfer::ChatRole::System, "second"),
                         chat_message(ninfer::ChatRole::User, "hi")},
                        no_generation)
                .text == "<|im_start|>system\nfirst\n\nsecond<|im_end|>\n"
                         "<|im_start|>user\nhi<|im_end|>\n",
        "leading system messages did not merge into a single system turn");
    return failures;
}

int test_assistant_reasoning_handling() {
    fi::ChatRenderOptions no_generation;
    int failures = 0;
    no_generation.add_generation_prompt = false;
    failures += check(
        v224_template()
                .render({chat_message(ninfer::ChatRole::User, "q1"),
                         chat_message(ninfer::ChatRole::Assistant,
                                     "<think>\nWait, let me check.\n</think>\nFinal answer.")},
                        no_generation)
                .text == "<|im_start|>user\nq1<|im_end|>\n"
                         "<|im_start|>assistant\n<think>\nWait, let me check.\n</think>\n\n"
                         "Final answer.<|im_end|>\n",
        "a derived <think> block was not split and preserved by default");

    fi::ChatMessage explicit_reasoning =
        chat_message(ninfer::ChatRole::Assistant, "<think>raw</think>body");
    explicit_reasoning.reasoning_content = "r";
    const std::string explicit_text =
        v224_template()
            .render({chat_message(ninfer::ChatRole::User, "q1"), explicit_reasoning}, no_generation)
            .text;
    failures += check(explicit_text == "<|im_start|>user\nq1<|im_end|>\n"
                                       "<|im_start|>assistant\n<think>\nr\n</think>\n\n"
                                       "body<|im_end|>\n",
                      "an explicit reasoning channel did not drop the think lead of the content");

    fi::ChatRenderOptions dropped;
    dropped.add_generation_prompt = false;
    dropped.preserve_thinking     = false;
    fi::ChatMessage old            = chat_message(ninfer::ChatRole::Assistant, "old answer");
    old.reasoning_content          = "old thought";
    const std::string dropped_text = v224_template()
                                         .render({chat_message(ninfer::ChatRole::User, "q1"), old,
                                                  chat_message(ninfer::ChatRole::User, "q2")},
                                                 dropped)
                                         .text;
    failures += check(dropped_text.find("old thought") == std::string::npos &&
                          dropped_text.find("old answer") != std::string::npos,
                      "preserve_thinking=false kept a reasoning block before the last query");
    return failures;
}

int test_xml_tool_calls() {
    fi::ChatRenderOptions no_generation;
    int failures = 0;
    no_generation.add_generation_prompt = false;
    const fi::ChatMessage call =
        tool_calling_assistant("plan", "", "lookup", R"({"city":"Paris"})");
    const std::string text =
        v224_template()
            .render({chat_message(ninfer::ChatRole::User, "call"), call}, no_generation)
            .text;
    failures += check(
        text.ends_with("<tool_call>\n<function=lookup>\n<parameter=city>\nParis\n</parameter>\n"
                       "</function>\n</tool_call><|im_end|>\n"),
        "assistant tool arguments did not render as xml parameters");

    const fi::ChatMessage parameterless =
        tool_calling_assistant("", "", "f", R"({})");
    const std::string parameterless_text =
        v224_template()
            .render({chat_message(ninfer::ChatRole::User, "call"), parameterless}, no_generation)
            .text;
    failures += check(
        parameterless_text.ends_with("<tool_call>\n<function=f>\n</function>\n"
                                     "</tool_call><|im_end|>\n"),
        "an empty arguments object did not render as a parameter-less xml call");
    return failures;
}

int test_consecutive_tool_failures() {
    fi::ChatRenderOptions no_generation;
    int failures = 0;
    no_generation.add_generation_prompt = false;
    const fi::ChatMessage call =
        tool_calling_assistant("thought", "", "lookup", R"({"city":"Paris"})");
    const std::string text =
        v224_template()
            .render({chat_message(ninfer::ChatRole::User, "q1"), call,
                     chat_message(ninfer::ChatRole::Tool, R"({"error": "boom"})"),
                     chat_message(ninfer::ChatRole::Tool, "err! again")},
                    no_generation)
            .text;
    failures += check(
        text.find("\n<tool_response>\n"
                  "{\"error\": \"boom\"}\n"
                  "\n\u26a0\ufe0f SYSTEM WARNING: The previous tool call returned an error. "
                  "Diagnose the "
                  "failure and retry with completely corrected arguments.\n"
                  "</tool_response>\n"
                  "<tool_response>\n"
                  "err! again\n"
                  "\n\u26a0\ufe0f SYSTEM WARNING: 2 consecutive tool errors detected. Your "
                  "previous "
                  "approach is incorrect. You MUST use a fundamentally different approach or "
                  "corrected arguments.\n"
                  "</tool_response><|im_end|>\n") != std::string::npos,
        "consecutive tool failures did not escalate the system warnings");
    return failures;
}

int test_message_boundaries() {
    fi::ChatRenderOptions no_generation;
    int failures = 0;
    no_generation.add_generation_prompt = false;
    fi::ChatMessage old                 = chat_message(ninfer::ChatRole::Assistant, "answer");
    old.reasoning_content               = "thought";
    const fi::RenderedChat rendered = v224_template().render(
        {chat_message(ninfer::ChatRole::User, "q1"), old,
         chat_message(ninfer::ChatRole::User, "q2")},
        no_generation);
    failures += check(rendered.message_boundaries.size() == 4 && rendered.message_boundaries[0] &&
                          *rendered.message_boundaries[0] == 0 && rendered.message_boundaries[1] &&
                          rendered.message_boundaries[2] && rendered.message_boundaries[3] &&
                          *rendered.message_boundaries[1] < *rendered.message_boundaries[2] &&
                          *rendered.message_boundaries[2] < *rendered.message_boundaries[3] &&
                          *rendered.message_boundaries[3] == rendered.text.size(),
                      "v22_4 message boundaries did not track the rendered turns");

    fi::ChatRenderOptions with_suffix;
    const fi::RenderedChat suffixed = v224_template().render(
        {chat_message(ninfer::ChatRole::System, "s1"),
         chat_message(ninfer::ChatRole::System, "s2"),
         chat_message(ninfer::ChatRole::User, "hi")},
        with_suffix);
    failures += check(
        suffixed.message_boundaries.size() == 4 && !suffixed.message_boundaries[0] &&
            !suffixed.message_boundaries[1] && suffixed.message_boundaries[2] &&
            *suffixed.message_boundaries[2] ==
                suffixed.text.find("<|im_start|>user\nhi<|im_end|>") &&
            suffixed.message_boundaries[3] &&
            suffixed.text.substr(*suffixed.message_boundaries[3]) ==
                "<|im_start|>assistant\n<think>\n",
        "folded head boundaries were not reported after the merged system turn");
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += test_registered_template_resolution();
    failures += test_default_render();
    failures += test_reasoning_instructions();
    failures += test_think_control_tags();
    failures += test_head_system_merge();
    failures += test_assistant_reasoning_handling();
    failures += test_xml_tool_calls();
    failures += test_consecutive_tool_failures();
    failures += test_message_boundaries();
    return failures;
}
