#include "targets/qwen3_6/impl/frontend/chat_template.h"

#include "targets/qwen3_6/impl/frontend/digest.h"

#include <minja/minja.hpp>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>

namespace ninfer::targets::qwen3_6::frontend_internal {
namespace {

using OrderedJson = nlohmann::ordered_json;

constexpr Sha256Digest kThinkingToggleTemplateDigest{
    0xe8, 0x4f, 0x32, 0xa2, 0x3f, 0xdd, 0xa2, 0x76, 0x89, 0xf8, 0x68, 0xaa, 0x4a, 0x1a, 0x56, 0x21,
    0xf4, 0x11, 0x33, 0xe5, 0x1a, 0x48, 0xd7, 0xf3, 0xef, 0xcb, 0xea, 0x28, 0x39, 0x57, 0x42, 0x59,
};

constexpr Sha256Digest kReasoningEffortTemplateDigest{
    0xc3, 0xcf, 0x9e, 0x34, 0xab, 0xf4, 0xf9, 0xe3, 0x6c, 0x2d, 0x72, 0x16, 0x5a, 0xa9, 0xc1, 0x32,
    0xd3, 0xe2, 0xa7, 0x25, 0xb6, 0xc2, 0x58, 0x6a, 0xaa, 0x3a, 0x8a, 0xf9, 0xd7, 0xa8, 0x10, 0x41,
};

constexpr std::string_view kLowReasoningInstructions =
    "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to "
    "the conclusion without unnecessary elaboration.";

constexpr std::string_view kXHighReasoningInstructions =
    "Reasoning effort is set to xhigh. Please think carefully through the task, validate key "
    "assumptions, consider plausible alternatives, and prioritize correctness, consistency, and "
    "clarity in the final answer.";

bool is_instruction_role(ChatRole role) noexcept {
    return role == ChatRole::System || role == ChatRole::Developer;
}

void validate_instruction_message(const ChatMessage& message) {
    if (message.has_media()) {
        throw std::invalid_argument(
            "system and developer messages cannot contain images or videos");
    }
    if (!message.reasoning_content.empty() || !message.tool_calls.empty() ||
        !message.tool_call_id.empty()) {
        throw std::invalid_argument("system and developer messages may contain only text content");
    }
}

void append_literal_span(std::vector<ByteSpan>& spans, ByteSpan span) {
    if (span.begin == span.end) { return; }
    if (!spans.empty() && spans.back().end == span.begin) {
        spans.back().end = span.end;
        return;
    }
    if (!spans.empty() && spans.back().end > span.begin) {
        throw std::logic_error("rendered literal byte spans overlap");
    }
    spans.push_back(span);
}

class RenderBuilder {
public:
    void append_template(std::string_view text) { fragment_.text += text; }

    void append_literal(std::string_view text) {
        const std::size_t begin = fragment_.text.size();
        fragment_.text += text;
        append_literal_span(fragment_.literal_spans, ByteSpan{begin, fragment_.text.size()});
    }

    void append_media_placeholder(std::string_view text, Modality modality,
                                  std::size_t item_index) {
        const std::size_t begin = fragment_.text.size();
        fragment_.text += text;
        fragment_.media_placeholders.push_back(MediaPlaceholderByteSpec{
            .bytes      = ByteSpan{begin, fragment_.text.size()},
            .modality   = modality,
            .item_index = item_index,
        });
    }

    void append(RenderedFragment fragment) {
        const std::size_t offset = fragment_.text.size();
        fragment_.text += fragment.text;
        for (const ByteSpan span : fragment.literal_spans) {
            append_literal_span(fragment_.literal_spans,
                                ByteSpan{offset + span.begin, offset + span.end});
        }
        for (MediaPlaceholderByteSpec placeholder : fragment.media_placeholders) {
            placeholder.bytes.begin += offset;
            placeholder.bytes.end += offset;
            fragment_.media_placeholders.push_back(placeholder);
        }
    }

    [[nodiscard]] std::size_t size() const noexcept { return fragment_.text.size(); }

    [[nodiscard]] RenderedFragment release() && { return std::move(fragment_); }

private:
    RenderedFragment fragment_;
};

RenderedFragment literal_fragment(std::string text) {
    RenderBuilder builder;
    builder.append_literal(text);
    return std::move(builder).release();
}

std::pair<std::size_t, std::size_t> trim_ascii_whitespace_bounds(std::string_view text) {
    std::size_t begin = 0;
    while (begin < text.size() && std::isspace(static_cast<unsigned char>(text[begin])) != 0) {
        ++begin;
    }

    std::size_t end = text.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1])) != 0) { --end; }
    return {begin, end};
}

RenderedFragment slice_fragment(const RenderedFragment& source, std::size_t begin,
                                std::size_t end) {
    if (begin > end || end > source.text.size()) {
        throw std::logic_error("rendered fragment slice is out of range");
    }
    RenderedFragment result;
    result.text = source.text.substr(begin, end - begin);
    for (const ByteSpan span : source.literal_spans) {
        const std::size_t clipped_begin = std::max(span.begin, begin);
        const std::size_t clipped_end   = std::min(span.end, end);
        if (clipped_begin < clipped_end) {
            append_literal_span(result.literal_spans,
                                ByteSpan{clipped_begin - begin, clipped_end - begin});
        }
    }
    for (MediaPlaceholderByteSpec placeholder : source.media_placeholders) {
        if (placeholder.bytes.end <= begin || placeholder.bytes.begin >= end) { continue; }
        if (placeholder.bytes.begin < begin || placeholder.bytes.end > end) {
            throw std::logic_error("rendered fragment slice intersects a media placeholder");
        }
        placeholder.bytes.begin -= begin;
        placeholder.bytes.end -= begin;
        result.media_placeholders.push_back(placeholder);
    }
    return result;
}

RenderedFragment trim_ascii_whitespace(const RenderedFragment& fragment) {
    const auto [begin, end] = trim_ascii_whitespace_bounds(fragment.text);
    return slice_fragment(fragment, begin, end);
}

bool starts_with(const std::string& text, std::string_view prefix) {
    return text.size() >= prefix.size() && text.compare(0, prefix.size(), prefix) == 0;
}

bool ends_with(const std::string& text, std::string_view suffix) {
    return text.size() >= suffix.size() &&
           text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

long last_real_user_query(const std::vector<ChatMessage>& messages) {
    long trailing_tool_query = -1;
    for (long i = static_cast<long>(messages.size()) - 1; i >= 0; --i) {
        const ChatMessage& message = messages[static_cast<std::size_t>(i)];
        if (message.role == ChatRole::Tool && trailing_tool_query < 0) { trailing_tool_query = i; }
        if (message.role != ChatRole::User) { continue; }
        const RenderedFragment content = trim_ascii_whitespace(message.rendered_content());
        if (!(starts_with(content.text, "<tool_response>") &&
              ends_with(content.text, "</tool_response>"))) {
            return i;
        }
    }
    // A truncated imported history may begin at the tool-result turn. That result is the current
    // query even though the user message that initiated the omitted tool call is not visible.
    if (trailing_tool_query >= 0) { return trailing_tool_query; }
    throw std::invalid_argument("no user query found in chat messages");
}

// Split an assistant turn into (reasoning, content) exactly as the Qwen3.6 jinja
// does when reasoning_content is not provided: reasoning is the text between the
// last <think> and the first </think>; content is everything after the last
// </think>. When there is no </think> the whole thing is content and reasoning is
// empty.
struct ThinkParts {
    RenderedFragment reasoning;
    RenderedFragment content;
};

ThinkParts derive_think_parts(const RenderedFragment& content) {
    ThinkParts parts;
    const std::size_t first_close = content.text.find("</think>");
    if (first_close == std::string::npos) {
        parts.content = content;
        return parts;
    }
    // reasoning = content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n')
    std::size_t before_end = first_close;
    while (before_end > 0 && content.text[before_end - 1] == '\n') { --before_end; }
    const std::size_t last_open = content.text.substr(0, before_end).rfind("<think>");
    std::size_t reasoning_begin =
        last_open == std::string::npos ? 0 : last_open + std::string("<think>").size();
    while (reasoning_begin < before_end && content.text[reasoning_begin] == '\n') {
        ++reasoning_begin;
    }
    parts.reasoning = slice_fragment(content, reasoning_begin, before_end);
    // content = content.split('</think>')[-1].lstrip('\n')
    const std::size_t last_close = content.text.rfind("</think>");
    std::size_t content_begin    = last_close + std::string("</think>").size();
    while (content_begin < content.text.size() && content.text[content_begin] == '\n') {
        ++content_begin;
    }
    parts.content = slice_fragment(content, content_begin, content.text.size());
    return parts;
}

constexpr std::string_view kToolInstructions =
    "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
    "<tool_call>\n"
    "<function=example_function_name>\n"
    "<parameter=example_parameter_1>\n"
    "value_1\n"
    "</parameter>\n"
    "<parameter=example_parameter_2>\n"
    "This is the value for the second parameter\n"
    "that can span\n"
    "multiple lines\n"
    "</parameter>\n"
    "</function>\n"
    "</tool_call>\n\n"
    "<IMPORTANT>\n"
    "Reminder:\n"
    "- Function calls MUST follow the specified format: an inner <function=...></function> block "
    "must be nested within <tool_call></tool_call> XML tags\n"
    "- Required parameters MUST be specified\n"
    "- You may provide optional reasoning for your function call in natural language BEFORE the "
    "function call, but NOT after\n"
    "- If there is no function call available, answer the question like normal with your current "
    "knowledge and do not tell the user about function calls\n"
    "</IMPORTANT>";

std::string tojson_text(const OrderedJson& value) {
    if (value.is_array()) {
        std::string rendered = "[";
        for (std::size_t index = 0; index < value.size(); ++index) {
            if (index != 0) { rendered += ", "; }
            rendered += tojson_text(value[index]);
        }
        rendered += "]";
        return rendered;
    }
    if (value.is_object()) {
        std::string rendered = "{";
        std::size_t index    = 0;
        for (auto it = value.begin(); it != value.end(); ++it, ++index) {
            if (index != 0) { rendered += ", "; }
            rendered += OrderedJson(it.key()).dump();
            rendered += ": ";
            rendered += tojson_text(it.value());
        }
        rendered += "}";
        return rendered;
    }
    return value.dump();
}

std::string parameter_text(const OrderedJson& value) {
    if (value.is_string()) { return value.get<std::string>(); }
    return tojson_text(value);
}

RenderedFragment render_tool_call(const ToolCall& call, bool allow_empty_arguments) {
    RenderBuilder rendered;
    if (allow_empty_arguments && call.arguments_json.empty()) {
        rendered.append_template("<tool_call>\n<function=");
        rendered.append_literal(call.name);
        rendered.append_template(">\n</function>\n</tool_call>");
        return std::move(rendered).release();
    }
    OrderedJson args = OrderedJson::parse(call.arguments_json);
    if (!args.is_object()) {
        throw std::invalid_argument("tool call arguments must be a JSON object");
    }

    rendered.append_template("<tool_call>\n<function=");
    rendered.append_literal(call.name);
    rendered.append_template(">\n");
    for (auto it = args.begin(); it != args.end(); ++it) {
        rendered.append_template("<parameter=");
        rendered.append_literal(it.key());
        rendered.append_template(">\n");
        rendered.append_literal(parameter_text(it.value()));
        rendered.append_template("\n</parameter>\n");
    }
    rendered.append_template("</function>\n</tool_call>");
    return std::move(rendered).release();
}

struct RenderedToolsSystemBlock {
    RenderedFragment fragment;
    std::vector<std::size_t> tool_boundaries;
    std::optional<std::size_t> instruction_begin;
};

RenderedToolsSystemBlock render_tools_system_block(const std::vector<std::string>& tool_jsons,
                                                   const RenderedFragment& leading_instruction,
                                                   std::string_view reasoning_instructions) {
    RenderedToolsSystemBlock out;
    RenderBuilder rendered;
    out.tool_boundaries.reserve(tool_jsons.size());
    rendered.append_template("<|im_start|>system\n");
    if (!reasoning_instructions.empty()) {
        rendered.append_template(reasoning_instructions);
        rendered.append_template("\n\n");
    }
    rendered.append_template("# Tools\n\nYou have access to the following functions:\n\n<tools>");
    for (const std::string& tool : tool_jsons) {
        rendered.append_template("\n");
        rendered.append_literal(tojson_text(OrderedJson::parse(tool)));
        out.tool_boundaries.push_back(rendered.size());
    }
    rendered.append_template("\n</tools>");
    rendered.append_template(kToolInstructions);
    if (!leading_instruction.text.empty()) {
        rendered.append_template("\n\n");
        out.instruction_begin = rendered.size();
        rendered.append(leading_instruction);
    }
    rendered.append_template("<|im_end|>\n");
    out.fragment = std::move(rendered).release();
    return out;
}

std::string_view resolve_reasoning_instructions(ChatTemplateSemantics semantics,
                                                const ChatRenderOptions& options) {
    if (semantics == ChatTemplateSemantics::ThinkingToggle) {
        if (options.reasoning_effort) {
            throw std::invalid_argument("loaded chat template does not support reasoning effort");
        }
        return {};
    }
    if (!options.enable_thinking) {
        if (options.reasoning_effort) {
            throw std::invalid_argument(
                "reasoning effort cannot be combined with disabled thinking");
        }
        return {};
    }

    switch (options.reasoning_effort.value_or(ReasoningEffort::XHigh)) {
    case ReasoningEffort::Low:
        return kLowReasoningInstructions;
    case ReasoningEffort::Medium:
        return {};
    case ReasoningEffort::XHigh:
        return kXHighReasoningInstructions;
    }
    throw std::invalid_argument("invalid reasoning effort");
}

std::string render_role(ChatRole role) {
    switch (role) {
    case ChatRole::System:
        return "system";
    case ChatRole::Developer:
        return "developer";
    case ChatRole::User:
        return "user";
    case ChatRole::Assistant:
        return "assistant";
    case ChatRole::Tool:
        return "tool";
    }
    throw std::invalid_argument("unsupported chat role value");
}

std::string_view render_reasoning_effort(ReasoningEffort effort) {
    switch (effort) {
    case ReasoningEffort::Low:
        return "low";
    case ReasoningEffort::Medium:
        return "medium";
    case ReasoningEffort::XHigh:
        return "xhigh";
    }
    throw std::invalid_argument("invalid reasoning effort");
}

OrderedJson render_template_content(const ChatMessage& message) {
    if (!message.has_media()) { return OrderedJson(message.rendered_content().text); }

    OrderedJson content = OrderedJson::array();
    for (const ChatPart& part : message.parts) {
        switch (part.kind) {
        case ChatPartKind::Text:
            content.push_back(OrderedJson{{"type", "text"}, {"text", part.text}});
            break;
        case ChatPartKind::Image:
            content.push_back(OrderedJson{{"type", "image"}});
            break;
        case ChatPartKind::Video:
            content.push_back(OrderedJson{{"type", "video"}});
            break;
        }
    }
    return content;
}

OrderedJson render_template_tool_calls(const ChatMessage& message) {
    OrderedJson calls = OrderedJson::array();
    for (const ToolCall& call : message.tool_calls) {
        OrderedJson arguments = OrderedJson::object();
        if (!call.arguments_json.empty()) {
            try {
                arguments = OrderedJson::parse(call.arguments_json);
            } catch (const std::exception& error) {
                throw std::invalid_argument("invalid tool call arguments: " +
                                            std::string(error.what()));
            }
            if (!arguments.is_object()) {
                throw std::invalid_argument("tool call arguments must be a JSON object");
            }
        }
        calls.push_back(OrderedJson{{"id", call.id},
                                    {"type", "function"},
                                    {"function",
                                     OrderedJson{{"name", call.name},
                                                 {"arguments", std::move(arguments)}}}});
    }
    return calls;
}

minja::Value jinja_context(const std::vector<ChatMessage>& messages,
                           const ChatRenderOptions& options) {
    OrderedJson jinja_messages = OrderedJson::array();
    for (const ChatMessage& message : messages) {
        jinja_messages.push_back(OrderedJson{{"role", render_role(message.role)},
                                              {"content", render_template_content(message)},
                                              {"reasoning_content", message.reasoning_content},
                                              {"tool_calls", render_template_tool_calls(message)},
                                              {"tool_call_id", message.tool_call_id}});
    }

    OrderedJson tools = OrderedJson::array();
    for (const std::string& tool : options.tool_jsons) {
        try {
            tools.push_back(OrderedJson::parse(tool));
        } catch (const std::exception& error) {
            throw std::invalid_argument("invalid tool JSON: " + std::string(error.what()));
        }
    }

    OrderedJson values{{"messages", std::move(jinja_messages)},
                       {"tools", std::move(tools)},
                       {"add_generation_prompt", options.add_generation_prompt},
                       {"enable_thinking", options.enable_thinking},
                       {"add_vision_id", options.add_vision_id},
                       {"chat_template_kwargs", OrderedJson::object()}};
    if (options.reasoning_effort) {
        values["reasoning_effort"] = render_reasoning_effort(*options.reasoning_effort);
    }
    if (options.preserve_thinking) {
        values["preserve_thinking"] = *options.preserve_thinking;
        values["chat_template_kwargs"]["preserve_thinking"] = *options.preserve_thinking;
    }
    return minja::Value(values);
}

} // namespace
namespace {

// Local hardening (upstream PR #42 review, P2): strip Jinja comments before
// capability detection so a token mentioned only in a comment ({# #} block
// comments or # line comments inside {{ }} / {% %} blocks) is not counted as
// template usage. Heuristic: string literals are respected, line comments run
// to end of line.
std::string strip_jinja_comments(std::string source) {
    // 1) {# ... #} block comments (may span multiple lines).
    size_t pos = 0;
    while (true) {
        const size_t begin = source.find("{#", pos);
        if (begin == std::string::npos) { break; }
        const size_t end = source.find("#}", begin + 2);
        if (end == std::string::npos) {
            source.erase(begin);  // Unterminated comment; template compilation would have rejected it anyway.
            break;
        }
        source.erase(begin, (end + 2) - begin);
    }

    // 2) # line comments inside {{ ... }} / {% ... %} blocks.
    size_t scan = 0;
    while (scan < source.size()) {
        const size_t brace = source.find('{', scan);
        if (brace == std::string::npos) { break; }
        const size_t open = brace + 1;
        if (open < source.size() && (source[open] == '{' || source[open] == '%')) {
            const char close_char = source[open] == '{' ? '}' : '%';
            // Locate the block end, skipping string literals and # comments (which run to end of line).
            size_t end = open + 1;
            bool in_single = false;
            bool in_double = false;
            while (end < source.size()) {
                const char c = source[end];
                if (in_single) {
                    if (c == '\'') { in_single = false; }
                    ++end;
                    continue;
                }
                if (in_double) {
                    if (c == '"') { in_double = false; }
                    ++end;
                    continue;
                }
                if (c == '\'') { in_single = true; ++end; continue; }
                if (c == '"') { in_double = true; ++end; continue; }
                if (c == '#') {
                    const size_t newline = source.find('\n', end);
                    end = (newline == std::string::npos) ? source.size() : newline + 1;
                    in_single = in_double = false;
                    continue;
                }
                if (c == close_char && end + 1 < source.size() && source[end + 1] == close_char) { break; }
                ++end;
            }
            const size_t body_end = (end < source.size()) ? end : source.size();
            // Blank comment text with spaces line by line so offsets stay stable.
            for (size_t i = brace + 2; i < body_end;) {
                if (source[i] == '\n') { ++i; continue; }
                const size_t line_end = source.find('\n', i);
                const size_t line_stop = (line_end == std::string::npos) ? source.size() : line_end;
                size_t hash = std::string::npos;
                bool s1 = false, s2 = false;
                for (size_t j = i; j < line_stop; ++j) {
                    const char c = source[j];
                    if (s1) { if (c == '\'') { s1 = false; } }
                    else if (s2) { if (c == '"') { s2 = false; } }
                    else if (c == '\'') { s1 = true; }
                    else if (c == '"') { s2 = true; }
                    else if (c == '#') { hash = j; break; }
                }
                if (hash != std::string::npos) {
                    for (size_t j = hash; j < line_stop; ++j) { source[j] = ' '; }
                }
                i = line_end;
            }
            scan = (end + 2 < source.size()) ? end + 2 : source.size();
        } else {
            scan = brace + 1;
        }
    }
    return source;
}

}  // namespace

bool prompt_starts_in_reasoning(std::string_view text) {
    struct MarkerPair
    {
        std::string_view open;
        std::string_view close;
    };
    static constexpr MarkerPair kMarkers[] = {
        {"<think>", "</think>"},
        {"<|think|>", "</|think|>"},
        {"<thinking>", "</thinking>"},
    };
    bool       found        = false;
    bool       last_is_open = false;
    std::size_t last_pos    = 0;
    for (const auto& marker : kMarkers) {
        const std::string_view pieces[2] = {marker.open, marker.close};
        for (const std::string_view piece : pieces) {
            const auto pos = text.rfind(piece);
            if (pos == std::string_view::npos) { continue; }
            if (!found || pos > last_pos) {
                last_pos     = pos;
                last_is_open = (piece == marker.open);
                found        = true;
            }
        }
    }
    return found && last_is_open;
}

class CompiledChatTemplate::JinjaTemplate {
public:
    JinjaTemplate(std::string source, std::string source_name)
        : source_name_(std::move(source_name)),
          supports_reasoning_effort_(strip_jinja_comments(source).find("reasoning_effort") != std::string::npos) {
        try {
            template_ = minja::Parser::parse(source, minja::Options{.trim_blocks           = true,
                                                                      .lstrip_blocks         = true,
                                                                      .keep_trailing_newline = false});
        } catch (const std::exception& error) {
            throw std::invalid_argument("failed to compile Jinja chat template '" + source_name_ +
                                        "': " + error.what());
        }
    }

    [[nodiscard]] RenderedChat render(const std::vector<ChatMessage>& messages,
                                      const ChatRenderOptions& options) const {
        if (options.reasoning_effort && !supports_reasoning_effort_) {
            throw std::invalid_argument(
            "custom Jinja chat template does not reference reasoning_effort");
        }
        try {
            RenderedChat rendered;
            rendered.text =
                template_->render(minja::Context::make(jinja_context(messages, options)));
            // A custom template exposes no byte frontier per message or cache marker:
            // report honest sizes with no resolved offsets so size guards downstream
            // (frontend context-cache preparation, media encoding) stay satisfied and
            // marker resolution degrades to "no independent boundary" instead of
            // failing on empty vectors.
            rendered.message_boundaries.assign(messages.size() + 1U, std::nullopt);
            rendered.cache_boundaries.assign(options.cache_markers.size(), std::nullopt);
            // A custom Jinja template is an opaque byte generator, so unlike the builtin
            // renderer the exact frontier that ends the last real user query cannot be
            // pointed at directly. Probe that frontier by rendering the message prefix up
            // to and including the query (the whole history when thinking is preserved)
            // without the generation suffix, then publish a rewrite checkpoint at the
            // probe boundary -- but only when the probe text is a byte prefix of the full
            // render. Templates whose output depends on later messages fail the probe and
            // keep no checkpoint, degrading to full-prompt reuse exactly as before.
            rendered.rewrite_checkpoint =
                probe_rewrite_checkpoint(messages, options, rendered.text);
            return rendered;
        } catch (const std::exception& error) {
            throw std::invalid_argument("failed to render Jinja chat template '" + source_name_ +
                                        "': " + error.what());
        }
    }

    [[nodiscard]] PromptCapabilities capabilities() const noexcept {
        PromptCapabilities result;
        result.enable_thinking = true;
        if (supports_reasoning_effort_) {
            result.reasoning_effort.low            = true;
            result.reasoning_effort.medium         = true;
            result.reasoning_effort.xhigh          = true;
            result.reasoning_effort.default_effort = ReasoningEffort::Medium;
        }
        return result;
    }

private:
    // Mirror the builtin renderer's closed-turn publication policy on an opaque custom
    // template: a request that closes the turn and appends a generation prompt publishes
    // one rewrite checkpoint so a successor may reuse the stable history. When thinking is
    // preserved the successor replays the response from the generation boundary
    // (ResponseReplay); otherwise the caller may close the turn and branch with a new user
    // message, which rewrites everything rendered after the last real user query
    // (TurnClosure). Both boundaries are probed by re-rendering without the generation
    // suffix -- the message prefix up to and including the last real user query for the
    // turn-closure point, the whole history for the response-replay point.
    [[nodiscard]] std::optional<RewriteCheckpointByteSpec>
    probe_rewrite_checkpoint(const std::vector<ChatMessage>& messages,
                             const ChatRenderOptions& options,
                             const std::string& full_text) const {
        if (options.continuation == PromptContinuationMode::ContinueFinalAssistant ||
            !options.add_generation_prompt) {
            return std::nullopt;
        }
        try {
            const bool preserve_thinking = options.preserve_thinking.value_or(false);
            ChatRenderOptions probe_options = options;
            probe_options.add_generation_prompt = false;
            std::size_t probe_end = messages.size();
            if (!preserve_thinking) {
                try {
                    probe_end =
                        static_cast<std::size_t>(last_real_user_query(messages)) + 1U;
                } catch (const std::invalid_argument&) {
                    // The history contains no user query yet the template accepted it:
                    // there is no stable branch point, so probe the whole history.
                    probe_end = messages.size();
                }
            }
            const std::string probe_text = template_->render(minja::Context::make(
                jinja_context(std::vector<ChatMessage>(messages.begin(),
                                                       messages.begin() +
                                                           static_cast<std::ptrdiff_t>(probe_end)),
                              probe_options)));
            if (probe_text.empty() || probe_text.size() > full_text.size() ||
                !starts_with(full_text, probe_text)) {
                return std::nullopt;
            }
            return RewriteCheckpointByteSpec{
                .kind   = preserve_thinking ? RewriteCheckpointKind::ResponseReplay
                                            : RewriteCheckpointKind::TurnClosure,
                .offset = probe_text.size()};
        } catch (const std::exception&) {
            // A failing probe must never fail a successful full render.
            return std::nullopt;
        }
    }

    std::string source_name_;
    bool supports_reasoning_effort_ = false;
    std::shared_ptr<minja::TemplateNode> template_;
};

bool ChatMessage::has_media() const noexcept {
    for (const ChatPart& part : parts) {
        if (part.kind != ChatPartKind::Text) { return true; }
    }
    return false;
}

RenderedFragment ChatMessage::rendered_content(bool add_vision_id, int* image_count,
                                               int* video_count, std::size_t* media_count,
                                               std::vector<std::size_t>* part_boundaries) const {
    int local_images        = 0;
    int local_videos        = 0;
    std::size_t local_media = 0;
    int& images             = image_count == nullptr ? local_images : *image_count;
    int& videos             = video_count == nullptr ? local_videos : *video_count;
    std::size_t& media      = media_count == nullptr ? local_media : *media_count;
    RenderBuilder out;
    if (part_boundaries != nullptr) {
        part_boundaries->clear();
        part_boundaries->reserve(parts.size());
    }
    for (const ChatPart& part : parts) {
        switch (part.kind) {
        case ChatPartKind::Text:
            out.append_literal(part.text);
            break;
        case ChatPartKind::Image:
            ++images;
            if (add_vision_id) { out.append_template("Picture " + std::to_string(images) + ": "); }
            out.append_template("<|vision_start|>");
            out.append_media_placeholder("<|image_pad|>", Modality::Image, media++);
            out.append_template("<|vision_end|>");
            break;
        case ChatPartKind::Video:
            ++videos;
            if (add_vision_id) { out.append_template("Video " + std::to_string(videos) + ": "); }
            out.append_template("<|vision_start|>");
            out.append_media_placeholder("<|video_pad|>", Modality::Video, media++);
            out.append_template("<|vision_end|>");
            break;
        }
        if (part_boundaries != nullptr) { part_boundaries->push_back(out.size()); }
    }
    return std::move(out).release();
}

CompiledChatTemplate CompiledChatTemplate::resolve(std::string_view source) {
    const Sha256Digest digest = sha256(source);
    if (digest == kThinkingToggleTemplateDigest) {
        return CompiledChatTemplate(ChatTemplateSemantics::ThinkingToggle);
    }
    if (digest == kReasoningEffortTemplateDigest) {
        return CompiledChatTemplate(ChatTemplateSemantics::ReasoningEffort);
    }
    throw std::invalid_argument("unsupported frontend/chat_template.jinja (sha256 " +
                                sha256_hex(digest) + ")");
}

CompiledChatTemplate CompiledChatTemplate::compile_jinja(std::string source,
                                                         std::string source_name) {
    if (source.empty()) {
        throw std::invalid_argument("Jinja chat template '" + source_name + "' is empty");
    }
    return CompiledChatTemplate(
        std::make_shared<const JinjaTemplate>(std::move(source), std::move(source_name)));
}

PromptCapabilities CompiledChatTemplate::capabilities() const noexcept {
    if (jinja_template_) { return jinja_template_->capabilities(); }
    PromptCapabilities result;
    result.enable_thinking = true;
    if (semantics_ == ChatTemplateSemantics::ReasoningEffort) {
        result.reasoning_effort.low            = true;
        result.reasoning_effort.medium         = true;
        result.reasoning_effort.xhigh          = true;
        result.reasoning_effort.default_effort = ReasoningEffort::XHigh;
    }
    return result;
}

RenderedChat CompiledChatTemplate::render(const std::vector<ChatMessage>& messages,
                                          ChatRenderOptions options) const {
    if (messages.empty()) { throw std::invalid_argument("chat messages must not be empty"); }
    if (jinja_template_) { return jinja_template_->render(messages, options); }

    const bool continue_final_assistant =
        options.continuation == PromptContinuationMode::ContinueFinalAssistant;
    if (continue_final_assistant) {
        const ChatMessage& final = messages.back();
        if (final.role != ChatRole::Assistant || final.parts.empty() ||
            !final.reasoning_content.empty() || !final.tool_calls.empty()) {
            throw std::invalid_argument(
                "assistant continuation requires a final text-only assistant message");
        }
        if (final.has_media()) {
            throw std::invalid_argument(
                "assistant continuation requires a final text-only assistant message");
        }
        if (options.enable_thinking) {
            throw std::invalid_argument("assistant continuation cannot start in thinking mode");
        }
    }

    const bool effort_template = semantics_ == ChatTemplateSemantics::ReasoningEffort;
    const std::string_view reasoning_instructions =
        resolve_reasoning_instructions(semantics_, options);

    std::size_t message_begin = 0;
    RenderedFragment leading_instruction_raw;
    RenderedFragment leading_instruction;
    std::size_t leading_trim_begin = 0;
    std::size_t leading_trim_end   = 0;
    if (is_instruction_role(messages[0].role)) {
        validate_instruction_message(messages[0]);
        leading_instruction_raw = messages[0].rendered_content();
        std::tie(leading_trim_begin, leading_trim_end) =
            trim_ascii_whitespace_bounds(leading_instruction_raw.text);
        leading_instruction =
            slice_fragment(leading_instruction_raw, leading_trim_begin, leading_trim_end);
        message_begin = 1;
    }

    RenderBuilder rendered;
    std::vector<std::size_t> tool_boundaries;
    std::optional<std::size_t> instruction_begin;
    const bool has_tools = !options.tool_jsons.empty();
    if (has_tools) {
        RenderedToolsSystemBlock preamble = render_tools_system_block(
            options.tool_jsons, leading_instruction, reasoning_instructions);
        rendered.append(std::move(preamble.fragment));
        tool_boundaries   = std::move(preamble.tool_boundaries);
        instruction_begin = preamble.instruction_begin;
    } else if (message_begin == 1) {
        if (!effort_template || !leading_instruction.text.empty() ||
            !reasoning_instructions.empty()) {
            rendered.append_template("<|im_start|>system\n");
            if (!reasoning_instructions.empty()) {
                rendered.append_template(reasoning_instructions);
                if (!leading_instruction.text.empty()) { rendered.append_template("\n\n"); }
            }
            if (!leading_instruction.text.empty()) { instruction_begin = rendered.size(); }
            rendered.append(leading_instruction);
            rendered.append_template("<|im_end|>\n");
        }
    } else if (!reasoning_instructions.empty()) {
        rendered.append_template("<|im_start|>system\n");
        rendered.append_template(reasoning_instructions);
        rendered.append_template("<|im_end|>\n");
    }

    std::vector<std::optional<std::size_t>> message_boundaries(messages.size() + 1U);
    std::vector<std::optional<std::size_t>> cache_boundaries(options.cache_markers.size());
    if (message_begin == 0) {
        message_boundaries[0] = rendered.size();
    } else {
        // The first instruction message is folded into the system preamble by this template.
        message_boundaries[1] = rendered.size();
    }

    const long last_query_index  = last_real_user_query(messages);
    const bool preserve_thinking = options.preserve_thinking.value_or(effort_template);
    std::optional<RewriteCheckpointByteSpec> rewrite_checkpoint;
    std::vector<std::size_t> rewrite_execution_boundaries;
    const auto add_rewrite_execution_boundary = [&] {
        if (rewrite_execution_boundaries.empty() ||
            rewrite_execution_boundaries.back() != rendered.size()) {
            rewrite_execution_boundaries.push_back(rendered.size());
        }
    };

    int image_count         = 0;
    int video_count         = 0;
    std::size_t media_count = 0;
    for (std::size_t i = 0; i < messages.size(); ++i) {
        const ChatMessage& message = messages[i];
        if (i < message_begin) { continue; }
        if (is_instruction_role(message.role)) { validate_instruction_message(message); }
        std::vector<std::size_t> raw_part_boundaries;
        const RenderedFragment raw_content = message.rendered_content(
            options.add_vision_id, &image_count, &video_count, &media_count, &raw_part_boundaries);
        const auto [content_trim_begin, content_trim_end] =
            trim_ascii_whitespace_bounds(raw_content.text);
        const RenderedFragment content =
            slice_fragment(raw_content, content_trim_begin, content_trim_end);
        const auto resolve_part_boundaries = [&](std::size_t content_begin) {
            for (std::size_t marker_index = 0; marker_index < options.cache_markers.size();
                 ++marker_index) {
                const PromptCacheMarker& marker = options.cache_markers[marker_index];
                if (marker.location != PromptCacheMarkerLocation::MessagePartBoundary ||
                    marker.after_message_count != i + 1U || marker.after_message_part_count == 0 ||
                    marker.after_message_part_count > raw_part_boundaries.size()) {
                    continue;
                }
                const std::size_t raw = raw_part_boundaries[marker.after_message_part_count - 1U];
                const std::size_t clamped = std::clamp(raw, content_trim_begin, content_trim_end);
                cache_boundaries[marker_index] = content_begin + clamped - content_trim_begin;
            }
        };
        if (is_instruction_role(message.role)) {
            rendered.append_template("<|im_start|>system\n");
            resolve_part_boundaries(rendered.size());
            rendered.append(content);
            rendered.append_template("<|im_end|>\n");
            message_boundaries[i + 1U] = rendered.size();
            continue;
        }
        if (message.role == ChatRole::User) {
            rendered.append_template("<|im_start|>user\n");
            resolve_part_boundaries(rendered.size());
            rendered.append(content);
            rendered.append_template("<|im_end|>\n");
            message_boundaries[i + 1U] = rendered.size();
            continue;
        }
        if (message.role == ChatRole::Tool) {
            const bool opens_group = i == 0 || messages[i - 1].role != ChatRole::Tool;
            const bool closes_group =
                i + 1 == messages.size() || messages[i + 1].role != ChatRole::Tool;
            if (opens_group) { rendered.append_template("<|im_start|>user"); }
            rendered.append_template("\n<tool_response>\n");
            resolve_part_boundaries(rendered.size());
            rendered.append(content);
            rendered.append_template("\n</tool_response>");
            if (closes_group) { rendered.append_template("<|im_end|>\n"); }
            message_boundaries[i + 1U] = rendered.size();
            continue;
        }

        if (message.role != ChatRole::Assistant) {
            throw std::invalid_argument("unsupported chat role value");
        }

        // assistant
        if (continue_final_assistant && i + 1U == messages.size()) {
            const std::size_t generation_begin = rendered.size();
            rewrite_checkpoint                 = RewriteCheckpointByteSpec{
                                .kind = RewriteCheckpointKind::ResponseReplay, .offset = generation_begin};
            rendered.append_template("<|im_start|>assistant\n");
            add_rewrite_execution_boundary();
            rendered.append(content);
            message_boundaries[i + 1U] = rendered.size();
            continue;
        }
        RenderedFragment reasoning;
        RenderedFragment body = content;
        if (!message.reasoning_content.empty()) {
            reasoning = literal_fragment(message.reasoning_content);
        } else if (!effort_template) {
            ThinkParts parts = derive_think_parts(content);
            reasoning        = std::move(parts.reasoning);
            body             = std::move(parts.content);
        }
        reasoning = trim_ascii_whitespace(reasoning);

        const bool keep_thinking = preserve_thinking || (static_cast<long>(i) > last_query_index);
        if (!preserve_thinking && !rewrite_checkpoint && static_cast<long>(i) > last_query_index) {
            // Closing the current turn may rewrite everything beginning with this assistant
            // segment. Keep the stable history before the opener recoverable; retaining the
            // deterministic opener itself is not worth losing the whole prefix when a caller
            // branches with a new user message instead.
            rewrite_checkpoint = RewriteCheckpointByteSpec{
                .kind = RewriteCheckpointKind::TurnClosure, .offset = rendered.size()};
        }
        rendered.append_template("<|im_start|>assistant\n");
        add_rewrite_execution_boundary();
        if (keep_thinking) {
            rendered.append_template("<think>\n");
            add_rewrite_execution_boundary();
            rendered.append(reasoning);
            rendered.append_template("\n</think>\n\n");
            add_rewrite_execution_boundary();
        }
        rendered.append(body);
        if (!message.tool_calls.empty()) {
            const bool body_has_text = !trim_ascii_whitespace(body).text.empty();
            for (std::size_t call_index = 0; call_index < message.tool_calls.size(); ++call_index) {
                if (call_index == 0) {
                    if (body_has_text) { rendered.append_template("\n\n"); }
                } else {
                    rendered.append_template("\n");
                }
                rendered.append(render_tool_call(message.tool_calls[call_index], effort_template));
            }
        }
        rendered.append_template("<|im_end|>\n");
        message_boundaries[i + 1U] = rendered.size();
    }

    if (!continue_final_assistant && options.add_generation_prompt) {
        // The generation suffix is replaceable as a unit. An immediate successor may replay the
        // response, close the turn, or branch by appending a different user message directly to
        // the input history. The rolling private checkpoint must therefore precede the assistant
        // opener; placing it after the deterministic prologue makes the complete history
        // unrecoverable for the branch case merely to save a handful of prompt tokens.
        const std::size_t generation_begin = rendered.size();
        if (preserve_thinking) {
            rewrite_checkpoint = RewriteCheckpointByteSpec{
                .kind = RewriteCheckpointKind::ResponseReplay, .offset = generation_begin};
        } else if (!rewrite_checkpoint) {
            rewrite_checkpoint = RewriteCheckpointByteSpec{
                .kind = RewriteCheckpointKind::TurnClosure, .offset = generation_begin};
        }
        rendered.append_template("<|im_start|>assistant\n");
        add_rewrite_execution_boundary();
        if (options.enable_thinking) {
            rendered.append_template("<think>\n");
            add_rewrite_execution_boundary();
        } else {
            rendered.append_template("<think>\n");
            add_rewrite_execution_boundary();
            rendered.append_template("\n</think>\n\n");
            add_rewrite_execution_boundary();
        }
    }
    for (std::size_t index = 0; index < options.cache_markers.size(); ++index) {
        const PromptCacheMarker marker = options.cache_markers[index];
        switch (marker.location) {
        case PromptCacheMarkerLocation::MessageBoundary:
            if (marker.after_message_count < message_boundaries.size()) {
                cache_boundaries[index] = message_boundaries[marker.after_message_count];
            }
            break;
        case PromptCacheMarkerLocation::MessagePartBoundary:
            // Resolved while the containing message's content offset is known. Assistant
            // reasoning/tool-call interiors deliberately remain advisory and unresolved.
            break;
        case PromptCacheMarkerLocation::LeadingInstructionBoundary:
            if (message_begin == 1 && instruction_begin &&
                marker.leading_instruction_bytes <= leading_instruction_raw.text.size()) {
                const std::size_t clamped = std::clamp<std::size_t>(
                    marker.leading_instruction_bytes, leading_trim_begin, leading_trim_end);
                cache_boundaries[index] = *instruction_begin + clamped - leading_trim_begin;
            }
            break;
        case PromptCacheMarkerLocation::ToolBoundary:
            if (marker.after_tool_count != 0 && marker.after_tool_count <= tool_boundaries.size()) {
                cache_boundaries[index] = tool_boundaries[marker.after_tool_count - 1U];
            }
            break;
        }
    }
    RenderedFragment final = std::move(rendered).release();
    return RenderedChat{.text                         = std::move(final.text),
                        .literal_spans                = std::move(final.literal_spans),
                        .media_placeholders           = std::move(final.media_placeholders),
                        .rewrite_checkpoint           = rewrite_checkpoint,
                        .rewrite_execution_boundaries = std::move(rewrite_execution_boundaries),
                        .message_boundaries           = std::move(message_boundaries),
                        .cache_boundaries             = std::move(cache_boundaries)};
}

} // namespace ninfer::targets::qwen3_6::frontend_internal
