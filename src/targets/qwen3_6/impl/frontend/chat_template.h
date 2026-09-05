#pragma once

#include "targets/qwen3_6/impl/frontend/tokenizer.h"

#include <ninfer/targets/qwen3_6/prepared_prompt.h>
#include <ninfer/types.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::targets::qwen3_6::frontend_internal {

struct ToolCall {
    std::string id;
    std::string name;
    std::string arguments_json;
};

enum class ChatPartKind {
    Text,
    Image,
    Video,
};

enum class Modality : std::uint8_t {
    Image = 1,
    Video = 2,
};

struct MediaPlaceholderByteSpec {
    ByteSpan bytes;
    Modality modality      = Modality::Image;
    std::size_t item_index = 0;
};

struct MediaTokenRunByteSpec {
    ByteSpan bytes;
    Modality modality       = Modality::Image;
    std::size_t item_index  = 0;
    std::size_t frame_index = 0;
};

struct RenderedFragment {
    std::string text;
    std::vector<ByteSpan> literal_spans;
    std::vector<MediaPlaceholderByteSpec> media_placeholders;
};

struct MediaData {
    std::vector<std::uint8_t> bytes;
    std::string media_type;
    std::string source_name;
    ImageResizePolicy image_resize_policy = ImageResizePolicy::Downsize;
};

struct ChatPart {
    ChatPartKind kind = ChatPartKind::Text;
    std::string text;
    MediaData media;

    static ChatPart text_part(std::string value) {
        ChatPart part;
        part.text = std::move(value);
        return part;
    }

    static ChatPart image(MediaData value) {
        ChatPart part;
        part.kind  = ChatPartKind::Image;
        part.media = std::move(value);
        return part;
    }

    static ChatPart video(MediaData value) {
        ChatPart part;
        part.kind  = ChatPartKind::Video;
        part.media = std::move(value);
        return part;
    }
};

struct ChatMessage {
    ChatRole role = ChatRole::User;
    std::vector<ChatPart> parts;
    std::string reasoning_content;
    std::vector<ToolCall> tool_calls;
    std::string tool_call_id;

    [[nodiscard]] bool has_media() const noexcept;
    [[nodiscard]] RenderedFragment
    rendered_content(bool add_vision_id = false, int* image_count = nullptr,
                     int* video_count = nullptr, std::size_t* media_count = nullptr,
                     std::vector<std::size_t>* part_boundaries = nullptr) const;
};

struct ChatRenderOptions {
    PromptContinuationMode continuation = PromptContinuationMode::NewAssistantTurn;
    // Internal renderer control used by frontend qualification. Product PromptInput always
    // selects either a new assistant turn or continuation of the final assistant.
    bool add_generation_prompt = true;
    bool enable_thinking       = true;
    std::optional<ReasoningEffort> reasoning_effort;
    std::optional<bool> preserve_thinking;
    bool add_vision_id = false;
    std::vector<std::string> tool_jsons;
    std::vector<PromptCacheMarker> cache_markers;
};

struct RewriteCheckpointByteSpec {
    RewriteCheckpointKind kind = RewriteCheckpointKind::TurnClosure;
    std::size_t offset         = 0;
};

struct RenderedChat {
    std::string text;
    std::vector<ByteSpan> literal_spans;
    std::vector<MediaPlaceholderByteSpec> media_placeholders;
    std::vector<MediaTokenRunByteSpec> media_token_runs;
    std::optional<RewriteCheckpointByteSpec> rewrite_checkpoint;
    std::vector<std::size_t> rewrite_execution_boundaries;
    // Index n is the exact byte frontier after serializing the first n input messages. A missing
    // value means the template has no independent boundary there (for example, before a leading
    // instruction message folded into the system preamble).
    std::vector<std::optional<std::size_t>> message_boundaries;
    // One rendered byte boundary per requested cache marker.
    std::vector<std::optional<std::size_t>> cache_boundaries;
};
// P1 fix (upstream PR #42 review): derive from the rendered prompt text whether
// generation starts in the reasoning channel. True only when the last reasoning
// marker in the text is an *opening* tag (e.g. the default template's trailing
// "<think>\n"); pre-closed blocks (thinking disabled) and prompts with no
// reasoning markers at all (plain custom templates) start in the content
// channel, so custom Jinja templates that do not emit the default thinking
// prologue cannot misclassify the whole generation as reasoning.
[[nodiscard]] bool prompt_starts_in_reasoning(std::string_view text);


enum class ChatTemplateSemantics : std::uint8_t {
    ThinkingToggle,
    ReasoningEffort,
};

class CompiledChatTemplate {
public:
    [[nodiscard]] static CompiledChatTemplate resolve(std::string_view source);
    [[nodiscard]] static CompiledChatTemplate compile_jinja(std::string source,
                                                             std::string source_name);

    [[nodiscard]] PromptCapabilities capabilities() const noexcept;
    [[nodiscard]] RenderedChat render(const std::vector<ChatMessage>& messages,
                                      ChatRenderOptions options = {}) const;

private:
    class JinjaTemplate;

    explicit CompiledChatTemplate(ChatTemplateSemantics semantics) noexcept
        : semantics_(semantics) {}
    explicit CompiledChatTemplate(std::shared_ptr<const JinjaTemplate> jinja_template) noexcept
        : jinja_template_(std::move(jinja_template)) {}

    ChatTemplateSemantics semantics_ = ChatTemplateSemantics::ThinkingToggle;
    std::shared_ptr<const JinjaTemplate> jinja_template_;
};

} // namespace ninfer::targets::qwen3_6::frontend_internal
