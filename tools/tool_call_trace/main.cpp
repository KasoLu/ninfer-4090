// Standalone diagnostic driver for the production Qwen3.6 tool-call output parser.
//
// It compiles src/targets/qwen3_6/impl/frontend/tool_call_parser.cpp with NINFER_TOOL_CALL_TRACE
// defined, so the parser itself reports every parsing decision, and replays a captured response
// file case by case. Each case is run three ways:
//   1. declared contract (the tool schemas the client advertised) with per-stage trace;
//   2. legacy contract (no declared types), to separate contract-dependent from unconditional
//      failures;
//   3. the streaming ToolCallOutputDecoder path used by serving, to show what the client receives.
//
// Build (inside the toolchain container, repository mounted at /src):
//   /usr/bin/c++ -DNINFER_TOOL_CALL_TRACE -I/src/include -I/src/src -I/src/third_party
//       -std=gnu++20 -O1 -o /tmp/tool_call_trace
//       /src/tools/tool_call_trace/main.cpp
//       /src/src/targets/qwen3_6/impl/frontend/tool_call_parser.cpp
//
// Run:
//   /tmp/tool_call_trace /path/to/captured-responses.md
//
// The capture file is split into cases at blank lines; each case is one captured model response.

#include "targets/qwen3_6/impl/frontend/tool_call_parser.h"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>
#include <string_view>
#include <vector>

namespace tcf = ninfer::targets::qwen3_6::frontend_internal;

namespace {
bool g_trace_enabled = false;

std::string escape_preview(std::string_view text, std::size_t max_length) {
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

// Cases are the blank-line separated groups of a capture file, whitespace-trimmed. This mirrors
// how separately captured model responses were concatenated into the file. Blank lines are
// recognized with either LF or CRLF line endings.
std::vector<std::string> split_cases(const std::string& text) {
    std::vector<std::string> cases;
    std::size_t begin = 0;
    std::size_t pos   = 0;
    while (pos <= text.size()) {
        const std::size_t line_end = text.find('\n', pos);
        const std::size_t line_stop =
            line_end == std::string::npos ? text.size() : line_end;
        bool blank = true;
        for (std::size_t index = pos; index < line_stop; ++index) {
            const char byte = text[index];
            if (byte != ' ' && byte != '\t' && byte != '\r') {
                blank = false;
                break;
            }
        }
        if (blank) {
            if (line_stop > begin) {
                std::string chunk = text.substr(begin, line_stop - begin);
                const std::size_t first = chunk.find_first_not_of(" \t\r\n");
                if (first != std::string::npos) {
                    const std::size_t last = chunk.find_last_not_of(" \t\r\n");
                    cases.push_back(chunk.substr(first, last - first + 1));
                }
            }
            begin = line_end == std::string::npos ? text.size() : line_end + 1;
        }
        if (line_end == std::string::npos) { break; }
        pos = line_end + 1;
    }
    if (begin < text.size()) {
        std::string chunk = text.substr(begin);
        const std::size_t first = chunk.find_first_not_of(" \t\r\n");
        if (first != std::string::npos) {
            const std::size_t last = chunk.find_last_not_of(" \t\r\n");
            cases.push_back(chunk.substr(first, last - first + 1));
        }
    }
    return cases;
}

std::string describe(const tcf::ParsedToolCallOutput& parsed) {
    std::string result = parsed.is_tool_call_response ? "TOOL CALL" : "FALLBACK (plain text)";
    result += "; calls=" + std::to_string(parsed.tool_calls.size()) +
              "; content len=" + std::to_string(parsed.content.size());
    for (std::size_t index = 0; index < parsed.tool_calls.size(); ++index) {
        const auto& call = parsed.tool_calls[index];
        result += "\n      call[" + std::to_string(index) + "] name=" + call.name +
                  " arguments=" + escape_preview(call.arguments_json, 200);
    }
    return result;
}

const std::vector<std::string>& declared_tool_jsons() {
    static const std::vector<std::string> tool_jsons = {
        R"json({"type":"function","function":{"name":"edit","parameters":{"type":"object",)json"
        R"json("properties":{"path":{"type":"string"},"edits":{"type":"array"}}}}})json",
        R"json({"type":"function","function":{"name":"pwsh","parameters":{"type":"object",)json"
        R"json("properties":{"command":{"type":"string"},"description":{"type":"string"},)json"
        R"json("run_in_background":{"type":"boolean"}}}}})json",
        R"json({"type":"function","function":{"name":"compress","parameters":{"type":"object",)json"
        R"json("properties":{"content":{"type":"array"},"topic":{"type":"string"}}}}})json",
    };
    return tool_jsons;
}
} // namespace

namespace ninfer::targets::qwen3_6::frontend_internal {
// Sink declared by tool_call_parser.cpp when NINFER_TOOL_CALL_TRACE is defined.
void tool_call_trace(const char* stage, std::string_view detail) {
    if (!g_trace_enabled) { return; }
    std::fprintf(stdout, "    [trace] %-9s %.*s\n", stage, static_cast<int>(detail.size()),
                 detail.data());
}
} // namespace ninfer::targets::qwen3_6::frontend_internal

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: tool_call_trace <capture-file>\n");
        return 2;
    }
    const std::string path = argv[1];
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        std::fprintf(stderr, "cannot open %s\n", path.c_str());
        return 2;
    }
    const std::string text((std::istreambuf_iterator<char>(input)),
                           std::istreambuf_iterator<char>());
    const std::size_t max_name_length = 128;
    std::printf("tool-call parser trace (real parser, NINFER_TOOL_CALL_TRACE)\n");
    std::printf("input: %s (%zu bytes)\n\n", path.c_str(), text.size());

    g_trace_enabled = true;
    std::printf("--- contract construction ---\n");
    const auto contract = tcf::build_tool_call_output_contract(declared_tool_jsons(), true);
    g_trace_enabled = false;

    std::vector<std::string> cases = split_cases(text);
    cases.push_back(text); // whole file as a single input

    for (std::size_t index = 0; index < cases.size(); ++index) {
        const std::string& body = cases[index];
        const bool whole        = index + 1 == cases.size();
        const std::string label =
            whole ? std::string("whole file") : ("case " + std::to_string(index + 1));
        std::printf("\n===== %s (len %zu): \"%s\" =====\n", label.c_str(), body.size(),
                    escape_preview(body, 80).c_str());

        g_trace_enabled = true;
        std::printf("--- pass 1: declared contract (per-stage trace) ---\n");
        const tcf::ParsedToolCallOutput parsed =
            tcf::parse_qwen_tool_call_output(body, max_name_length, contract->argument_types);
        g_trace_enabled = false;
        std::printf("verdict: %s\n", describe(parsed).c_str());
        if (!parsed.is_tool_call_response) {
            std::printf("         fallback content == input: %s\n",
                        parsed.content == body ? "yes" : "no");
        }

        std::printf("--- pass 2: legacy contract (no declared types, trace) ---\n");
        tcf::ToolArgumentTypeContracts legacy;
        g_trace_enabled = true;
        const tcf::ParsedToolCallOutput legacy_parsed =
            tcf::parse_qwen_tool_call_output(body, max_name_length, legacy);
        g_trace_enabled = false;
        std::printf("verdict: %s\n", describe(legacy_parsed).c_str());

        tcf::ToolCallOutputDecoder decoder(contract, max_name_length);
        const std::string visible                           = decoder.feed(body);
        const tcf::ToolCallOutputDecoder::Terminal terminal = decoder.finish();
        std::printf("pass 3: streaming decoder: streamed %zu bytes; terminal %zu call(s), "
                    "terminal content len %zu; verbatim restore: %s\n",
                    visible.size(), terminal.tool_calls.size(), terminal.content.size(),
                    (visible + terminal.content == body) ? "yes" : "no");
    }
    return 0;
}
