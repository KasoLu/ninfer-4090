#include "options.h"

#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

ninfer::cli::Options parse(std::vector<std::string> arguments) {
    std::vector<char*> argv;
    argv.reserve(arguments.size());
    for (std::string& argument : arguments) { argv.push_back(argument.data()); }
    return ninfer::cli::parse_options(static_cast<int>(argv.size()), argv.data());
}

bool rejects(const std::function<void()>& operation) {
    try {
        operation();
    } catch (const std::invalid_argument&) { return true; }
    return false;
}

} // namespace

int main() {
    int failures = 0;

    const ninfer::cli::Options defaults =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello"});
    failures += check(defaults.chat_template_path.empty(),
                      "CLI template override is unexpectedly configured by default");

    const ninfer::cli::Options configured =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--chat-template-file",
               "templates/sharp.jinja"});
    failures += check(configured.chat_template_path == "templates/sharp.jinja",
                      "CLI template override path was not preserved");

    bool empty_path_rejected = false;
    try {
        (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--chat-template-file", ""});
    } catch (const std::invalid_argument&) { empty_path_rejected = true; }
    failures += check(empty_path_rejected, "CLI accepted an empty template override path");
    failures += check(ninfer::cli::usage_text("ninfer-cli").find("--chat-template-file") !=
                          std::string::npos,
                      "CLI help omits --chat-template-file");

    const ninfer::cli::Options with_budget =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--thinking-budget", "37"});
    failures += check(with_budget.thinking_budget == 37,
                      "--thinking-budget did not preserve its positive value");
    failures += check(ninfer::cli::usage_text("ninfer-cli").find("--thinking-budget") !=
                          std::string::npos,
                      "CLI help omits --thinking-budget");
    failures += check(rejects([] {
                          (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello",
                                       "--thinking-budget", "0"});
                      }),
                      "zero --thinking-budget was accepted");
    failures += check(rejects([] {
                          (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello",
                                       "--thinking-budget", "8", "--no-thinking"});
                      }),
                      "--thinking-budget was accepted with --no-thinking");
    const ninfer::cli::Options with_effort =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--thinking-budget", "8",
               "--reasoning-effort", "medium"});
    failures += check(with_effort.thinking_budget == 8 && with_effort.reasoning_effort,
                      "thinking budget did not coexist with reasoning effort");
    failures += check(rejects([] {
                          (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello",
                                       "--top-k", "21"});
                      }),
                      "CLI accepted top_k beyond the executable candidate domain");

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
