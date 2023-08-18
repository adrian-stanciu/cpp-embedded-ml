#pragma once

#include <optional>
#include <string_view>

struct Options {
    const char *labels_path{nullptr};
    const char *image_path{nullptr};
    const char *model_path{nullptr};
    int num_threads{1};
};

[[nodiscard]] std::optional<Options> parse_options(int argc, char **argv);

[[nodiscard]] std::optional<int> to_int(std::string_view sv);

