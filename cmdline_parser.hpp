#pragma once

#include <optional>

namespace ic {
    struct Options {
        const char *labels_path{nullptr};
        const char *model_path{nullptr};
        const char *input_image_path{nullptr};
        const char *output_image_path{nullptr};
        int num_threads{1};
        bool play_rps{false};
    };

    [[nodiscard]] std::optional<ic::Options> parse_options(int argc, char **argv);
}

