#pragma once

#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>

struct Options {
    const char *labels_path{nullptr};
    const char *image_path{nullptr};
    const char *model_path{nullptr};
    int num_threads{1};
};

[[nodiscard]] std::optional<Options> parse_options(int argc, char **argv);

[[nodiscard]] std::optional<int> to_int(std::string_view sv);

void write_results(const std::vector<std::pair<double, std::string>>& results, cv::Mat& image);

