#include "camera.hpp"
#include "ml.hpp"
#include "util.hpp"

#include <cstdlib>
#include <exception>

#include <fmt/core.h>
#include <opencv2/core/utils/logger.hpp>

int main(int argc, char **argv)
{
    static constexpr auto ConfidenceThreshold{0.1};

    cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_SILENT);

    auto options{parse_options(argc, argv)};
    if (!options) {
        fmt::print(stderr, "failed to parse options - run '{:s} -h' for usage\n", argv[0]);
        return EXIT_FAILURE;
    }

    auto image{retrieve_image(options->image_path)};
    if (!image || image->empty()) {
        fmt::print(stderr, "failed to read image\n");
        return EXIT_FAILURE;
    }

    try {
        ImageClassifier image_classifier{options->model_path, options->labels_path, options->num_threads};

        auto results{image_classifier.run(*image, ConfidenceThreshold)};
        if (results.empty()) {
            fmt::print(stderr, "failed to classify the image\n");
            return EXIT_FAILURE;
        }

        fmt::print("confidence | label\n");
        for (const auto& [confidence, label] : results)
            fmt::print("{:.2f} | {:s}\n", confidence, label);

        return EXIT_SUCCESS;
    } catch (const std::exception& e) {
        fmt::print("error: {:s}\n", e.what());
        return EXIT_FAILURE;
    }
}

