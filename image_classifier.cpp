#include "camera.hpp"
#include "ml.hpp"

#include <cstdlib>
#include <exception>
#include <optional>

#include <unistd.h>

#include <fmt/core.h>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/opencv.hpp>

namespace {
    struct Options {
        const char *labels_path{nullptr};
        const char *image_path{nullptr};
        const char *model_path{nullptr};
    };

    [[nodiscard]] std::optional<Options> parse_options(int argc, char **argv)
    {
        Options options{};

        // print nothing on errors
        opterr = 0;

        int opt;
        while ((opt = getopt(argc, argv, "hi:l:m:")) != -1) {
            switch (opt) {
            case 'h':
                fmt::print("usage: {:s} [-i <path-to-image>] -l <path-to-labels> -m <path-to-model>\n", argv[0]);
                exit(EXIT_SUCCESS);
            case 'i':
                options.image_path = optarg;
                break;
            case 'l':
                options.labels_path = optarg;
                break;
            case 'm':
                options.model_path = optarg;
                break;
            default:
                fmt::print(stderr, "unknown option '{:c}'\n", optopt);
                break;
            }
        }

        if (!options.labels_path) {
            fmt::print(stderr, "labels not provided\n");
            return std::nullopt;
        }
        if (!options.model_path) {
            fmt::print(stderr, "model not provided\n");
            return std::nullopt;
        }

        return options;
    }

    [[nodiscard]] std::optional<cv::Mat> retrieve_image(const char *image_path)
    {
        if (image_path)
            return cv::imread(image_path);

        Camera camera;
        if (!camera.is_open()) {
            fmt::print(stderr, "failed to open camera\n");
            return std::nullopt;
        }

        while (true) {
            auto image{camera.read_image()};
            cv::imshow("camera", image);
            if (cv::waitKey(1) > 0)
                return image;
        }
    }
}

int main(int argc, char **argv)
{
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
        ImageClassifier image_classifier{options->model_path, options->labels_path};

        auto results{image_classifier.run(*image)};
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

