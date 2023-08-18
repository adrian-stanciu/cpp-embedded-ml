#include "camera.hpp"
#include "cmdline_parser.hpp"
#include "image_classifier.hpp"

#include <cstdlib>
#include <exception>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <fmt/core.h>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/opencv.hpp>

namespace {
    constexpr auto ConfidenceThreshold{0.1};

    void show_results(const std::vector<std::pair<double, std::string>>& results, cv::Mat& image)
    {
        for (const auto& [confidence, label] : results)
            fmt::print("{:.2f} | {:s}\n", confidence, label);

        const auto& [confidence, label]{results.front()};
        auto text{fmt::format("{:.2f} | {:s}", confidence, label)};
        cv::putText(image, text.data(), cv::Point(image.rows / 10, image.cols / 10), cv::FONT_HERSHEY_SIMPLEX, 1.0,
            cv::Scalar(0, 0, 255), 2);
    }

    [[nodiscard]] auto handle_image(const ImageClassifier& image_classifier, std::string_view image_path)
    {
        auto image{cv::imread(image_path.data())};
        if (image.empty()) {
            fmt::print(stderr, "empty image\n");
            return EXIT_FAILURE;
        }

        if (auto results{image_classifier.run(image, ConfidenceThreshold)}; !results.empty()) {
            show_results(results, image);

            cv::imshow("camera", image);
            cv::waitKey(0);
            return EXIT_SUCCESS;
        } else {
            fmt::print(stderr, "failed to classify image\n");
            return EXIT_FAILURE;
        }
    }

    [[nodiscard]] auto handle_camera_stream(const ImageClassifier& image_classifier)
    {
        Camera camera;
        if (!camera.is_open()) {
            fmt::print(stderr, "failed to open camera\n");
            return EXIT_FAILURE;
        }

        while (true) {
            auto image{camera.read_image()};
            if (image.empty()) {
                fmt::print(stderr, "empty image\n");
                continue;
            }

            if (auto results{image_classifier.run(image, ConfidenceThreshold)}; !results.empty())
                show_results(results, image);
            else
                fmt::print(stdout, "no results\n");
            fmt::print("\n");

            cv::imshow("camera", image);
            if (cv::waitKey(1) > 0)
                break;
        }

        return EXIT_SUCCESS;
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

    try {
        ImageClassifier image_classifier{options->model_path, options->labels_path, options->num_threads};

        if (options->image_path)
            return handle_image(image_classifier, options->image_path);
        else
            return handle_camera_stream(image_classifier);
    } catch (const std::exception& e) {
        fmt::print("error: {:s}\n", e.what());
        return EXIT_FAILURE;
    }
}

