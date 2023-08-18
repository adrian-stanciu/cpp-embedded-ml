#include "camera.hpp"
#include "ml.hpp"
#include "util.hpp"

#include <cstdlib>
#include <exception>

#include <fmt/core.h>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/opencv.hpp>

int main(int argc, char **argv)
{
    static constexpr auto ConfidenceThreshold{0.1};

    cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_SILENT);

    auto options{parse_options(argc, argv)};
    if (!options) {
        fmt::print(stderr, "failed to parse options - run '{:s} -h' for usage\n", argv[0]);
        return EXIT_FAILURE;
    }

    try {
        ImageClassifier image_classifier{options->model_path, options->labels_path, options->num_threads};

        if (options->image_path) {
            auto image{cv::imread(options->image_path)};
            if (image.empty()) {
                fmt::print(stderr, "empty image\n");
                return EXIT_FAILURE;
            }

            if (auto results{image_classifier.run(image, ConfidenceThreshold)}; !results.empty()) {
                write_results(results, image);
                cv::imshow("camera", image);
                cv::waitKey(0);
            } else {
                fmt::print(stderr, "failed to classify image\n");
                return EXIT_FAILURE;
            }
        } else {
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
                    write_results(results, image);
                else
                    fmt::print(stdout, "no results\n");
                fmt::print("\n");

                cv::imshow("camera", image);
                if (cv::waitKey(1) > 0)
                    break;
            }
        }

        return EXIT_SUCCESS;
    } catch (const std::exception& e) {
        fmt::print("error: {:s}\n", e.what());
        return EXIT_FAILURE;
    }
}

