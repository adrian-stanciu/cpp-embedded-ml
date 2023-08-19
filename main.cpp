#include "camera.hpp"
#include "cmdline_parser.hpp"
#include "image_classifier.hpp"
#include "rps.hpp"

#include <cstdlib>
#include <exception>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <fmt/core.h>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/opencv.hpp>

namespace {
    constexpr auto ConfidenceThreshold{0.1};

    void show_on_image(const std::string& text, cv::Mat& image, cv::Point position)
    {
        cv::putText(image, text.data(), position, cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
    }

    void report_results(const std::vector<ic::ImageClassifier::Result>& results, cv::Mat& image)
    {
        for (const auto& [confidence, label] : results)
            fmt::print("{:.2f} | {:s}\n", confidence, label);

        const auto& [confidence, label]{results.front()};
        auto text{fmt::format("{:.2f} | {:s}", confidence, label)};
        show_on_image(text, image, cv::Point(image.cols / 20, image.rows / 10));
    }

    void handle_rps(std::optional<ic::RockPaperScissors>& rps, const std::string& player_hand, cv::Mat& image)
    {
        if (!rps)
            return;

        if (auto rps_game = rps->play(player_hand); rps_game) {
            auto text{rps_game->to_string()};
            fmt::print("{:s}\n", text);
            show_on_image(text, image, cv::Point(image.cols / 20, image.rows / 5));
        } else {
            fmt::print(stderr, "failed to classify image as rock, paper or scissors\n");
        }
    }

    [[nodiscard]] auto handle_image(const ic::ImageClassifier& image_classifier, std::string_view image_path,
        std::optional<ic::RockPaperScissors>& rps)
    {
        auto image{cv::imread(image_path.data())};
        if (image.empty()) {
            fmt::print(stderr, "empty image\n");
            return EXIT_FAILURE;
        }

        if (auto results{image_classifier.run(image, ConfidenceThreshold)}; !results.empty()) {
            report_results(results, image);

            handle_rps(rps, results.front().label, image);

            cv::imshow("camera", image);
            cv::waitKey(0);
            return EXIT_SUCCESS;
        } else {
            fmt::print(stderr, "failed to classify image\n");
            return EXIT_FAILURE;
        }
    }

    [[nodiscard]] auto handle_camera_stream(const ic::ImageClassifier& image_classifier,
        std::optional<ic::RockPaperScissors>& rps)
    {
        ic::Camera camera;
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

            if (auto results{image_classifier.run(image, ConfidenceThreshold)}; !results.empty()) {
                report_results(results, image);

                handle_rps(rps, results.front().label, image);
            } else {
                fmt::print(stdout, "no results\n");
            }
            fmt::print("\n");

            cv::imshow("camera", image);
            if (cv::waitKey(1) > 0)
                break;
        }

        if (rps)
            rps->print_stats();

        return EXIT_SUCCESS;
    }
}

int main(int argc, char **argv)
{
    cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_SILENT);

    auto options{ic::parse_options(argc, argv)};
    if (!options) {
        fmt::print(stderr, "failed to parse options - run '{:s} -h' for usage\n", argv[0]);
        return EXIT_FAILURE;
    }

    std::optional<ic::RockPaperScissors> rps;
    if (options->play_rps)
        rps = ic::RockPaperScissors{};

    try {
        ic::ImageClassifier image_classifier{options->model_path, options->labels_path, options->num_threads};

        if (options->image_path)
            return handle_image(image_classifier, options->image_path, rps);
        else
            return handle_camera_stream(image_classifier, rps);
    } catch (const std::exception& e) {
        fmt::print("error: {:s}\n", e.what());
        return EXIT_FAILURE;
    }
}

