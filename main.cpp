#include "camera.hpp"
#include "cmdline_parser.hpp"
#include "image_classifier.hpp"
#include "rps.hpp"

#include <cmath>
#include <cstdlib>
#include <exception>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <fmt/core.h>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/opencv.hpp>

namespace {
    constexpr auto ProbabilityThreshold{0.1f};

    [[nodiscard]] bool is_result_significant(const ic::ImageClassifier::Result &result)
    {
        constexpr auto SignificantProbabilityThreshold{0.8f};

        return result.probability > SignificantProbabilityThreshold ||
            std::abs(result.probability - SignificantProbabilityThreshold) < std::numeric_limits<double>::epsilon();
    }

    void print_results(const std::vector<ic::ImageClassifier::Result> &results)
    {
        for (const auto &[probability, label] : results)
            fmt::print("{:.2f} | {:s}\n", probability, label);
    }

    void show_on_image(const std::string &text, cv::Mat &image, cv::Point position)
    {
        cv::putText(image, text.data(), position, cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
    }

    void show_result_on_image(const ic::ImageClassifier::Result &result, cv::Mat &image)
    {
        auto text{fmt::format("{:.2f} | {:s}", result.probability, result.label)};
        show_on_image(text, image, cv::Point(image.cols / 20, image.rows / 10));
    }

    [[nodiscard]] std::string handle_rps(std::optional<ic::RockPaperScissors> &rps, const std::string &player_hand)
    {
        if (auto rps_game = rps->play(player_hand); rps_game) {
            auto text{rps_game->to_string()};
            fmt::print("{:s}\n", text);
            return text;
        } else {
            fmt::print(stderr, "failed to classify image as rock, paper or scissors\n");
            return "";
        }
    }

    [[nodiscard]] bool handle_image(const ic::ImageClassifier &image_classifier, std::string_view input_image_path,
        const char *output_image_path, std::optional<ic::RockPaperScissors> &rps)
    {
        auto image{cv::imread(input_image_path.data())};
        if (image.empty()) {
            fmt::print(stderr, "empty image\n");
            return false;
        }

        auto results{image_classifier.run(image, ProbabilityThreshold)};
        if (results.empty()) {
            fmt::print(stderr, "failed to classify image\n");
            return false;
        }

        print_results(results);

        if (is_result_significant(results.front())) {
            show_result_on_image(results.front(), image);

            if (rps) {
                auto rps_outcome{handle_rps(rps, results.front().label)};
                show_on_image(rps_outcome, image, cv::Point(image.cols / 20, image.rows / 5));
            }
        } else {
            fmt::print(stdout, "ambiguous results\n");
        }

        cv::imshow("camera", image);

        cv::waitKey(0);

        if (output_image_path)
            cv::imwrite(output_image_path, image);

        return true;
    }

    [[nodiscard]] bool handle_camera_stream(const ic::ImageClassifier &image_classifier, const char *output_image_path,
        std::optional<ic::RockPaperScissors> &rps)
    {
        static constexpr auto SpaceKey{0x20};

        ic::Camera camera;
        if (!camera.is_open()) {
            fmt::print(stderr, "failed to open camera\n");
            return false;
        }

        auto player_ready{false};
        std::string rps_outcome;

        while (true) {
            auto image{camera.read_image()};
            if (image.empty()) {
                fmt::print(stderr, "empty image\n");
                continue;
            }

            if (auto results{image_classifier.run(image, ProbabilityThreshold)}; !results.empty()) {
                print_results(results);

                if (is_result_significant(results.front())) {
                    show_result_on_image(results.front(), image);

                    if (rps && std::exchange(player_ready, false))
                        rps_outcome = handle_rps(rps, results.front().label);
                } else {
                    fmt::print(stdout, "ambiguous results\n");
                }
            } else {
                fmt::print(stdout, "no results\n");
            }
            fmt::print("\n");

            show_on_image(rps_outcome, image, cv::Point(image.cols / 20, image.rows / 5));

            cv::imshow("camera", image);

            auto key = cv::waitKey(1);
            if (key == SpaceKey) {
                player_ready = true;
            } else if (key > 0) {
                if (output_image_path)
                    cv::imwrite(output_image_path, image);
                break;
            }
        }

        if (rps)
            rps->print_stats();

        return true;
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

    bool successful{false};

    try {
        ic::ImageClassifier image_classifier{options->model_path, options->labels_path, options->num_threads};

        successful = options->input_image_path ?
            handle_image(image_classifier, options->input_image_path, options->output_image_path, rps) :
            handle_camera_stream(image_classifier, options->output_image_path, rps);
    } catch (const std::exception &e) {
        fmt::print("error: {:s}\n", e.what());
    }

    return successful ? EXIT_SUCCESS : EXIT_FAILURE;
}
