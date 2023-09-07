#include "cmdline_parser.hpp"

#include <charconv>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <string_view>

#include <unistd.h>

#include <fmt/core.h>

namespace {
    void print_help(const char *binary)
    {
        fmt::print(
            R"(usage: {:s} [-g] [-i <path-to-input-image>] -l <path-to-labels> -m <path-to-model> [-o <path-to-output-image>] [-t <number-of-threads>]
options:
    -g: optional rock-paper-scissors game enabled (disabled by default); press SPACE key to play a round
    -i: optional path to input image (live camera stream by default)
    -l: mandatory path to labels
    -m: mandatory path to model
    -o: optional path where to save the output image (not saved by default)
    -t: optional number of threads to run the inference (1 by default)
press any (non-SPACE in '-g' mode) key to exit
)",
            binary);
    }
}

[[nodiscard]] std::optional<ic::Options> ic::parse_options(int argc, char **argv)
{
    Options options{};

    // print nothing on errors
    opterr = 0;

    int opt;
    while ((opt = getopt(argc, argv, "ghi:l:m:o:t:")) != -1) {
        switch (opt) {
        case 'g':
            options.play_rps = true;
            break;
        case 'h':
            print_help(argv[0]);
            exit(EXIT_SUCCESS);
        case 'i':
            options.input_image_path = optarg;
            break;
        case 'l':
            options.labels_path = optarg;
            break;
        case 'm':
            options.model_path = optarg;
            break;
        case 'o':
            options.output_image_path = optarg;
            break;
        case 't':
            options.num_threads = [](std::string_view sv) -> int {
                if (int value; std::from_chars(sv.data(), sv.data() + sv.size(), value).ec == std::errc{})
                    return value;
                else
                    return 1;
            }(std::string_view{optarg, strlen(optarg)});
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
