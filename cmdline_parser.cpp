#include "cmdline_parser.hpp"

#include <charconv>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <string_view>

#include <unistd.h>

#include <fmt/core.h>

[[nodiscard]] std::optional<ic::Options> ic::parse_options(int argc, char **argv)
{
    Options options{};

    // print nothing on errors
    opterr = 0;

    int opt;
    while ((opt = getopt(argc, argv, "hi:l:m:t:")) != -1) {
        switch (opt) {
        case 'h':
            fmt::print("usage: {:s} [-i <path-to-image>] -l <path-to-labels> -m <path-to-model> [-t <number-of-threads>]\n",
                argv[0]);
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
        case 't':
            options.num_threads = [](std::string_view sv) -> std::optional<int> {
                if (int value; std::from_chars(sv.data(), sv.data() + sv.size(), value).ec == std::errc{})
                    return value;
                else
                    return std::nullopt;
            }(std::string_view{optarg, strlen(optarg)}).value_or(1);
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
