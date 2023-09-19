#pragma once

#include <optional>
#include <random>
#include <string>

namespace ic {
    struct RockPaperScissors {
        struct Game {
            enum struct Result { Win, Draw, Loss };

            [[nodiscard]] std::string to_string() const noexcept;

            std::string ai_hand;
            std::string player_hand;
            Result result;
        };

        RockPaperScissors() : gen{std::random_device{}()}, distrib{0, 2} {}

        [[nodiscard]] std::optional<Game> play(std::string player_hand) noexcept;

        void print_stats() const noexcept;

    private:
        std::mt19937 gen;
        std::uniform_int_distribution<> distrib;

        size_t wins{0};
        size_t draws{0};
        size_t losses{0};
    };
}
