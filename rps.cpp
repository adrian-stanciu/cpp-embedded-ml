#include "rps.hpp"

#include <algorithm>
#include <array>
#include <iterator>
#include <optional>
#include <string>
#include <utility>

#include <fmt/core.h>

[[nodiscard]] std::string ic::RockPaperScissors::Game::to_string() const noexcept
{
    return fmt::format("AI: {:s} | player: {:s} | {:s}", ai_hand, player_hand,
        result == Result::Win ? "win" : result == Result::Loss ? "loss" : "draw");
}

[[nodiscard]] std::optional<ic::RockPaperScissors::Game> ic::RockPaperScissors::play(std::string player_hand) noexcept
{
    static constexpr std::array hands{"rock", "paper", "scissors"};

    auto it = std::find(hands.begin(), hands.end(), player_hand);
    if (it == hands.end())
        return std::nullopt;
    auto player_hand_idx = std::distance(hands.begin(), it);

    auto ai_hand_idx = distrib(gen);

    Game game;
    game.ai_hand = hands[ai_hand_idx];
    game.player_hand = std::move(player_hand);

    if (player_hand_idx == ai_hand_idx) {
        game.result = Game::Result::Draw;
        ++draws;
    } else if (player_hand_idx == ((ai_hand_idx + 1) % 3)) {
        game.result = Game::Result::Win;
        ++wins;
    } else {
        game.result = Game::Result::Loss;
        ++losses;
    }

    return game;
}

void ic::RockPaperScissors::print_stats() const noexcept
{
    fmt::print("wins: {:d} | draws: {:d} | losses: {:d}\n", wins, draws, losses);
}

