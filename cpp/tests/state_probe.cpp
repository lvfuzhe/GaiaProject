#include "gaiazero/contracts.hpp"
#include "gaiazero/gaia_state.hpp"

#include <iostream>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace {

void print_encoded_actions(const gaiazero::GaiaState& state) {
    const auto actions = state.legal_action_tuples();
    for (std::size_t action_index = 0; action_index < actions.size(); ++action_index) {
        if (action_index) std::cout << '|';
        const auto& action = actions[action_index];
        std::cout << gaiazero::action_type_name(action.action_type) << '[';
        for (std::size_t index = 0; index < action.argument_count; ++index) {
            if (index) std::cout << '.';
            std::cout << action.arguments[index];
        }
        std::cout << ']';
    }
}

void print_scenario_step(int step, const gaiazero::GaiaState& state) {
    std::cout << "scenario_step=" << step << ",hash=" << state.state_hash()
              << ",actions=";
    print_encoded_actions(state);
    std::cout << '\n';
}

int first_active_planet(const gaiazero::GaiaState& state) {
    for (int planet = 0; planet < gaiazero::kMaxPlanets; ++planet) {
        if (state.active_planets[static_cast<std::size_t>(planet)]) return planet;
    }
    throw std::logic_error("scenario requires an active planet");
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 3 || argc > 4) {
            throw std::invalid_argument("usage: gaiazero_state_probe <players> <seed> [trace]");
        }
        const int players = std::stoi(argv[1]);
        const auto seed = std::stoll(argv[2]);
        const auto state = gaiazero::GaiaState::initial(players, seed);

        if (argc == 4 && std::string(argv[3]) == "scenario-itars-gaia") {
            auto current = state;
            const int planet = first_active_planet(current);
            gaiazero::PlayerState itars;
            itars.faction = 13;
            itars.ore = 5;
            itars.bowl_one = 4;
            itars.bowl_two = 4;
            itars.gaia_power = 8;
            itars.colonized_types = 1U << static_cast<unsigned>(gaiazero::Terrain::ice);
            current.players[0] = itars;
            current.owners[static_cast<std::size_t>(planet)] = 0;
            current.buildings[static_cast<std::size_t>(planet)] =
                static_cast<int>(gaiazero::Building::planetary_institute);
            current.starting_planet_count[0] = 1;
            current.starting_planets[0][0] = planet;
            current.round_number = 1;
            current.placement_step = current.placement_order_length;
            current.booster_selection_step = current.player_count;
            current.player_to_move = 0;
            current.pending_itars_gaia_player = 0;

            print_scenario_step(0, current);
            current = current.apply(gaiazero::ActionTuple::create(
                gaiazero::ActionType::itars_gaia_technology, {}));
            print_scenario_step(1, current);
            current = current.apply(gaiazero::ActionTuple::create(
                gaiazero::ActionType::tech_take, {8}));
            print_scenario_step(2, current);
            current = current.apply(gaiazero::ActionTuple::create(
                gaiazero::ActionType::skip_tech_research, {}));
            print_scenario_step(3, current);
            current = current.apply(gaiazero::ActionTuple::create(
                gaiazero::ActionType::itars_gaia_finish, {}));
            print_scenario_step(4, current);
            return 0;
        }

        if (argc == 4 && std::string(argv[3]) == "scenario-gleens-qic") {
            auto current = state;
            const int planet = first_active_planet(current);
            gaiazero::PlayerState gleens;
            gleens.faction = 3;
            gleens.qic = 3;
            gleens.tracks[1] = 1;
            gleens.federation_tokens = 1;
            gleens.federation_keys = 1;
            gleens.gleens_federation_tokens = 1;
            gleens.colonized_types = 1U << static_cast<unsigned>(gaiazero::Terrain::desert);
            current.players[0] = gleens;
            current.owners[static_cast<std::size_t>(planet)] = 0;
            current.buildings[static_cast<std::size_t>(planet)] =
                static_cast<int>(gaiazero::Building::planetary_institute);
            current.starting_planet_count[0] = 1;
            current.starting_planets[0][0] = planet;
            current.round_number = 1;
            current.placement_step = current.placement_order_length;
            current.booster_selection_step = current.player_count;
            current.player_to_move = 0;

            print_scenario_step(0, current);
            current = current.apply(gaiazero::ActionTuple::create(
                gaiazero::ActionType::qic_federation, {6}));
            print_scenario_step(1, current);
            return 0;
        }

        if (argc == 4 && std::string(argv[3]).starts_with("trace")) {
            auto current = state;
            // Keep the default trace long enough to cover a complete short
            // opening sequence.  This is intentionally a diagnostic mode;
            // the state is still advanced one legal first action at a time.
            int limit = 200;
            const std::string mode = argv[3];
            if (const auto equals = mode.find('='); equals != std::string::npos) {
                limit = std::stoi(mode.substr(equals + 1));
            }
            const bool random_trace = mode.starts_with("trace-random");
            const auto choice_for = [&](int step, std::size_t count) {
                std::uint64_t value = static_cast<std::uint64_t>(seed);
                value += static_cast<std::uint64_t>(step) * UINT64_C(1442695040888963407);
                value ^= value >> 30;
                value *= UINT64_C(0xbf58476d1ce4e5b9);
                value ^= value >> 27;
                value *= UINT64_C(0x94d049bb133111eb);
                value ^= value >> 31;
                return count == 0 ? std::size_t{0} : static_cast<std::size_t>(value % count);
            };
            for (int step = 0; step < limit && !current.is_terminal(); ++step) {
                const auto actions = current.legal_action_tuples();
                if (mode == "trace") {
                    std::cout << "step=" << step << ",hash=" << current.state_hash()
                              << ",action_count=" << actions.size();
                    std::cout << ",all_actions=";
                    print_encoded_actions(current);
                    if (!actions.empty()) {
                        std::cout << ",first=" << gaiazero::action_type_name(actions.front().action_type) << '(';
                        for (std::size_t index = 0; index < actions.front().argument_count; ++index) {
                            if (index) std::cout << ',';
                            std::cout << actions.front().arguments[index];
                        }
                        std::cout << ')';
                    }
                    std::cout << '\n';
                } else if (random_trace) {
                    const auto chosen = choice_for(step, actions.size());
                    std::cout << "step=" << step << ",hash=" << current.state_hash()
                              << ",action_count=" << actions.size() << ",chosen=" << chosen
                              << ",all_actions=";
                    print_encoded_actions(current);
                    std::cout << '\n';
                }
                if (actions.empty()) break;
                const auto chosen = random_trace ? choice_for(step, actions.size()) : 0;
                current = current.apply(actions[chosen]);
            }
            if (mode == "trace" || mode.starts_with("trace-random")) {
                std::cout << "state_hash=" << current.state_hash() << '\n';
                std::cout << "canonical_json=" << current.canonical_json() << '\n';
                std::cout << "federation_plan=" << current.debug_federation_plan() << '\n';
            }
            return 0;
        }
        std::cout << "state_hash=" << state.state_hash() << '\n';
        std::cout << "setup_hash=" << state.setup_hash << '\n';
        std::cout << "canonical_json=" << state.canonical_json() << '\n';
        std::cout << "legal_actions=";
        const auto actions = state.legal_action_tuples();
        for (std::size_t index = 0; index < actions.size(); ++index) {
            if (index) std::cout << ';';
            const auto& action = actions[index];
            std::cout << gaiazero::action_type_name(action.action_type) << '(';
            for (std::size_t argument = 0; argument < action.argument_count; ++argument) {
                if (argument) std::cout << ',';
                std::cout << action.arguments[argument];
            }
            std::cout << ')';
        }
        std::cout << '\n';
        std::cout << "federation_plan=" << state.debug_federation_plan() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
