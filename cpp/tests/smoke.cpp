#include "gaiazero/contracts.hpp"
#include "gaiazero/gaia_state.hpp"
#include "gaiazero/graph_encoder.hpp"
#include "gaiazero/inference.hpp"
#include "gaiazero/onnxruntime_backend.hpp"
#include "gaiazero/sha256.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {

void require(bool condition) {
    if (!condition) throw std::runtime_error("smoke test requirement failed");
}

void require_close(float actual, float expected, float tolerance = 1.0e-6F) {
    require(std::abs(actual - expected) <= tolerance);
}

}  // namespace

int main() {
    using namespace gaiazero;

    static_assert(kActionTypeCount == 54);
    require(kRulesVersion == "standard-v22");
    require(action_type_id(ActionType::build_mine) == 0);
    require(action_type_id(ActionType::legacy_action) == 53);
    require(action_type_name(ActionType::research) == "research");
    require(action_type_name(static_cast<ActionType>(99)).empty());
    require(sha256_hex("abc") ==
           "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");

    const auto mine = ActionTuple::create(ActionType::build_mine, {3, 17});
    const auto same_mine = ActionTuple::create(ActionType::build_mine, {3, 17});
    const auto other_mine = ActionTuple::create(ActionType::build_mine, {3, 18});
    require(mine.valid());
    require(mine.arguments[2] == -1);
    require(mine == same_mine);
    require(mine != other_mine);

    bool rejected_action = false;
    try {
        (void)ActionTuple::create(ActionType::legacy_action,
                                  {0, 1, 2, 3, 4, 5, 6, 7, 8});
    } catch (const std::invalid_argument&) {
        rejected_action = true;
    }
    require(rejected_action);

    GraphBatch batch;
    batch.shape = GraphShape{2, 8, 12, 3, 4, 5, 6};
    batch.node_features.resize(2 * 8 * 4);
    batch.edge_index.resize(2 * 12 * 2);
    batch.edge_type.resize(2 * 12);
    batch.edge_mask.resize(2 * 12);
    batch.node_mask.resize(2 * 8);
    batch.global_features.resize(2 * 5);
    batch.player_features.resize(2 * 3 * 6);
    batch.player_mask.resize(2 * 3);
    validate_graph_batch(batch);

    batch.player_mask.pop_back();
    bool rejected = false;
    try {
        validate_graph_batch(batch);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected);

    bool backend_unavailable = false;
    try {
        OnnxRuntimeCpuBackend backend(std::filesystem::path{});
        require(backend.name() == "onnxruntime-cpu");
    } catch (const std::exception&) {
        backend_unavailable = true;
    }
    require(backend_unavailable);

    // Rules/state contract: value copies are safe MCTS branches and every
    // legal state transition changes the state hash deterministically.
    auto state = GaiaState::initial(2, 20260828);
    require(!state.is_terminal());
    require(state.state_hash().size() == 64);
    require(state.setup_hash.size() == 64);
    require(state.setup_hash != GaiaState::initial(2, 20260829).setup_hash);
    require(state_hash_from_canonical_json(state.canonical_json()) == state.state_hash());
    require(state.state_hash() == GaiaState::initial(2, 20260828).state_hash());

    // Graph tensors use the exact ordering and normalization consumed by the
    // Python graph_inputs_from_state function and the ONNX boundary.
    const auto encoded = encode_graph_batch(state);
    require(encoded.shape.batch == 1);
    require(encoded.shape.nodes == 128);
    require(encoded.shape.edges == 512);
    require(encoded.shape.players == 2);
    require(encoded.shape.node_features == 16);
    require(encoded.shape.global_features == 16);
    require(encoded.shape.player_features == 16);
    validate_graph_batch(encoded);
    std::size_t first_active = 0;
    while (first_active < state.active_planets.size() &&
           !state.active_planets[first_active]) ++first_active;
    require(first_active < state.active_planets.size());
    require_close(encoded.node_mask[0], 1.0F);
    require_close(encoded.node_features[0],
                  static_cast<float>(state.planet_q[first_active]) / 16.0F);
    require_close(encoded.node_features[1],
                  static_cast<float>(state.planet_r[first_active]) / 16.0F);
    require_close(encoded.node_features[2],
                  static_cast<float>(state.terrains[first_active]) / 9.0F);
    require_close(encoded.node_features[3],
                  static_cast<float>(state.owners[first_active]) / 2.0F);
    require_close(encoded.global_features[0], 0.0F);
    require_close(encoded.global_features[1], 0.5F);
    require_close(encoded.global_features[2], 1.0F);
    require_close(encoded.global_features[3], 0.0F);
    require_close(encoded.global_features[6],
                  state.player_to_move == 0 ? 1.0F : 0.0F);
    require_close(encoded.global_features[7],
                  state.player_to_move == 1 ? 1.0F : 0.0F);
    require_close(encoded.player_features[0],
                  static_cast<float>(state.players[0].credits) / 30.0F);
    require_close(encoded.player_features[15],
                  state.player_to_move == 0 ? 1.0F : 0.0F);
    require_close(encoded.player_features[16 + 15],
                  state.player_to_move == 1 ? 1.0F : 0.0F);
    require_close(encoded.player_mask[0], 1.0F);
    require_close(encoded.player_mask[1], 1.0F);
    require(encoded.edge_mask[0] == 1.0F);

    auto encoded_mutated = state;
    encoded_mutated.used_power_actions = (1 << 0) | (1 << 6);
    encoded_mutated.used_qic_actions = (1 << 1);
    const auto encoded_actions = encode_graph_batch(encoded_mutated);
    // With two players, power bits begin at observation/global index 10.
    require_close(encoded_actions.global_features[10], 1.0F);
    require_close(encoded_actions.global_features[15], 0.0F);

    bool rejected_encoder_config = false;
    try {
        auto invalid_encoder = GraphEncoderConfig{};
        invalid_encoder.max_nodes = 0;
        (void)encode_graph_batch(state, invalid_encoder);
    } catch (const std::invalid_argument&) {
        rejected_encoder_config = true;
    }
    require(rejected_encoder_config);

    for (int player_count = 2; player_count <= 4; ++player_count) {
        const auto multiplayer = GaiaState::initial(player_count, 20260828);
        const auto multiplayer_graph = encode_graph_batch(multiplayer);
        require(multiplayer_graph.shape.players == player_count);
        require(multiplayer_graph.player_mask.size() ==
                static_cast<std::size_t>(player_count));
        for (int player = 0; player < player_count; ++player) {
            require_close(
                multiplayer_graph.player_features[
                    static_cast<std::size_t>(player) * 16 + 15],
                multiplayer.player_to_move == player ? 1.0F : 0.0F);
        }
    }
    require(state.setup_seed_streams[0].first == "map");
    require(state.setup_seed_streams[0].second == 20260828ULL);
    require(state.setup_seed_streams[1].first == "factions");
    require(state.setup_seed_streams[1].second == 10232423578781582718ULL);
    const auto placement_actions = state.legal_action_tuples();
    require(!placement_actions.empty());
    const auto placed = state.apply(placement_actions.front());
    require(placed.state_hash() != state.state_hash());
    require(state.owners[static_cast<std::size_t>(placement_actions.front().arguments[0])] < 0);

    // Finish setup using the first legal choice at every decision point.
    auto setup_state = state;
    while (!setup_state.is_terminal() &&
           (setup_state.is_starting_placement() || setup_state.is_booster_selection())) {
        const auto actions = setup_state.legal_action_tuples();
        require(!actions.empty());
        setup_state = setup_state.apply(actions.front());
    }
    require(setup_state.round_number == 1);
    const auto round_actions = setup_state.legal_action_tuples();
    require(!round_actions.empty());
    const auto after_pass = setup_state.apply(round_actions.back());
    require(after_pass.state_hash() != setup_state.state_hash());
    require(state_hash_from_canonical_json("{\"a\":1}") ==
           "5fcacb6679bf9140ede62a21078c25157798fd6d2485649c9898de1cb567543c");

    bool checked_xenos = false;
    bool checked_taklons = false;
    bool checked_ivits = false;
    bool checked_lantids = false;
    for (std::int64_t seed = 0; seed < 256 &&
         !(checked_xenos && checked_taklons && checked_ivits && checked_lantids); ++seed) {
        auto candidate = GaiaState::initial(2, seed);
        for (int player = 0; player < candidate.player_count; ++player) {
            const auto& info = candidate.players[static_cast<std::size_t>(player)];
            if (info.faction == 2) checked_xenos = true;
            if (info.faction == 4) {
                require(info.brainstone_bowl == 1);
                require(info.bowl_one == 3);
                checked_taklons = true;
            }
            if (info.faction == 7) checked_ivits = true;
            if (info.faction == 1) {
                for (const int level : info.tracks) require(level == 0);
                checked_lantids = true;
            }
        }
        while (candidate.is_starting_placement() || candidate.is_booster_selection()) {
            const auto actions = candidate.legal_action_tuples();
            require(!actions.empty());
            candidate = candidate.apply(actions.front());
        }
        for (int player = 0; player < candidate.player_count; ++player) {
            const auto faction = candidate.players[static_cast<std::size_t>(player)].faction;
            if (faction == 2) {
                require(candidate.starting_planet_count[static_cast<std::size_t>(player)] == 3);
                // The third Xenos mine uncovers no extra ore income: 1 + 3 - 1 = 3.
                int booster = -1;
                for (int index = 0; index < kBoosterCount; ++index)
                    if (candidate.booster_owner[static_cast<std::size_t>(index)] == player) booster = index;
                const int booster_ore = booster == 2 || booster == 3 || booster == 5 || booster == 6 ? 1 : 0;
                require(candidate.players[static_cast<std::size_t>(player)].ore == 7 + booster_ore);
            }
            if (faction == 7) {
                require(candidate.starting_planet_count[static_cast<std::size_t>(player)] == 1);
                const int planet = candidate.starting_planets[static_cast<std::size_t>(player)][0];
                require(candidate.buildings[static_cast<std::size_t>(planet)] ==
                        static_cast<int>(Building::planetary_institute));
            }
        }
    }
    require(checked_xenos && checked_taklons && checked_ivits && checked_lantids);

    auto complete_game = setup_state;
    int pass_decisions = 0;
    while (!complete_game.is_terminal()) {
        const auto actions = complete_game.legal_action_tuples();
        require(!actions.empty());
        const auto& pass = actions.back();
        require(pass.action_type == ActionType::pass_booster ||
                pass.action_type == ActionType::pass_final);
        complete_game = complete_game.apply(pass);
        require(++pass_decisions <= kMaxRounds * complete_game.player_count);
    }
    require(complete_game.round_number == kMaxRounds + 1);
    const auto final_scores = complete_game.final_scores();
    for (int player = 0; player < complete_game.player_count; ++player)
        require(final_scores[static_cast<std::size_t>(player)] > 0.0);

    std::cout << "gaiazero_cpp_smoke ok\n";
    return 0;
}
