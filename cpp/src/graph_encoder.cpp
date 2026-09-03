#include "gaiazero/graph_encoder.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace gaiazero {

bool GraphEncoderConfig::valid() const noexcept {
    return max_nodes > 0 && max_edges > 0 && node_features > 0 &&
           global_features > 0 && player_features > 0 && relation_types > 0;
}

namespace {

constexpr std::int64_t kPowerActionCount = 7;
constexpr std::int64_t kQicActionCount = 3;

std::vector<float> observation_prefix(const GaiaState& state) {
    std::vector<float> values;
    values.reserve(static_cast<std::size_t>(
        6 + state.player_count * 2 + kPowerActionCount + kQicActionCount));
    values.push_back(static_cast<float>(state.round_number) /
                     static_cast<float>(kMaxRounds));
    values.push_back(static_cast<float>(state.player_count) / 4.0F);
    values.push_back(state.is_starting_placement() ? 1.0F : 0.0F);
    values.push_back(state.is_booster_selection() ? 1.0F : 0.0F);
    values.push_back(static_cast<float>(state.booster_selection_step) /
                     static_cast<float>(std::max(1, state.player_count)));
    values.push_back(state.brainstone_selected ? 1.0F : 0.0F);
    for (int player = 0; player < state.player_count; ++player) {
        values.push_back(state.player_to_move == player ? 1.0F : 0.0F);
    }
    for (int player = 0; player < state.player_count; ++player) {
        values.push_back(state.first_player == player ? 1.0F : 0.0F);
    }
    for (std::int64_t action = 0; action < kPowerActionCount; ++action) {
        values.push_back((state.used_power_actions & (1 << action)) != 0 ? 1.0F : 0.0F);
    }
    for (std::int64_t action = 0; action < kQicActionCount; ++action) {
        values.push_back((state.used_qic_actions & (1 << action)) != 0 ? 1.0F : 0.0F);
    }
    return values;
}

void copy_prefix(float* destination, std::int64_t destination_size,
                 const std::vector<float>& values) {
    const auto count = std::min(
        static_cast<std::size_t>(destination_size), values.size());
    std::copy_n(values.begin(), count, destination);
}

}  // namespace

GraphBatch encode_graph_batch(const GaiaState& state,
                              const GraphEncoderConfig& config) {
    if (!config.valid()) {
        throw std::invalid_argument("GraphEncoderConfig dimensions must all be positive");
    }
    if (state.player_count < 2 || state.player_count > kMaxPlayers) {
        throw std::invalid_argument("GaiaState player_count must be in [2, 4]");
    }

    const auto globals = observation_prefix(state);
    if (static_cast<std::size_t>(config.global_features) > globals.size()) {
        throw std::invalid_argument(
            "graph encoder v1 only exposes the Python observation prefix; "
            "global_features exceeds the supported prefix size");
    }

    GraphBatch batch;
    batch.shape = GraphShape{
        1,
        config.max_nodes,
        config.max_edges,
        state.player_count,
        config.node_features,
        config.global_features,
        config.player_features,
    };

    const auto nodes = static_cast<std::size_t>(config.max_nodes);
    const auto edges = static_cast<std::size_t>(config.max_edges);
    const auto players = static_cast<std::size_t>(state.player_count);
    const auto node_features = static_cast<std::size_t>(config.node_features);
    const auto global_features = static_cast<std::size_t>(config.global_features);
    const auto player_features = static_cast<std::size_t>(config.player_features);
    batch.node_features.assign(nodes * node_features, 0.0F);
    batch.edge_index.assign(edges * 2, 0);
    batch.edge_type.assign(edges, 0);
    batch.edge_mask.assign(edges, 0.0F);
    batch.node_mask.assign(nodes, 0.0F);
    batch.global_features.assign(global_features, 0.0F);
    batch.player_features.assign(players * player_features, 0.0F);
    batch.player_mask.assign(players, 1.0F);

    std::vector<std::size_t> active;
    active.reserve(kMaxPlanets);
    for (std::size_t planet = 0; planet < state.active_planets.size(); ++planet) {
        if (state.active_planets[planet]) active.push_back(planet);
    }
    if (active.size() > nodes) active.resize(nodes);

    const float player_denominator =
        static_cast<float>(std::max(1, state.player_count));
    for (std::size_t node = 0; node < active.size(); ++node) {
        const auto planet = active[node];
        batch.node_mask[node] = 1.0F;
        const std::vector<float> values{
            static_cast<float>(state.planet_q[planet]) / 16.0F,
            static_cast<float>(state.planet_r[planet]) / 16.0F,
            static_cast<float>(state.terrains[planet]) / 9.0F,
            static_cast<float>(state.owners[planet]) / player_denominator,
            static_cast<float>(state.buildings[planet]) / 5.0F,
            state.federated[planet] ? 1.0F : 0.0F,
            static_cast<float>(state.gaiaformer_owner[planet]) / player_denominator,
            static_cast<float>(state.coexisting_mine_owner[planet]) / player_denominator,
        };
        copy_prefix(batch.node_features.data() + node * node_features,
                    config.node_features, values);
    }

    std::size_t edge_count = 0;
    for (std::size_t left = 0; left < active.size(); ++left) {
        const auto planet_left = active[left];
        for (std::size_t right = 0; right < active.size(); ++right) {
            if (left == right) continue;
            const auto planet_right = active[right];
            int distance = std::abs(state.planet_q[planet_left] -
                                    state.planet_q[planet_right]);
            distance += std::abs(state.planet_r[planet_left] -
                                 state.planet_r[planet_right]);
            distance += std::abs(
                (state.planet_q[planet_left] + state.planet_r[planet_left]) -
                (state.planet_q[planet_right] + state.planet_r[planet_right]));
            if (distance / 2 > 1 || edge_count >= edges) continue;
            batch.edge_index[edge_count * 2] = static_cast<std::int64_t>(left);
            batch.edge_index[edge_count * 2 + 1] = static_cast<std::int64_t>(right);
            const auto terrain = state.terrains[planet_right];
            batch.edge_type[edge_count] =
                ((terrain % config.relation_types) + config.relation_types) %
                config.relation_types;
            batch.edge_mask[edge_count] = 1.0F;
            ++edge_count;
        }
    }

    std::copy_n(globals.begin(), global_features, batch.global_features.begin());

    for (std::size_t player = 0; player < players; ++player) {
        const auto& info = state.players[player];
        const std::vector<float> values{
            static_cast<float>(info.credits) / 30.0F,
            static_cast<float>(info.ore) / 15.0F,
            static_cast<float>(info.knowledge) / 15.0F,
            static_cast<float>(info.qic) / 10.0F,
            static_cast<float>(info.vp) / 150.0F,
            static_cast<float>(info.bowl_one) / 15.0F,
            static_cast<float>(info.bowl_two) / 15.0F,
            static_cast<float>(info.bowl_three) / 15.0F,
            static_cast<float>(info.gaia_power) / 15.0F,
            static_cast<float>(info.gaiaformers) / 3.0F,
            static_cast<float>(info.federation_tokens) / 6.0F,
            static_cast<float>(info.tracks[0]) / 5.0F,
            static_cast<float>(info.tracks[1]) / 5.0F,
            static_cast<float>(info.tracks[2]) / 5.0F,
            info.passed ? 1.0F : 0.0F,
            static_cast<std::size_t>(state.player_to_move) == player ? 1.0F : 0.0F,
        };
        copy_prefix(batch.player_features.data() + player * player_features,
                    config.player_features, values);
    }

    validate_graph_batch(batch);
    return batch;
}

}  // namespace gaiazero
