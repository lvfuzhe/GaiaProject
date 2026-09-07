#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace gaiazero {

inline constexpr int kPrintedPlanetSlots = 70;

struct GaiaSetupData {
    std::int32_t player_count{2};
    std::int64_t seed{0};
    std::string seed_stream_version{"setup-seed-stream-v1"};
    std::vector<std::pair<std::string, std::uint64_t>> seed_streams;
    std::string setup_hash;
    std::int32_t first_player{0};
    std::array<std::int32_t, 4> factions{};
    std::vector<std::int32_t> placement_order;
    std::array<bool, kPrintedPlanetSlots> active_planets{};
    std::array<std::int32_t, kPrintedPlanetSlots> planet_q{};
    std::array<std::int32_t, kPrintedPlanetSlots> planet_r{};
    std::array<std::int32_t, kPrintedPlanetSlots> planet_source_q{};
    std::array<std::int32_t, kPrintedPlanetSlots> planet_source_r{};
    std::array<std::int32_t, kPrintedPlanetSlots> planet_source_ids{};
    std::vector<std::array<std::int32_t, 5>> planet_source_catalog;
    std::array<std::int32_t, kPrintedPlanetSlots> terrains{};
    std::array<std::int32_t, kPrintedPlanetSlots> planet_sectors{};
    std::vector<std::int32_t> sector_tiles;
    std::vector<std::int32_t> sector_rotations;
    std::vector<std::array<std::int32_t, 2>> sector_centers;
    std::string map_mode{"bga-random"};
    std::array<std::int32_t, 10> booster_owner{};
    std::array<std::int32_t, 6> round_scoring_tiles{};
    std::array<std::int32_t, 2> final_scoring_tiles{};
    std::array<std::int32_t, 9> standard_tech_tiles{};
    std::array<std::int32_t, 6> advanced_tech_tiles{};
    std::int32_t terraforming_federation_tile{0};
};

// Exact C++ implementation of gaia_setup.py's bga-random path. map_size may
// be empty (2p reduced, otherwise normal), "reduced", or "normal".
[[nodiscard]] GaiaSetupData generate_gaia_setup(
    std::int32_t player_count,
    std::int64_t seed,
    std::string map_size = {});

}  // namespace gaiazero
