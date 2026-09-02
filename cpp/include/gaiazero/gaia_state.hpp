#pragma once

#include "gaiazero/contracts.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace gaiazero {

inline constexpr int kMaxPlayers = 4;
inline constexpr int kMaxPlanets = 71;
inline constexpr int kMaxBoardSpaces = 190;
inline constexpr int kMaxSectors = 10;
inline constexpr int kBoosterCount = 10;
inline constexpr int kTrackCount = 6;
inline constexpr int kMaxRounds = 6;
inline constexpr int kMaxPlacementSteps = kMaxPlayers * 3;

enum class Terrain : std::uint8_t {
    terra = 0, desert, swamp, volcanic, oxide, titanium, ice, transdim, gaia, lost
};

enum class Building : std::uint8_t {
    empty = 0, mine, trading_station, research_lab, planetary_institute, academy
};

struct PlayerState {
    std::int32_t faction{-1};
    std::int32_t credits{15};
    std::int32_t ore{4};
    std::int32_t knowledge{3};
    std::int32_t qic{1};
    std::int32_t vp{10};
    std::int32_t bowl_one{2};
    std::int32_t bowl_two{4};
    std::int32_t bowl_three{0};
    std::int32_t brainstone_bowl{0};
    std::int32_t gaia_power{0};
    std::int32_t gaiaformers{0};
    std::int32_t gaiaformers_in_gaia{0};
    std::array<std::int32_t, kTrackCount> tracks{};
    std::uint32_t tech_tiles{0};
    std::uint32_t advanced_tech_tiles{0};
    std::uint32_t covered_tech_tiles{0};
    std::int32_t knowledge_academies{0};
    std::int32_t qic_academies{0};
    bool used_qic_academy_action{false};
    bool used_standard_tech_action{false};
    std::int32_t used_advanced_tech_actions{0};
    bool used_booster_action{false};
    bool used_ambas_swap_action{false};
    bool used_firaks_downgrade_action{false};
    bool used_bescods_research_action{false};
    bool used_ivits_space_station_action{false};
    std::int32_t federation_tokens{0};
    std::int32_t federation_keys{0};
    std::int32_t board_federations{0};
    std::array<std::int32_t, 6> federation_tile_counts{};
    std::int32_t gleens_federation_tokens{0};
    std::int32_t satellites{0};
    std::uint32_t colonized_types{0};
    bool passed{false};
};

struct GaiaState {
    std::int32_t player_count{2};
    std::int64_t setup_seed{0};
    std::string setup_seed_stream_version{"setup-seed-stream-v1"};
    std::vector<std::pair<std::string, std::uint64_t>> setup_seed_streams;
    std::string setup_hash;
    std::int32_t round_number{0};
    std::int32_t player_to_move{0};
    std::int32_t first_player{0};
    std::int32_t next_first_player{-1};
    std::array<PlayerState, kMaxPlayers> players{};
    std::array<std::int32_t, kMaxPlanets> planet_q{};
    std::array<std::int32_t, kMaxPlanets> planet_r{};
    std::array<std::int32_t, kMaxPlanets> planet_source_q{};
    std::array<std::int32_t, kMaxPlanets> planet_source_r{};
    std::array<std::int32_t, kMaxPlanets> planet_source_ids{};
    std::array<std::array<std::int32_t, 5>, kMaxPlanets> planet_source_catalog{};
    std::array<std::int32_t, kMaxPlanets> planet_sectors{};
    std::array<std::int32_t, kMaxSectors> sector_tiles{};
    std::array<std::int32_t, kMaxSectors> sector_rotations{};
    std::array<std::array<std::int32_t, 2>, kMaxSectors> sector_centers{};
    std::string map_mode{"bga-random"};
    std::array<std::int32_t, kMaxPlanets> owners{};
    std::array<std::int32_t, kMaxPlanets> buildings{};
    std::array<std::int32_t, kMaxPlanets> terrains{};
    std::array<std::int32_t, kMaxPlanets> gaiaformer_owner{};
    std::array<std::int32_t, kMaxPlanets> coexisting_mine_owner{};
    std::array<bool, kMaxPlanets> coexisting_mine_federated{};
    std::array<bool, kMaxPlanets> active_planets{};
    std::array<bool, kMaxPlanets> federated{};
    std::array<std::int32_t, kMaxBoardSpaces> satellite_owners{};
    std::array<std::int32_t, kMaxBoardSpaces> space_station_owner{};
    std::array<bool, kMaxBoardSpaces> space_station_federated{};
    std::array<std::int32_t, kBoosterCount> booster_owner{};
    std::array<std::int32_t, kBoosterCount> booster_selection_order{};
    std::array<std::int32_t, 6> round_scoring_tiles{};
    std::array<std::int32_t, 2> final_scoring_tiles{};
    std::array<std::int32_t, 9> standard_tech_tiles{};
    std::array<std::int32_t, 6> advanced_tech_tiles{};
    std::array<std::int32_t, 6> federation_tile_supply{};
    std::array<std::int32_t, kMaxPlayers> starting_planet_count{};
    std::array<std::array<std::int32_t, 3>, kMaxPlayers> starting_planets{};
    std::array<std::int32_t, kMaxPlacementSteps> placement_order{};
    std::int32_t placement_order_length{0};
    std::int32_t sector_count{0};
    std::int32_t planet_source_catalog_length{0};
    std::int32_t placement_step{0};
    std::int32_t booster_selection_step{0};
    std::int32_t used_power_actions{0};
    std::int32_t used_qic_actions{0};
    std::int32_t pending_tech_player{-1};
    std::int32_t pending_advanced_tech{-1};
    std::int32_t pending_research_player{-1};
    std::int32_t pending_research_track{-1};
    bool pending_research_optional{false};
    std::int32_t pending_lost_planet_player{-1};
    std::int32_t pending_power_terraform_player{-1};
    std::int32_t pending_power_terraform_steps{0};
    std::int32_t pending_booster_terraform_player{-1};
    std::int32_t pending_booster_range_player{-1};
    bool brainstone_selected{false};
    std::int32_t pending_gaia_conversion_player{-1};
    std::int32_t pending_gaia_conversion_power{0};
    std::int32_t pending_itars_gaia_player{-1};
    std::int32_t pending_passive_charge_player{-1};
    std::int32_t pending_passive_charge_acting{-1};
    std::int32_t pending_passive_charge_planet{-1};
    std::int32_t pending_passive_charge_amount{0};
    std::int32_t pending_taklons_charge_player{-1};
    std::int32_t pending_taklons_charge_acting{-1};
    std::int32_t pending_taklons_charge_amount{0};
    std::array<std::array<std::int32_t, 2>, kMaxPlayers> pending_passive_charge_queue{};
    std::int32_t pending_passive_charge_queue_length{0};
    std::int32_t terraforming_federation_tile{-1};

    static GaiaState initial(std::int32_t players = 2, std::int64_t seed = 0);
    [[nodiscard]] bool is_terminal() const noexcept;
    [[nodiscard]] bool is_starting_placement() const noexcept;
    [[nodiscard]] bool is_booster_selection() const noexcept;
    [[nodiscard]] std::string state_hash() const;
    [[nodiscard]] std::string canonical_json() const;
    [[nodiscard]] std::vector<ActionTuple> legal_action_tuples() const;
    [[nodiscard]] GaiaState apply(const ActionTuple& action) const;
    [[nodiscard]] std::array<double, kMaxPlayers> final_scores() const;
};

}  // namespace gaiazero
