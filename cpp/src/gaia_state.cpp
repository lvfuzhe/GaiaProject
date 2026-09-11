#include "gaiazero/gaia_state.hpp"

#include "gaiazero/gaia_setup.hpp"
#include "gaiazero/sha256.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdlib>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <queue>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <vector>
#include <cmath>

namespace gaiazero {
namespace {

void append_json_string(std::ostringstream& out, std::string_view value) {
    out << '"';
    for (const char ch : value) {
        switch (ch) {
        case '"': out << "\\\""; break;
        case '\\': out << "\\\\"; break;
        case '\n': out << "\\n"; break;
        case '\r': out << "\\r"; break;
        case '\t': out << "\\t"; break;
        default: out << ch; break;
        }
    }
    out << '"';
}

template <typename T, std::size_t N>
void append_array(std::ostringstream& out, const std::array<T, N>& values) {
    out << '[';
    for (std::size_t i = 0; i < N; ++i) {
        if (i != 0) out << ',';
        if constexpr (std::is_same_v<T, bool>) out << (values[i] ? "true" : "false");
        else out << values[i];
    }
    out << ']';
}

template <typename T, std::size_t Outer, std::size_t Inner>
void append_nested_array_fixed(std::ostringstream& out,
                               const std::array<std::array<T, Inner>, Outer>& values) {
    out << '[';
    for (std::size_t i = 0; i < Outer; ++i) {
        if (i != 0) out << ',';
        append_array(out, values[i]);
    }
    out << ']';
}

void append_player(std::ostringstream& out, const PlayerState& p) {
    // Keep this list lexicographically sorted; Python canonical_json uses sort_keys=True.
    out << "{\"advanced_tech_tiles\":" << p.advanced_tech_tiles
        << ",\"board_federations\":" << p.board_federations
        << ",\"bowl_one\":" << p.bowl_one
        << ",\"bowl_three\":" << p.bowl_three
        << ",\"bowl_two\":" << p.bowl_two
        << ",\"brainstone_bowl\":" << p.brainstone_bowl
        << ",\"colonized_types\":" << p.colonized_types
        << ",\"covered_tech_tiles\":" << p.covered_tech_tiles
        << ",\"credits\":" << p.credits
        << ",\"faction\":" << p.faction
        << ",\"federation_keys\":" << p.federation_keys
        << ",\"federation_tile_counts\":";
    append_array(out, p.federation_tile_counts);
    out << ",\"federation_tokens\":" << p.federation_tokens
        << ",\"gaia_power\":" << p.gaia_power
        << ",\"gaiaformers\":" << p.gaiaformers
        << ",\"gaiaformers_in_gaia\":" << p.gaiaformers_in_gaia
        << ",\"gleens_federation_tokens\":" << p.gleens_federation_tokens
        << ",\"knowledge\":" << p.knowledge
        << ",\"knowledge_academies\":" << p.knowledge_academies
        << ",\"ore\":" << p.ore
        << ",\"passed\":" << (p.passed ? "true" : "false")
        << ",\"qic\":" << p.qic
        << ",\"qic_academies\":" << p.qic_academies
        << ",\"satellites\":" << p.satellites
        << ",\"tech_tiles\":" << p.tech_tiles
        << ",\"tracks\":";
    append_array(out, p.tracks);
    out << ",\"used_advanced_tech_actions\":" << p.used_advanced_tech_actions
        << ",\"used_ambas_swap_action\":" << (p.used_ambas_swap_action ? "true" : "false")
        << ",\"used_bescods_research_action\":" << (p.used_bescods_research_action ? "true" : "false")
        << ",\"used_booster_action\":" << (p.used_booster_action ? "true" : "false")
        << ",\"used_firaks_downgrade_action\":" << (p.used_firaks_downgrade_action ? "true" : "false")
        << ",\"used_ivits_space_station_action\":" << (p.used_ivits_space_station_action ? "true" : "false")
        << ",\"used_qic_academy_action\":" << (p.used_qic_academy_action ? "true" : "false")
        << ",\"used_standard_tech_action\":" << (p.used_standard_tech_action ? "true" : "false")
        << ",\"vp\":" << p.vp << '}';
}

void append_key(std::ostringstream& out, std::string_view key) {
    append_json_string(out, key);
    out << ':';
}

int vp_offset_limit(int players) {
    if (players == 2) return 30;
    if (players == 3 || players == 4) return 50;
    throw std::invalid_argument("player_count must be two to four");
}

void validate_vp_offsets(int players, const std::array<std::int32_t, kMaxPlayers>& offsets,
                         std::string_view name) {
    const int limit = vp_offset_limit(players);
    int sum = 0;
    for (int player = 0; player < players; ++player) {
        const int value = offsets[static_cast<std::size_t>(player)];
        if (std::abs(value) > limit)
            throw std::invalid_argument(std::string(name) + " value outside player-count bound");
        sum += value;
    }
    for (int player = players; player < kMaxPlayers; ++player) {
        if (offsets[static_cast<std::size_t>(player)] != 0)
            throw std::invalid_argument(std::string(name) + " contains values beyond player_count");
    }
    if (sum != 0) throw std::invalid_argument(std::string(name) + " must sum to zero");
}

struct FactionDefaults {
    Terrain home;
    std::array<int, 3> power;
    int starting_structures;
    bool starts_with_pi;
    int credits;
    int ore;
    int knowledge;
    int qic;
    int start_track;
    int income_ore;
    int income_credits;
    int income_knowledge;
    int income_qic;
    int income_power;
    bool brainstone;
};

constexpr std::array<FactionDefaults, 14> kFactions{{
    {Terrain::terra, {4, 4, 0}, 2, false, 15, 4, 3, 1, 3, 0, 0, 0, 0, 0, false}, // Terrans
    {Terrain::terra, {4, 0, 0}, 2, false, 13, 4, 3, 1, -1, 0, 0, 0, 0, 0, false}, // Lantids
    {Terrain::desert, {2, 4, 0}, 3, false, 15, 4, 3, 1, 2, 0, 0, 0, 0, 0, false}, // Xenos
    {Terrain::desert, {2, 4, 0}, 2, false, 15, 4, 3, 0, 1, 0, 0, 0, 0, 0, false}, // Gleens
    {Terrain::swamp, {2, 4, 0}, 2, false, 15, 4, 3, 1, -1, 0, 0, 0, 0, 0, true}, // Taklons
    {Terrain::swamp, {2, 4, 0}, 2, false, 15, 4, 3, 1, 1, 1, 0, 0, 0, 0, false}, // Ambas
    {Terrain::oxide, {2, 4, 0}, 2, false, 15, 4, 3, 1, 4, 0, 3, 0, 0, 0, false}, // Hadsch Hallas
    {Terrain::oxide, {2, 4, 0}, 1, true, 15, 4, 3, 1, -1, 0, 0, 0, 1, 0, false}, // Ivits
    {Terrain::volcanic, {2, 4, 0}, 2, false, 15, 4, 3, 1, 0, 0, 0, 0, 0, 0, false}, // Geodens
    {Terrain::volcanic, {2, 2, 0}, 2, false, 15, 4, 3, 0, 3, 0, 0, 0, 0, 0, false}, // Bal T'aks
    {Terrain::titanium, {2, 4, 0}, 2, false, 15, 3, 2, 1, -1, 0, 0, 1, 0, 0, false}, // Firaks
    {Terrain::titanium, {2, 4, 0}, 2, false, 15, 4, 1, 1, -1, 0, 0, -1, 0, 0, false}, // Bescods
    {Terrain::ice, {2, 4, 0}, 2, false, 15, 4, 2, 1, 5, 0, 0, 0, 0, 0, false}, // Nevlas
    {Terrain::ice, {4, 4, 0}, 2, false, 15, 5, 3, 1, -1, 0, 0, 0, 0, 1, false}, // Itars
}};

int current_player(const GaiaState& state) {
    if (state.player_to_move < 0 || state.player_to_move >= state.player_count) {
        throw std::logic_error("player_to_move is outside player_count");
    }
    return state.player_to_move;
}

bool is_home_planet(const GaiaState& state, int player, int planet) {
    return planet >= 0 && planet < kMaxPlanets && state.active_planets[static_cast<std::size_t>(planet)] &&
           state.owners[static_cast<std::size_t>(planet)] < 0 &&
           state.terrains[static_cast<std::size_t>(planet)] ==
               static_cast<int>(kFactions[static_cast<std::size_t>(state.players[static_cast<std::size_t>(player)].faction)].home);
}

int building_count(const GaiaState& state, int player, Building building) {
    int count = 0;
    for (int planet = 0; planet < kMaxPlanets; ++planet) {
        const auto index = static_cast<std::size_t>(planet);
        count += state.owners[index] == player && state.buildings[index] == static_cast<int>(building);
        if (building == Building::mine) count += state.coexisting_mine_owner[index] == player;
    }
    return count;
}

int mine_supply_count(const GaiaState& state, int player) {
    int count = 0;
    for (int planet = 0; planet < kMaxPlanets; ++planet) {
        const auto index = static_cast<std::size_t>(planet);
        count += state.owners[index] == player &&
                 state.buildings[index] == static_cast<int>(Building::mine) &&
                 state.terrains[index] != static_cast<int>(Terrain::lost);
        count += state.coexisting_mine_owner[index] == player;
    }
    return count;
}

int player_booster(const GaiaState& state, int player) {
    for (int booster = 0; booster < kBoosterCount; ++booster)
        if (state.booster_owner[static_cast<std::size_t>(booster)] == player) return booster;
    return -1;
}

bool has_pi(const GaiaState& state, int player) {
    return building_count(state, player, Building::planetary_institute) > 0;
}

int hex_distance(int aq, int ar, int bq, int br) {
    const int dq = aq - bq;
    const int dr = ar - br;
    return (std::abs(dq) + std::abs(dr) + std::abs(dq + dr)) / 2;
}

std::vector<std::array<int, 2>> board_spaces(const GaiaState& state) {
    std::vector<std::array<int, 2>> result;
    result.reserve(static_cast<std::size_t>(state.sector_count * 19));
    for (int sector = 0; sector < state.sector_count; ++sector) {
        const auto center = state.sector_centers[static_cast<std::size_t>(sector)];
        for (int q = -2; q <= 2; ++q) {
            for (int r = -2; r <= 2; ++r) {
                if (std::max({std::abs(q), std::abs(r), std::abs(q + r)}) <= 2) {
                    result.push_back({center[0] + q, center[1] + r});
                }
            }
        }
    }
    std::sort(result.begin(), result.end());
    return result;
}

bool player_has_structure(const GaiaState& state, int player, int planet) {
    const auto index = static_cast<std::size_t>(planet);
    return state.owners[index] == player || state.coexisting_mine_owner[index] == player;
}

bool can_lantids_coexist(const GaiaState& state, int player, int planet) {
    const auto index = static_cast<std::size_t>(planet);
    return state.players[static_cast<std::size_t>(player)].faction == 1 && state.active_planets[index] &&
           state.owners[index] >= 0 && state.owners[index] != player &&
           state.buildings[index] != static_cast<int>(Building::empty) &&
           state.terrains[index] != static_cast<int>(Terrain::lost) && state.coexisting_mine_owner[index] < 0;
}

int terrain_steps(Terrain home, Terrain destination) {
    const int left = static_cast<int>(home);
    const int right = static_cast<int>(destination);
    if (left >= 7 || right >= 7) return 0;
    const int direct = std::abs(left - right);
    return std::min(direct, 7 - direct);
}

int coordinate_range_qic_cost(const GaiaState& state, int player, int q, int r,
                              int range_bonus = 0) {
    int distance = kMaxPlanets;
    for (int source = 0; source < kMaxPlanets; ++source) {
        const auto index = static_cast<std::size_t>(source);
        if (player_has_structure(state, player, source)) {
            distance = std::min(distance, hex_distance(state.planet_q[index], state.planet_r[index],
                                                       q, r));
        }
    }
    const auto spaces = board_spaces(state);
    for (std::size_t space = 0; space < spaces.size(); ++space) {
        if (state.space_station_owner[space] == player) {
            distance = std::min(distance, hex_distance(spaces[space][0], spaces[space][1], q, r));
        }
    }
    if (distance == kMaxPlanets) return kMaxPlanets;
    constexpr std::array<int, 6> ranges{{1, 1, 2, 2, 3, 4}};
    const int reach = ranges[static_cast<std::size_t>(std::clamp(state.players[static_cast<std::size_t>(player)].tracks[1], 0, 5))] + range_bonus;
    return (std::max(0, distance - reach) + 1) / 2;
}

int range_qic_cost(const GaiaState& state, int player, int destination,
                   int range_bonus = 0) {
    const auto index = static_cast<std::size_t>(destination);
    return coordinate_range_qic_cost(state, player, state.planet_q[index],
                                     state.planet_r[index], range_bonus);
}

struct ResourceCost { int credits; int ore; int qic; };

ResourceCost mine_cost(const GaiaState& state, int player, int planet,
                       int free_steps = 0, int range_bonus = 0) {
    const auto index = static_cast<std::size_t>(planet);
    const auto& p = state.players[static_cast<std::size_t>(player)];
    const bool coexisting = can_lantids_coexist(state, player, planet);
    const int range_qic = state.gaiaformer_owner[index] == player
                              ? 0
                              : range_qic_cost(state, player, planet, range_bonus);
    if (coexisting) return {2, 1, range_qic};
    const auto terrain = static_cast<Terrain>(state.terrains[index]);
    if (terrain == Terrain::gaia) {
        const int gaia_qic = state.gaiaformer_owner[index] == player ? 0 : 1;
        return p.faction == 3 && gaia_qic != 0 ? ResourceCost{2, 2, range_qic}
                                               : ResourceCost{2, 1, gaia_qic + range_qic};
    }
    constexpr std::array<int, 6> ore_per_step{{3, 3, 2, 1, 1, 1}};
    const int steps = std::max(0, terrain_steps(
        kFactions[static_cast<std::size_t>(p.faction)].home, terrain) - free_steps);
    return {2, 1 + steps * ore_per_step[static_cast<std::size_t>(std::clamp(p.tracks[0], 0, 5))], range_qic};
}

bool can_build_mine(const GaiaState& state, int player, int planet,
                    int free_steps = 0, int range_bonus = 0) {
    if (planet < 0 || planet >= kMaxPlanets) return false;
    const auto index = static_cast<std::size_t>(planet);
    const bool coexisting = can_lantids_coexist(state, player, planet);
    if (!state.active_planets[index] || (state.owners[index] >= 0 && !coexisting) ||
        state.terrains[index] == static_cast<int>(Terrain::transdim) ||
        mine_supply_count(state, player) >= 8 ||
        (state.gaiaformer_owner[index] >= 0 && state.gaiaformer_owner[index] != player)) return false;
    const auto cost = mine_cost(state, player, planet, free_steps, range_bonus);
    const auto& p = state.players[static_cast<std::size_t>(player)];
    return p.credits >= cost.credits && p.ore >= cost.ore && p.qic >= cost.qic &&
           (state.gaiaformer_owner[index] == player || cost.qic <= p.qic);
}

bool has_active_standard_tech(const GaiaState& state, const PlayerState& player, int tile) {
    const auto mask = std::uint32_t{1} << static_cast<unsigned>(tile);
    return (player.tech_tiles & mask) != 0 && (player.covered_tech_tiles & mask) == 0;
}

bool has_research_choice(const GaiaState& state, int player);

int gaia_cost(const PlayerState& player) {
    constexpr std::array<int, 6> costs{{99, 6, 6, 4, 3, 3}};
    return costs[static_cast<std::size_t>(std::clamp(player.tracks[3], 0, 5))];
}

int cycle_power(const PlayerState& player) {
    return player.bowl_one + player.bowl_two + player.bowl_three;
}

bool can_start_gaia(const GaiaState& state, int player, int planet, int range_bonus = 0) {
    if (planet < 0 || planet >= kMaxPlanets) return false;
    const auto index = static_cast<std::size_t>(planet);
    const auto& p = state.players[static_cast<std::size_t>(player)];
    return state.active_planets[index] &&
           state.terrains[index] == static_cast<int>(Terrain::transdim) &&
           state.owners[index] == -1 && state.gaiaformer_owner[index] == -1 &&
           p.gaiaformers > 0 && cycle_power(p) >= gaia_cost(p) &&
           p.qic >= range_qic_cost(state, player, planet, range_bonus);
}

bool is_coordinate_reachable(const GaiaState& state, int player, int q, int r,
                             int range_bonus = 0) {
    constexpr std::array<int, 6> ranges{{1, 1, 2, 2, 3, 4}};
    const auto& p = state.players[static_cast<std::size_t>(player)];
    const int reach = ranges[static_cast<std::size_t>(std::clamp(p.tracks[1], 0, 5))] + range_bonus;
    for (int planet = 0; planet < kMaxPlanets; ++planet) {
        if (player_has_structure(state, player, planet) &&
            hex_distance(state.planet_q[static_cast<std::size_t>(planet)],
                         state.planet_r[static_cast<std::size_t>(planet)], q, r) <= reach)
            return true;
    }
    const auto spaces = board_spaces(state);
    for (std::size_t space = 0; space < spaces.size(); ++space) {
        if (state.space_station_owner[space] == player &&
            hex_distance(spaces[space][0], spaces[space][1], q, r) <= reach)
            return true;
    }
    return false;
}

int structure_power(const GaiaState& state, int player, Building building, int planet = -1) {
    int power = std::array<int, 6>{{0, 1, 2, 2, 3, 3}}[static_cast<std::size_t>(building)];
    const auto& p = state.players[static_cast<std::size_t>(player)];
    if ((building == Building::planetary_institute || building == Building::academy) &&
        has_active_standard_tech(state, p, 4)) power = 4;
    if (planet >= 0 && p.faction == 11 && has_pi(state, player) &&
        state.terrains[static_cast<std::size_t>(planet)] == static_cast<int>(Terrain::titanium)) ++power;
    return power;
}

int location_power(const GaiaState& state, int player, int location) {
    if (location >= 2 * kMaxPlanets) return 1;
    const int planet = location < kMaxPlanets ? location : location - kMaxPlanets;
    const auto building = location < kMaxPlanets
        ? static_cast<Building>(state.buildings[static_cast<std::size_t>(planet)])
        : Building::mine;
    return structure_power(state, player, building, planet);
}

bool has_nearby_opponent(const GaiaState& state, int player, int planet) {
    for (int other = 0; other < kMaxPlanets; ++other) {
        const auto oi = static_cast<std::size_t>(other);
        if (state.owners[oi] < 0 || state.owners[oi] == player) {
            if (state.coexisting_mine_owner[oi] < 0 || state.coexisting_mine_owner[oi] == player) continue;
        }
        if (hex_distance(state.planet_q[static_cast<std::size_t>(planet)],
                         state.planet_r[static_cast<std::size_t>(planet)],
                         state.planet_q[oi], state.planet_r[oi]) <= 2) return true;
    }
    return false;
}

bool can_advance_research(const GaiaState& state, int player, int track) {
    if (track < 0 || track >= kTrackCount) return false;
    const auto& p = state.players[static_cast<std::size_t>(player)];
    const int level = p.tracks[static_cast<std::size_t>(track)];
    if (level >= 5 || (level == 4 && p.federation_keys <= 0)) return false;
    if (p.faction == 9 && track == 1 && !has_pi(state, player)) return false;
    if (level == 4) {
        for (int opponent = 0; opponent < state.player_count; ++opponent)
            if (opponent != player && state.players[static_cast<std::size_t>(opponent)].tracks[static_cast<std::size_t>(track)] == 5) return false;
    }
    return true;
}

bool has_research_choice(const GaiaState& state, int player) {
    for (int track = 0; track < kTrackCount; ++track)
        if (can_advance_research(state, player, track)) return true;
    return false;
}

bool has_tech_choice(const GaiaState& state, int player) {
    const auto& p = state.players[static_cast<std::size_t>(player)];
    for (const int tile : state.standard_tech_tiles)
        if (tile >= 0 && (p.tech_tiles & (std::uint32_t{1} << static_cast<unsigned>(tile))) == 0) return true;
    bool cover = false;
    for (const int tile : state.standard_tech_tiles)
        if (tile >= 0 && (p.tech_tiles & (std::uint32_t{1} << static_cast<unsigned>(tile))) != 0 &&
            (p.covered_tech_tiles & (std::uint32_t{1} << static_cast<unsigned>(tile))) == 0) cover = true;
    if (!cover || p.federation_keys <= 0) return false;
    for (int track = 0; track < kTrackCount; ++track) {
        const int tile = state.advanced_tech_tiles[static_cast<std::size_t>(track)];
        bool taken = false;
        for (int opponent = 0; opponent < state.player_count; ++opponent)
            taken = taken || (state.players[static_cast<std::size_t>(opponent)].advanced_tech_tiles & (std::uint32_t{1} << static_cast<unsigned>(tile))) != 0;
        if (p.tracks[static_cast<std::size_t>(track)] >= 4 && !taken) return true;
    }
    return false;
}

int power_action_cost(const GaiaState& state, int player, int action) {
    constexpr std::array<int, 7> costs{{7, 5, 4, 4, 4, 3, 3}};
    int cost = costs[static_cast<std::size_t>(action)];
    if (state.players[static_cast<std::size_t>(player)].faction == 12 && has_pi(state, player)) cost = (cost + 1) / 2;
    return cost;
}

int power_terraform_steps(int action) {
    return action == 1 ? 2 : action == 5 ? 1 : 0;
}

int ordinary_power(const PlayerState& p) {
    return p.bowl_three - (p.brainstone_bowl == 3 ? 1 : 0);
}

bool can_spend_power(const PlayerState& p, int amount, bool use_brainstone = false) {
    if (!use_brainstone) return p.bowl_three >= amount;
    return amount >= 3 && p.brainstone_bowl == 3 && p.bowl_three - 1 + 3 >= amount;
}

bool brainstone_action_available(const GaiaState& state, int player) {
    const auto& p = state.players[static_cast<std::size_t>(player)];
    if (p.faction != 4 || p.brainstone_bowl != 3) return false;
    if (ordinary_power(p) >= 1 && p.credits < 30) return true;
    if (ordinary_power(p) >= 3 && p.ore < 15) return true;
    if (ordinary_power(p) >= 4 && (p.knowledge < 15 || p.qic >= 0)) return true;
    for (int action = 0; action < 7; ++action) {
        if ((state.used_power_actions & (1 << action)) != 0) continue;
        const int cost = power_action_cost(state, player, action);
        if (can_spend_power(p, cost, true) &&
            (power_terraform_steps(action) == 0 || std::any_of(state.active_planets.begin(), state.active_planets.end(), [&](bool active) { return active; }))) return true;
    }
    return false;
}

int charge_power(PlayerState& p, int amount) {
    int charged = 0;
    for (int i = 0; i < amount; ++i) {
        if (p.bowl_one > 0) {
            const bool brainstone = p.brainstone_bowl == 1 && p.bowl_one == 1;
            --p.bowl_one; ++p.bowl_two;
            if (brainstone) p.brainstone_bowl = 2;
        } else if (p.bowl_two > 0) {
            const bool brainstone = p.brainstone_bowl == 2 && p.bowl_two == 1;
            --p.bowl_two; ++p.bowl_three;
            if (brainstone) p.brainstone_bowl = 3;
        } else break;
        ++charged;
    }
    return charged;
}

void spend_power(PlayerState& p, int amount, bool use_brainstone = false) {
    if (!can_spend_power(p, amount, use_brainstone))
        throw std::invalid_argument("insufficient charged power");
    if (use_brainstone) {
        const int ordinary = std::max(0, amount - 3);
        const int physical = ordinary + 1;
        p.bowl_one += physical;
        p.bowl_three -= physical;
        p.brainstone_bowl = 1;
        return;
    }
    const bool spends_brainstone = p.brainstone_bowl == 3 && p.bowl_three - 1 < amount;
    p.bowl_one += amount;
    p.bowl_three -= amount;
    if (spends_brainstone) p.brainstone_bowl = 1;
}

void discard_power(PlayerState& p, int amount) {
    if (cycle_power(p) < amount) throw std::invalid_argument("insufficient power tokens");
    std::array<int*, 3> bowls{{&p.bowl_one, &p.bowl_two, &p.bowl_three}};
    for (int bowl = 0; bowl < 3 && amount; ++bowl) {
        const int ordinary = *bowls[static_cast<std::size_t>(bowl)] - (p.brainstone_bowl == bowl + 1 ? 1 : 0);
        const int take = std::min(ordinary, amount);
        *bowls[static_cast<std::size_t>(bowl)] -= take;
        amount -= take;
    }
    if (amount && p.brainstone_bowl >= 1 && p.brainstone_bowl <= 3) {
        --*bowls[static_cast<std::size_t>(p.brainstone_bowl - 1)];
        p.brainstone_bowl = 0;
        --amount;
    }
    if (amount) throw std::logic_error("power discard accounting failed");
}

void move_power_to_gaia(PlayerState& p, int amount) {
    const int old_brainstone = p.brainstone_bowl;
    discard_power(p, amount);
    if (old_brainstone && p.brainstone_bowl == 0) p.brainstone_bowl = 4;
    p.gaia_power += amount;
}

void gain_qic(PlayerState& p, int amount) {
    if (p.faction == 3 && p.qic_academies == 0) p.ore = std::min(15, p.ore + amount);
    else p.qic += amount;
}

void gain_federation_reward(PlayerState& p, int tile) {
    constexpr std::array<int, 6> vp{{6, 7, 8, 8, 7, 12}};
    p.vp += vp[static_cast<std::size_t>(tile)];
    if (tile == 0) p.knowledge = std::min(15, p.knowledge + 2);
    else if (tile == 1) p.ore = std::min(15, p.ore + 2);
    else if (tile == 2) gain_qic(p, 1);
    else if (tile == 3) p.bowl_one += 2;
    else if (tile == 4) p.credits = std::min(30, p.credits + 6);
}

void gain_gleens_federation_reward(PlayerState& p) {
    // Gleens' federation tile is a special QIC action reward and has no VP
    // component.  Keep the same resource caps as the Python rules engine.
    p.credits = std::min(30, p.credits + 2);
    p.ore = std::min(15, p.ore + 1);
    p.knowledge = std::min(15, p.knowledge + 1);
}

void score(GaiaState& state, int player, int kind, int amount = 1) {
    if (state.round_number < 1 || state.round_number > kMaxRounds) return;
    // 0 terraform, 1 research, 2 mine, 3 federation, 4/5 trading,
    // 6/7 Gaia mine, 8/9 PI or academy.
    const int tile = state.round_scoring_tiles[static_cast<std::size_t>(state.round_number - 1)];
    constexpr std::array<int, 10> kinds{{0, 1, 2, 3, 4, 4, 5, 5, 6, 6}};
    constexpr std::array<int, 10> points{{2, 2, 2, 5, 3, 4, 3, 4, 5, 5}};
    if (tile >= 0 && tile < 10 && kinds[static_cast<std::size_t>(tile)] == kind)
        state.players[static_cast<std::size_t>(player)].vp += points[static_cast<std::size_t>(tile)] * amount;
}

int player_structure_power_at(const GaiaState& state, int player, int planet) {
    const auto index = static_cast<std::size_t>(planet);
    int power = 0;
    if (state.owners[index] == player)
        power = structure_power(state, player, static_cast<Building>(state.buildings[index]), planet);
    if (state.coexisting_mine_owner[index] == player)
        power = std::max(power, structure_power(state, player, Building::mine, planet));
    return power;
}

int passive_charge_power(const GaiaState& state, int player, int source) {
    int power = 0;
    for (int planet = 0; planet < kMaxPlanets; ++planet) {
        if (!player_has_structure(state, player, planet)) continue;
        const auto si = static_cast<std::size_t>(source);
        const auto pi = static_cast<std::size_t>(planet);
        if (hex_distance(state.planet_q[si], state.planet_r[si],
                         state.planet_q[pi], state.planet_r[pi]) <= 2)
            power = std::max(power, player_structure_power_at(state, player, planet));
    }
    return power;
}

void trigger_passive_charge(GaiaState& state, int acting, int planet) {
    std::vector<std::array<int, 2>> offers;
    for (int offset = 1; offset < state.player_count; ++offset) {
        const int opponent = (acting + offset) % state.player_count;
        const auto& p = state.players[static_cast<std::size_t>(opponent)];
        const int amount = std::min(passive_charge_power(state, opponent, planet), p.vp + 1);
        if (amount <= 0) continue;
        auto copy = p;
        if (charge_power(copy, amount) > 0) offers.push_back({opponent, amount});
    }
    if (offers.empty()) return;
    state.player_to_move = offers[0][0];
    state.pending_passive_charge_player = offers[0][0];
    state.pending_passive_charge_acting = acting;
    state.pending_passive_charge_planet = planet;
    state.pending_passive_charge_amount = offers[0][1];
    state.pending_passive_charge_queue_length = static_cast<int>(offers.size()) - 1;
    for (std::size_t index = 1; index < offers.size(); ++index)
        state.pending_passive_charge_queue[index - 1] = offers[index];
}

bool touches_existing_federation(const GaiaState& state, int player, int q, int r) {
    for (int planet = 0; planet < kMaxPlanets; ++planet) {
        const auto index = static_cast<std::size_t>(planet);
        if (((state.owners[index] == player && state.federated[index]) ||
             (state.coexisting_mine_owner[index] == player && state.coexisting_mine_federated[index])) &&
            hex_distance(q, r, state.planet_q[index], state.planet_r[index]) <= 1) return true;
    }
    const auto spaces = board_spaces(state);
    for (std::size_t space = 0; space < spaces.size(); ++space) {
        if (state.space_station_owner[space] == player && state.space_station_federated[space] &&
            hex_distance(q, r, spaces[space][0], spaces[space][1]) <= 1) return true;
        if ((state.satellite_owners[space] & (1 << player)) &&
            hex_distance(q, r, spaces[space][0], spaces[space][1]) <= 1) return true;
    }
    return false;
}

void mark_adjacent_federated(GaiaState& state, int player, int location) {
    int q = 0;
    int r = 0;
    if (location < 2 * kMaxPlanets) {
        const int planet = location < kMaxPlanets ? location : location - kMaxPlanets;
        q = state.planet_q[static_cast<std::size_t>(planet)];
        r = state.planet_r[static_cast<std::size_t>(planet)];
    } else {
        const auto spaces = board_spaces(state);
        const int space = location - 2 * kMaxPlanets;
        q = spaces[static_cast<std::size_t>(space)][0];
        r = spaces[static_cast<std::size_t>(space)][1];
    }
    if (!touches_existing_federation(state, player, q, r)) return;
    if (location < kMaxPlanets) state.federated[static_cast<std::size_t>(location)] = true;
    else if (location < 2 * kMaxPlanets) state.coexisting_mine_federated[static_cast<std::size_t>(location - kMaxPlanets)] = true;
    else state.space_station_federated[static_cast<std::size_t>(location - 2 * kMaxPlanets)] = true;
}

using LocationList = std::vector<int>;
using LocationClusters = std::vector<LocationList>;

std::array<int, 2> location_coordinate(const GaiaState& state, int location) {
    if (location < 2 * kMaxPlanets) {
        const int planet = location < kMaxPlanets ? location : location - kMaxPlanets;
        return {state.planet_q[static_cast<std::size_t>(planet)],
                state.planet_r[static_cast<std::size_t>(planet)]};
    }
    return board_spaces(state)[static_cast<std::size_t>(location - 2 * kMaxPlanets)];
}

int location_distance(const GaiaState& state, int first, int second) {
    const auto a = location_coordinate(state, first);
    const auto b = location_coordinate(state, second);
    return hex_distance(a[0], a[1], b[0], b[1]);
}

LocationList structure_locations(const GaiaState& state, int player, bool federated) {
    LocationList result;
    for (int planet = 0; planet < kMaxPlanets; ++planet) {
        const auto index = static_cast<std::size_t>(planet);
        if (state.owners[index] == player && state.federated[index] == federated) result.push_back(planet);
    }
    for (int planet = 0; planet < kMaxPlanets; ++planet) {
        const auto index = static_cast<std::size_t>(planet);
        if (state.coexisting_mine_owner[index] == player && state.coexisting_mine_federated[index] == federated)
            result.push_back(kMaxPlanets + planet);
    }
    for (int space = 0; space < kMaxBoardSpaces; ++space) {
        const auto index = static_cast<std::size_t>(space);
        if (state.space_station_owner[index] == player && state.space_station_federated[index] == federated)
            result.push_back(2 * kMaxPlanets + space);
    }
    return result;
}

LocationClusters location_clusters(const GaiaState& state, LocationList locations) {
    std::set<int> remaining(locations.begin(), locations.end());
    LocationClusters clusters;
    while (!remaining.empty()) {
        std::set<int> component{*remaining.begin()};
        LocationList frontier{*remaining.begin()};
        remaining.erase(remaining.begin());
        while (!frontier.empty()) {
            const int source = frontier.back();
            frontier.pop_back();
            LocationList adjacent;
            for (const int candidate : remaining)
                if (location_distance(state, source, candidate) <= 1) adjacent.push_back(candidate);
            for (const int candidate : adjacent) {
                remaining.erase(candidate); component.insert(candidate); frontier.push_back(candidate);
            }
        }
        clusters.emplace_back(component.begin(), component.end());
    }
    return clusters;
}

int federation_threshold(const GaiaState& state, int player) {
    const auto& info = state.players[static_cast<std::size_t>(player)];
    if (info.faction == 7) return 7 * (info.board_federations + 1);
    if (info.faction == 2 && has_pi(state, player)) return 6;
    return 7;
}

int federation_distance_estimate(const GaiaState& state, const LocationClusters& clusters,
                                 const LocationList& existing) {
    LocationClusters groups;
    if (!existing.empty()) groups.push_back(existing);
    groups.insert(groups.end(), clusters.begin(), clusters.end());
    if (groups.size() < 2) return 0;
    std::set<int> connected{0};
    std::set<int> remaining;
    for (int index = 1; index < static_cast<int>(groups.size()); ++index) remaining.insert(index);
    int estimate = 0;
    while (!remaining.empty()) {
        int best_distance = std::numeric_limits<int>::max();
        int best_target = -1;
        for (const int source_group : connected) for (const int target_group : remaining) {
            int distance = std::numeric_limits<int>::max();
            for (const int source : groups[static_cast<std::size_t>(source_group)])
                for (const int target : groups[static_cast<std::size_t>(target_group)])
                    distance = std::min(distance, location_distance(state, source, target));
            if (std::tie(distance, target_group) < std::tie(best_distance, best_target)) {
                best_distance = distance; best_target = target_group;
            }
        }
        estimate += std::max(0, best_distance - 1);
        connected.insert(best_target); remaining.erase(best_target);
    }
    return estimate;
}

std::vector<int> minimum_satellite_path(const GaiaState& state, int player,
                                        const LocationList& selected,
                                        const LocationList& existing,
                                        bool& valid) {
    valid = true;
    if (selected.empty()) {
        if (existing.empty()) valid = false;
        return {};
    }
    const auto spaces = board_spaces(state);
    std::map<std::array<int, 2>, int> coordinate_to_space;
    for (int space = 0; space < static_cast<int>(spaces.size()); ++space)
        coordinate_to_space[spaces[static_cast<std::size_t>(space)]] = space;
    const auto candidate_locations = structure_locations(state, player, false);
    std::set<std::array<int, 2>> free_coordinates;
    for (const int location : candidate_locations) free_coordinates.insert(location_coordinate(state, location));
    std::set<std::array<int, 2>> root_coordinates;
    for (const int location : existing) root_coordinates.insert(location_coordinate(state, location));
    if (!existing.empty()) for (int space = 0; space < static_cast<int>(spaces.size()); ++space)
        if (state.satellite_owners[static_cast<std::size_t>(space)] & (1 << player)) root_coordinates.insert(spaces[static_cast<std::size_t>(space)]);

    constexpr std::array<std::array<int, 2>, 6> directions{{{{1,0}},{{0,1}},{{-1,1}},{{-1,0}},{{0,-1}},{{1,-1}}}};
    std::set<std::array<int, 2>> forbidden;
    if (existing.empty()) {
        std::set<std::array<int, 2>> old;
        for (const int location : structure_locations(state, player, true)) old.insert(location_coordinate(state, location));
        for (int space = 0; space < static_cast<int>(spaces.size()); ++space)
            if (state.satellite_owners[static_cast<std::size_t>(space)] & (1 << player)) old.insert(spaces[static_cast<std::size_t>(space)]);
        for (const auto coordinate : old) {
            forbidden.insert(coordinate);
            for (const auto delta : directions) forbidden.insert({coordinate[0] + delta[0], coordinate[1] + delta[1]});
        }
    }
    std::set<std::array<int, 2>> planet_coordinates;
    for (int planet = 0; planet < kMaxPlanets; ++planet)
        if (state.active_planets[static_cast<std::size_t>(planet)])
            planet_coordinates.insert({state.planet_q[static_cast<std::size_t>(planet)], state.planet_r[static_cast<std::size_t>(planet)]});
    std::vector<int> allowed_spaces;
    std::set<int> allowed;
    for (int space = 0; space < static_cast<int>(spaces.size()); ++space) {
        const auto coordinate = spaces[static_cast<std::size_t>(space)];
        if (!forbidden.contains(coordinate) &&
            (!planet_coordinates.contains(coordinate) || free_coordinates.contains(coordinate) || root_coordinates.contains(coordinate))) {
            allowed_spaces.push_back(space); allowed.insert(space);
        }
    }
    std::vector<std::vector<int>> neighbors(spaces.size());
    std::vector<int> node_cost(spaces.size());
    for (const int space : allowed_spaces) {
        const auto coordinate = spaces[static_cast<std::size_t>(space)];
        node_cost[static_cast<std::size_t>(space)] = !free_coordinates.contains(coordinate) && !root_coordinates.contains(coordinate);
        for (const auto delta : directions) {
            const auto it = coordinate_to_space.find({coordinate[0] + delta[0], coordinate[1] + delta[1]});
            if (it != coordinate_to_space.end() && allowed.contains(it->second)) neighbors[static_cast<std::size_t>(space)].push_back(it->second);
        }
    }
    const auto selected_clusters = location_clusters(state, selected);
    std::vector<std::vector<int>> terminals;
    if (!existing.empty()) {
        std::vector<int> roots;
        for (const auto& coordinate : root_coordinates) {
            const auto it = coordinate_to_space.find(coordinate);
            if (it != coordinate_to_space.end() && allowed.contains(it->second)) roots.push_back(it->second);
        }
        if (roots.empty()) { valid = false; return {}; }
        terminals.push_back(std::move(roots));
    }
    for (const auto& cluster : selected_clusters) {
        std::vector<int> group;
        for (const int location : cluster) {
            const auto it = coordinate_to_space.find(location_coordinate(state, location));
            if (it != coordinate_to_space.end() && allowed.contains(it->second)) group.push_back(it->second);
        }
        if (group.empty()) { valid = false; return {}; }
        terminals.push_back(std::move(group));
    }
    const int terminal_count = static_cast<int>(terminals.size());
    const int full_mask = (1 << terminal_count) - 1;
    const int infinity = kMaxBoardSpaces * 10;
    std::vector<std::vector<int>> distances(static_cast<std::size_t>(full_mask + 1),
                                            std::vector<int>(spaces.size(), infinity));
    struct Parent { int type{-1}; int a{0}; int b{0}; };
    std::map<std::pair<int, int>, Parent> parents;
    auto relax = [&](int mask) {
        using Item = std::pair<int, int>;
        // Python's reference implementation intentionally starts relax() with
        // a list comprehension and then calls heapq.heappop().  The initial
        // list is not heapified, so reproducing heapq's sift operations (rather
        // than constructing a C++ priority_queue) is part of the parity
        // contract for tied Steiner paths.
        std::vector<Item> heap;
        for (const int space : allowed_spaces)
            if (distances[static_cast<std::size_t>(mask)][static_cast<std::size_t>(space)] < infinity)
                heap.push_back({distances[static_cast<std::size_t>(mask)][static_cast<std::size_t>(space)], space});
        auto sift_down = [](std::vector<Item>& values, std::size_t start, std::size_t position) {
            const std::size_t end = values.size();
            const Item item = values[position];
            std::size_t child = 2 * position + 1;
            while (child < end) {
                const std::size_t right = child + 1;
                if (right < end && !(values[child] < values[right])) child = right;
                values[position] = values[child];
                position = child;
                child = 2 * position + 1;
            }
            values[position] = item;
            while (position > start) {
                const std::size_t parent = (position - 1) >> 1;
                if (!(values[position] < values[parent])) break;
                std::swap(values[position], values[parent]);
                position = parent;
            }
        };
        auto heappop = [&](std::vector<Item>& values) {
            const Item last = values.back();
            values.pop_back();
            if (values.empty()) return last;
            const Item result = values.front();
            values.front() = last;
            sift_down(values, 0, 0);
            return result;
        };
        auto heappush = [&](std::vector<Item>& values, const Item& item) {
            values.push_back(item);
            std::size_t position = values.size() - 1;
            while (position > 0) {
                const std::size_t parent = (position - 1) >> 1;
                if (!(values[position] < values[parent])) break;
                std::swap(values[position], values[parent]);
                position = parent;
            }
        };
        while (!heap.empty()) {
            const auto [distance, space] = heappop(heap);
            if (distance != distances[static_cast<std::size_t>(mask)][static_cast<std::size_t>(space)]) continue;
            for (const int neighbor : neighbors[static_cast<std::size_t>(space)]) {
                const int candidate = distance + node_cost[static_cast<std::size_t>(neighbor)];
                if (candidate >= distances[static_cast<std::size_t>(mask)][static_cast<std::size_t>(neighbor)]) continue;
                distances[static_cast<std::size_t>(mask)][static_cast<std::size_t>(neighbor)] = candidate;
                parents[{mask, neighbor}] = {1, space, 0};
                heappush(heap, {candidate, neighbor});
            }
        }
    };
    for (int terminal = 0; terminal < terminal_count; ++terminal) {
        const int mask = 1 << terminal;
        for (const int space : terminals[static_cast<std::size_t>(terminal)]) {
            distances[static_cast<std::size_t>(mask)][static_cast<std::size_t>(space)] = 0;
            parents[{mask, space}] = {0, terminal, 0};
        }
        relax(mask);
    }
    for (int mask = 1; mask <= full_mask; ++mask) {
        if ((mask & (mask - 1)) == 0) continue;
        for (int subset = (mask - 1) & mask; subset; subset = (subset - 1) & mask) {
            const int other = mask ^ subset;
            if (other && subset < other) for (const int space : allowed_spaces) {
                const int candidate = distances[static_cast<std::size_t>(subset)][static_cast<std::size_t>(space)] +
                    distances[static_cast<std::size_t>(other)][static_cast<std::size_t>(space)] - node_cost[static_cast<std::size_t>(space)];
                if (candidate < distances[static_cast<std::size_t>(mask)][static_cast<std::size_t>(space)]) {
                    distances[static_cast<std::size_t>(mask)][static_cast<std::size_t>(space)] = candidate;
                    parents[{mask, space}] = {2, subset, other};
                }
            }
        }
        relax(mask);
    }
    int best_space = -1;
    for (const int space : allowed_spaces)
        if (best_space < 0 || std::pair{distances[static_cast<std::size_t>(full_mask)][static_cast<std::size_t>(space)], space} <
                              std::pair{distances[static_cast<std::size_t>(full_mask)][static_cast<std::size_t>(best_space)], best_space}) best_space = space;
    if (best_space < 0 || distances[static_cast<std::size_t>(full_mask)][static_cast<std::size_t>(best_space)] >= infinity) {
        valid = false; return {};
    }
    std::set<int> used;
    std::set<std::pair<int, int>> visited;
    std::vector<std::pair<int, int>> visit_order;
    std::function<void(int, int)> collect = [&](int mask, int space) {
        if (!visited.insert({mask, space}).second) return;
        visit_order.push_back({mask, space});
        used.insert(space);
        const auto it = parents.find({mask, space});
        if (it == parents.end() || it->second.type == 0) return;
        if (it->second.type == 1) collect(mask, it->second.a);
        else { collect(it->second.a, space); collect(it->second.b, space); }
    };
    collect(full_mask, best_space);
    // Python's set iteration is deterministic for these integer space IDs
    // in the golden process.  Reproduce its collected-node order before the
    // final tuple sort, rather than relying on std::set's numeric order while
    // reconstructing a tied Steiner tree.
    std::vector<int> satellites;
    for (const auto [mask, space] : visit_order) {
        (void)mask;
        if (node_cost[static_cast<std::size_t>(space)] == 1 &&
            std::find(satellites.begin(), satellites.end(), space) == satellites.end())
            satellites.push_back(space);
    }
    std::sort(satellites.begin(), satellites.end());
    return satellites;
}

LocationList included_federation_locations(const GaiaState& state, const LocationList& selected,
                                           const std::vector<int>& satellite_spaces,
                                           const LocationClusters& clusters) {
    const auto spaces = board_spaces(state);
    std::set<std::array<int, 2>> connected;
    for (const int location : selected) connected.insert(location_coordinate(state, location));
    for (const int space : satellite_spaces) connected.insert(spaces[static_cast<std::size_t>(space)]);
    std::set<int> included(selected.begin(), selected.end());
    bool changed = true;
    while (changed) {
        changed = false;
        for (const auto& cluster : clusters) {
            bool already = false;
            for (const int location : cluster) already = already || included.contains(location);
            if (already) continue;
            bool adjacent = false;
            for (const int location : cluster) {
                const auto coordinate = location_coordinate(state, location);
                for (const auto& linked : connected)
                    if (hex_distance(coordinate[0], coordinate[1], linked[0], linked[1]) <= 1) adjacent = true;
            }
            if (!adjacent) continue;
            for (const int location : cluster) { included.insert(location); connected.insert(location_coordinate(state, location)); }
            changed = true;
        }
    }
    return {included.begin(), included.end()};
}

bool valid_federation_reduction(const GaiaState& state, int player, const LocationList& locations,
                                const std::vector<int>& satellites, int threshold) {
    if (satellites.empty()) return true;
    const auto spaces = board_spaces(state);
    const auto clusters = location_clusters(state, locations);
    std::vector<std::set<std::array<int, 2>>> cluster_coordinates;
    for (const auto& cluster : clusters) {
        std::set<std::array<int, 2>> coordinates;
        for (const int location : cluster) coordinates.insert(location_coordinate(state, location));
        cluster_coordinates.push_back(std::move(coordinates));
    }
    constexpr std::array<std::array<int, 2>, 6> directions{{{{1,0}},{{0,1}},{{-1,1}},{{-1,0}},{{0,-1}},{{1,-1}}}};
    for (const int removed : satellites) {
        std::set<std::array<int, 2>> remaining;
        for (const auto& group : cluster_coordinates) remaining.insert(group.begin(), group.end());
        for (const int space : satellites) if (space != removed) remaining.insert(spaces[static_cast<std::size_t>(space)]);
        std::vector<std::set<std::array<int, 2>>> components;
        while (!remaining.empty()) {
            std::set<std::array<int, 2>> component{*remaining.begin()};
            std::vector<std::array<int, 2>> frontier{*remaining.begin()};
            remaining.erase(remaining.begin());
            while (!frontier.empty()) {
                const auto current = frontier.back(); frontier.pop_back();
                for (const auto delta : directions) {
                    const std::array<int, 2> adjacent{current[0] + delta[0], current[1] + delta[1]};
                    if (remaining.erase(adjacent)) { component.insert(adjacent); frontier.push_back(adjacent); }
                }
            }
            components.push_back(std::move(component));
        }
        for (const auto& component : components) {
            std::vector<int> included_clusters;
            for (int index = 0; index < static_cast<int>(cluster_coordinates.size()); ++index) {
                bool intersects = false;
                for (const auto& coordinate : cluster_coordinates[static_cast<std::size_t>(index)])
                    intersects = intersects || component.contains(coordinate);
                if (intersects) included_clusters.push_back(index);
            }
            if (included_clusters.size() >= clusters.size()) continue;
            int power = 0;
            for (const int index : included_clusters) for (const int location : clusters[static_cast<std::size_t>(index)])
                power += location_power(state, player, location);
            if (power >= threshold) return false;
        }
    }
    return true;
}

struct FederationPlan { LocationList locations; std::vector<int> satellites; };

// Python's implementation uses heap entries (distance, space), but when a
// merge has equal cost it keeps the first parent produced by the subset loop.
// Keep the same strict tie handling in both relaxation phases.

bool federation_plan_details(const GaiaState& state, int player, FederationPlan& result) {
    const auto& info = state.players[static_cast<std::size_t>(player)];
    const bool ivits = info.faction == 7;
    const auto existing = ivits ? structure_locations(state, player, true) : LocationList{};
    auto clusters = location_clusters(state, structure_locations(state, player, false));
    if (!ivits) {
        clusters.erase(std::remove_if(clusters.begin(), clusters.end(), [&](const auto& cluster) {
            for (const int location : cluster) {
                const auto coordinate = location_coordinate(state, location);
                if (touches_existing_federation(state, player, coordinate[0], coordinate[1])) return true;
            }
            return false;
        }), clusters.end());
    }
    const int threshold = federation_threshold(state, player);
    int existing_power = 0;
    for (const int location : existing) existing_power += location_power(state, player, location);
    const int required = std::max(0, threshold - existing_power);
    std::vector<int> cluster_powers;
    for (const auto& cluster : clusters) {
        int power = 0; for (const int location : cluster) power += location_power(state, player, location);
        cluster_powers.push_back(power);
    }
    const int available = ivits ? info.qic : cycle_power(info);
    std::vector<std::vector<int>> selections;
    if (required == 0) selections.push_back({});
    else {
        std::function<void(int,int,int,std::vector<int>&)> choose = [&](int start, int need, int size, std::vector<int>& selected) {
            if (need == 0) {
                int power = 0; for (const int index : selected) power += cluster_powers[static_cast<std::size_t>(index)];
                if (power < required) return;
                for (const int index : selected) if (power - cluster_powers[static_cast<std::size_t>(index)] >= required) return;
                selections.push_back(selected); return;
            }
            for (int index = start; index <= static_cast<int>(clusters.size()) - need; ++index) {
                selected.push_back(index); choose(index + 1, need - 1, size, selected); selected.pop_back();
            }
            (void)size;
        };
        for (int size = 1; size <= static_cast<int>(clusters.size()); ++size) {
            std::vector<int> selected; choose(0, size, size, selected);
        }
    }
    std::sort(selections.begin(), selections.end(), [&](const auto& left, const auto& right) {
        auto key = [&](const auto& selection) {
            LocationClusters selected_clusters;
            int power = 0;
            for (const int index : selection) { selected_clusters.push_back(clusters[static_cast<std::size_t>(index)]); power += cluster_powers[static_cast<std::size_t>(index)]; }
            return std::tuple{federation_distance_estimate(state, selected_clusters, existing),
                              power - required, static_cast<int>(selection.size()), selection};
        };
        return key(left) < key(right);
    });
    for (const auto& selection : selections) {
        LocationList selected;
        for (const int index : selection) selected.insert(selected.end(), clusters[static_cast<std::size_t>(index)].begin(), clusters[static_cast<std::size_t>(index)].end());
        bool valid = false;
        auto satellites = minimum_satellite_path(state, player, selected, existing, valid);
        if (!valid) continue;
        auto included = included_federation_locations(state, selected, satellites, clusters);
        int total = existing_power;
        for (const int location : included) total += location_power(state, player, location);
        if (total < threshold) continue;
        if (!ivits && !valid_federation_reduction(state, player, included, satellites, threshold)) continue;
        if (static_cast<int>(satellites.size()) > available || info.satellites + static_cast<int>(satellites.size()) > 25) continue;
        result.locations = existing;
        result.locations.insert(result.locations.end(), included.begin(), included.end());
        result.satellites = std::move(satellites);
        return true;
    }
    return false;
}

void advance_after_action(GaiaState& state);

void continue_passive_charge(GaiaState& state) {
    int acting = state.pending_passive_charge_acting;
    if (acting < 0) acting = state.pending_taklons_charge_acting;
    if (acting < 0) throw std::logic_error("passive-charge acting player is missing");
    if (state.pending_passive_charge_queue_length > 0) {
        const auto offer = state.pending_passive_charge_queue[0];
        for (int index = 1; index < state.pending_passive_charge_queue_length; ++index)
            state.pending_passive_charge_queue[static_cast<std::size_t>(index - 1)] =
                state.pending_passive_charge_queue[static_cast<std::size_t>(index)];
        --state.pending_passive_charge_queue_length;
        state.player_to_move = offer[0];
        state.pending_passive_charge_player = offer[0];
        state.pending_passive_charge_acting = acting;
        state.pending_passive_charge_amount = offer[1];
        state.pending_taklons_charge_player = -1;
        state.pending_taklons_charge_acting = -1;
        state.pending_taklons_charge_amount = 0;
        return;
    }
    state.player_to_move = acting;
    state.pending_passive_charge_player = -1;
    state.pending_passive_charge_acting = -1;
    state.pending_passive_charge_planet = -1;
    state.pending_passive_charge_amount = 0;
    state.pending_passive_charge_queue_length = 0;
    state.pending_taklons_charge_player = -1;
    state.pending_taklons_charge_acting = -1;
    state.pending_taklons_charge_amount = 0;
    advance_after_action(state);
}

void advance_research(GaiaState& state, int player, int track, bool score_round) {
    if (!can_advance_research(state, player, track)) throw std::invalid_argument("cannot advance research track");
    auto& p = state.players[static_cast<std::size_t>(player)];
    const int old_level = p.tracks[static_cast<std::size_t>(track)]++;
    const int level = old_level + 1;
    if (old_level == 4) --p.federation_keys;
    if (old_level == 2) charge_power(p, 3);
    if (track == 0) {
        if (level == 1 || level == 4) p.ore = std::min(15, p.ore + 2);
        else if (level == 5) {
            gain_federation_reward(p, state.terraforming_federation_tile);
            ++p.federation_tokens;
            p.federation_keys += state.terraforming_federation_tile != 5;
            ++p.federation_tile_counts[static_cast<std::size_t>(state.terraforming_federation_tile)];
            score(state, player, 3);
        }
    } else if (track == 1 && (level == 1 || level == 3)) gain_qic(p, 1);
    else if (track == 2) gain_qic(p, std::array<int, 5>{1, 1, 2, 2, 4}[static_cast<std::size_t>(level - 1)]);
    else if (track == 3) {
        if (level == 1 || level == 3 || level == 4) ++p.gaiaformers;
        else if (level == 2) p.bowl_one += 3;
        else if (level == 5) {
            int gaia_planets = 0;
            for (int planet = 0; planet < kMaxPlanets; ++planet)
                gaia_planets += state.owners[static_cast<std::size_t>(planet)] == player && state.terrains[static_cast<std::size_t>(planet)] == static_cast<int>(Terrain::gaia);
            p.vp += 4 + gaia_planets;
        }
    } else if (track == 4 && level == 5) {
        p.credits = std::min(30, p.credits + 6); p.ore = std::min(15, p.ore + 3); charge_power(p, 6);
    } else if (track == 5 && level == 5) p.knowledge = std::min(15, p.knowledge + 9);
    if (p.advanced_tech_tiles & (std::uint32_t{1} << 12)) p.vp += 2;
    if (score_round) score(state, player, 1);
}

void score_mine(GaiaState& state, int player, int terrain) {
    if (state.round_number < 1 || state.round_number > kMaxRounds) return;
    const int tile = state.round_scoring_tiles[static_cast<std::size_t>(state.round_number - 1)];
    if (tile == 2) state.players[static_cast<std::size_t>(player)].vp += 2;
    else if (terrain == static_cast<int>(Terrain::gaia) && (tile == 6 || tile == 7))
        state.players[static_cast<std::size_t>(player)].vp += tile == 6 ? 3 : 4;
}

void grant_income(GaiaState& state) {
    constexpr std::array<std::array<int, 3>, 6> economy{{
        {0, 0, 0}, {2, 0, 1}, {2, 1, 2}, {3, 1, 3}, {4, 2, 4}, {0, 0, 0},
    }};
    constexpr std::array<int, 6> science{{0, 1, 2, 3, 4, 0}};
    constexpr std::array<std::array<int, 6>, 10> boosters{{
        {2, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 2}, {0, 1, 1, 0, 0, 0},
        {0, 1, 0, 0, 2, 0}, {2, 0, 0, 1, 0, 0}, {0, 1, 0, 0, 0, 0},
        {0, 1, 0, 0, 0, 0}, {0, 0, 1, 0, 0, 0}, {0, 0, 0, 0, 0, 4},
        {4, 0, 0, 0, 0, 0},
    }};
    constexpr std::array<int, 4> trading_credits{{3, 4, 4, 5}};
    constexpr std::array<int, 3> bescods_lab_credits{{3, 4, 5}};
    for (int player = 0; player < state.player_count; ++player) {
        auto& p = state.players[static_cast<std::size_t>(player)];
        const auto& f = kFactions[static_cast<std::size_t>(p.faction)];
        const int mines = building_count(state, player, Building::mine);
        const int trading = building_count(state, player, Building::trading_station);
        const int labs = building_count(state, player, Building::research_lab);
        const int institutes = building_count(state, player, Building::planetary_institute);
        const int booster = player_booster(state, player);
        const auto booster_income = booster >= 0 ? boosters[static_cast<std::size_t>(booster)] : std::array<int, 6>{};
        const auto economy_income = economy[static_cast<std::size_t>(std::clamp(p.tracks[4], 0, 5))];
        int credits = f.income_credits + economy_income[0] + booster_income[0];
        int ore = 1 + mines - (mines >= 3 ? 1 : 0) + f.income_ore + economy_income[1] + booster_income[1];
        // Research-lab income is faction-specific.  Nevlas has no printed
        // knowledge income from labs; each lab instead contributes two extra
        // power charge steps (handled below).
        int knowledge = 1 + f.income_knowledge + science[static_cast<std::size_t>(std::clamp(p.tracks[5], 0, 5))] + booster_income[2];
        int qic = f.income_qic + booster_income[3];
        int power_tokens = f.income_power + booster_income[4];
        int power_charge = institutes * 4 + economy_income[2] + booster_income[5];
        if (p.faction == 11) {
            knowledge += trading;
            for (int i = 0; i < std::min(labs, 3); ++i) credits += bescods_lab_credits[static_cast<std::size_t>(i)];
        } else {
            for (int i = 0; i < std::min(trading, 4); ++i) credits += trading_credits[static_cast<std::size_t>(i)];
            if (p.faction != 12) knowledge += labs;
        }
        knowledge += p.knowledge_academies * (p.faction == 13 ? 3 : 2);
        if (p.faction == 12) power_charge += labs * 2;
        if (institutes) {
            if (p.faction == 2) qic += institutes;
            else if (p.faction == 3) ore += institutes;
            else if (p.faction == 5 || p.faction == 11) power_tokens += 2 * institutes;
            else if (p.faction != 1) power_tokens += institutes;
        }
        if (has_active_standard_tech(state, p, 5)) { ++ore; ++power_charge; }
        if (has_active_standard_tech(state, p, 6)) { ++credits; ++knowledge; }
        if (has_active_standard_tech(state, p, 7)) credits += 4;
        if (p.faction == 3 && p.qic_academies == 0) { ore += qic; qic = 0; }
        p.credits = std::min(30, p.credits + credits);
        p.ore = std::min(15, p.ore + ore);
        p.knowledge = std::min(15, p.knowledge + knowledge);
        p.qic += qic;
        p.bowl_one += power_tokens;
        charge_power(p, power_charge);
    }
}

void gaia_phase(GaiaState& state) {
    int pending_terrans = -1;
    int pending_itars = -1;
    for (int player = 0; player < state.player_count; ++player) {
        auto& info = state.players[static_cast<std::size_t>(player)];
        if (info.faction == 9 && info.gaiaformers_in_gaia) {
            info.gaiaformers += info.gaiaformers_in_gaia;
            info.gaiaformers_in_gaia = 0;
        }
        if (info.faction == 0) {
            if (info.gaia_power && has_pi(state, player)) {
                if (pending_terrans < 0 && pending_itars < 0) pending_terrans = player;
            } else {
                info.bowl_two += info.gaia_power;
                info.gaia_power = 0;
            }
        } else if (info.faction == 13 && has_pi(state, player) &&
                   info.gaia_power >= 4 && has_tech_choice(state, player)) {
            if (pending_terrans < 0 && pending_itars < 0) pending_itars = player;
        } else {
            info.bowl_one += info.gaia_power;
            info.gaia_power = 0;
        }
        if (info.brainstone_bowl == 4) info.brainstone_bowl = info.faction == 0 ? 2 : 1;
    }
    for (int planet = 0; planet < kMaxPlanets; ++planet) {
        const auto index = static_cast<std::size_t>(planet);
        if (state.gaiaformer_owner[index] >= 0 && state.terrains[index] == static_cast<int>(Terrain::transdim))
            state.terrains[index] = static_cast<int>(Terrain::gaia);
    }
    state.player_to_move = pending_terrans >= 0 ? pending_terrans
        : pending_itars >= 0 ? pending_itars : state.first_player;
    state.pending_gaia_conversion_player = pending_terrans;
    state.pending_gaia_conversion_power = pending_terrans >= 0
        ? state.players[static_cast<std::size_t>(pending_terrans)].gaia_power : 0;
    state.pending_itars_gaia_player = pending_itars;
}

int booster_pass_points(const GaiaState& state, int player, int booster) {
    int points = 0;
    if (booster == 5) points += building_count(state, player, Building::mine);
    if (booster == 6) points += 2 * building_count(state, player, Building::trading_station);
    if (booster == 7) points += 3 * building_count(state, player, Building::research_lab);
    if (booster == 8) points += 4 * (building_count(state, player, Building::planetary_institute) + building_count(state, player, Building::academy));
    if (booster == 9) {
        int gaia = 0;
        for (int planet = 0; planet < kMaxPlanets; ++planet)
            gaia += state.owners[static_cast<std::size_t>(planet)] == player && state.terrains[static_cast<std::size_t>(planet)] == static_cast<int>(Terrain::gaia);
        points += gaia;
    }
    const auto& info = state.players[static_cast<std::size_t>(player)];
    if (info.advanced_tech_tiles & (std::uint32_t{1} << 9)) points += 3 * info.federation_tokens;
    if (info.advanced_tech_tiles & (std::uint32_t{1} << 10)) points += 3 * building_count(state, player, Building::research_lab);
    if (info.advanced_tech_tiles & (std::uint32_t{1} << 11)) points += static_cast<int>(std::popcount(info.colonized_types));
    return points;
}

void advance_after_action(GaiaState& state) {
    if (state.pending_gaia_conversion_player >= 0 || state.pending_itars_gaia_player >= 0 ||
        state.pending_passive_charge_player >= 0 || state.pending_taklons_charge_player >= 0 ||
        state.pending_tech_player >= 0 || state.pending_advanced_tech >= 0 ||
        state.pending_research_player >= 0 || state.pending_lost_planet_player >= 0 ||
        state.pending_power_terraform_player >= 0 || state.pending_booster_terraform_player >= 0 ||
        state.pending_booster_range_player >= 0) return;
    if (state.player_count <= 0) return;
    for (int i = 0; i < state.player_count; ++i) {
        const int candidate = (state.player_to_move + 1 + i) % state.player_count;
        if (!state.players[static_cast<std::size_t>(candidate)].passed) {
            state.player_to_move = candidate;
            return;
        }
    }
    ++state.round_number;
    if (state.round_number > kMaxRounds) {
        state.player_to_move = state.next_first_player >= 0 ? state.next_first_player : state.first_player;
        return;
    }
    for (int i = 0; i < state.player_count; ++i) {
        auto& p = state.players[static_cast<std::size_t>(i)];
        p.passed = false;
        p.used_qic_academy_action = false;
        p.used_standard_tech_action = false;
        p.used_advanced_tech_actions = 0;
        p.used_booster_action = false;
        p.used_ambas_swap_action = false;
        p.used_firaks_downgrade_action = false;
        p.used_bescods_research_action = false;
        p.used_ivits_space_station_action = false;
    }
    state.first_player = state.next_first_player >= 0 ? state.next_first_player : state.first_player;
    state.next_first_player = -1;
    state.player_to_move = state.first_player;
    state.used_power_actions = 0;
    state.used_qic_actions = 0;
    grant_income(state);
    gaia_phase(state);
}

} // namespace

GaiaState GaiaState::initial(std::int32_t players_count, std::int64_t seed) {
    return initial(players_count, seed, {});
}

GaiaState GaiaState::initial(
    std::int32_t players_count,
    std::int64_t seed,
    std::array<std::int32_t, kMaxPlayers> starting_offsets) {
    if (players_count < 2 || players_count > kMaxPlayers) throw std::invalid_argument("GaiaState supports two to four players");
    if (seed < 0) throw std::invalid_argument("setup seed must be non-negative");
    validate_vp_offsets(players_count, starting_offsets, "starting_vp_offsets");
    const auto setup = generate_gaia_setup(players_count, seed);
    GaiaState state;
    state.player_count = players_count;
    state.setup_seed = seed;
    state.setup_seed_stream_version = setup.seed_stream_version;
    state.setup_seed_streams = setup.seed_streams;
    state.setup_hash = setup.setup_hash;
    state.starting_vp_offsets = starting_offsets;
    state.published_vp_offsets = starting_offsets;
    state.first_player = setup.first_player;
    state.owners.fill(-1);
    state.buildings.fill(static_cast<int>(Building::empty));
    state.terrains.fill(static_cast<int>(Terrain::lost));
    state.planet_sectors.fill(-1);
    state.planet_source_ids.fill(-1);
    state.gaiaformer_owner.fill(-1);
    state.coexisting_mine_owner.fill(-1);
    state.satellite_owners.fill(0);
    state.space_station_owner.fill(-1);
    state.sector_tiles.fill(-1);
    state.booster_owner = setup.booster_owner;
    state.round_scoring_tiles = setup.round_scoring_tiles;
    state.final_scoring_tiles = setup.final_scoring_tiles;
    state.standard_tech_tiles = setup.standard_tech_tiles;
    state.advanced_tech_tiles = setup.advanced_tech_tiles;
    state.terraforming_federation_tile = setup.terraforming_federation_tile;
    for (int i = 0; i < 6; ++i) state.federation_tile_supply[static_cast<std::size_t>(i)] = i == state.terraforming_federation_tile ? 2 : 3;
    state.placement_order_length = static_cast<int>(setup.placement_order.size());
    std::copy(setup.placement_order.begin(), setup.placement_order.end(),
              state.placement_order.begin());
    for (int player = 0; player < players_count; ++player) {
        auto& p = state.players[static_cast<std::size_t>(player)];
        const auto faction = setup.factions[static_cast<std::size_t>(player)];
        const auto& f = kFactions[static_cast<std::size_t>(faction)];
        p.faction = faction;
        p.credits = f.credits; p.ore = f.ore; p.knowledge = f.knowledge; p.qic = f.qic;
        p.vp = 10 + starting_offsets[static_cast<std::size_t>(player)];
        p.bowl_one = f.power[0]; p.bowl_two = f.power[1]; p.bowl_three = f.power[2];
        if (f.brainstone) { p.brainstone_bowl = 1; ++p.bowl_one; }
        if (f.start_track >= 0) advance_research(state, player, f.start_track, false);
        state.booster_selection_order[static_cast<std::size_t>(player)] =
            (state.first_player - player - 1 + players_count * 2) % players_count;
    }
    state.sector_count = static_cast<int>(setup.sector_tiles.size());
    for (int i = 0; i < state.sector_count; ++i) {
        state.sector_tiles[static_cast<std::size_t>(i)] = setup.sector_tiles[static_cast<std::size_t>(i)];
        state.sector_rotations[static_cast<std::size_t>(i)] = setup.sector_rotations[static_cast<std::size_t>(i)];
        state.sector_centers[static_cast<std::size_t>(i)] = setup.sector_centers[static_cast<std::size_t>(i)];
    }
    for (int planet = 0; planet < kPrintedPlanetSlots; ++planet) {
        const auto index = static_cast<std::size_t>(planet);
        state.active_planets[index] = setup.active_planets[index];
        state.planet_q[index] = setup.planet_q[index];
        state.planet_r[index] = setup.planet_r[index];
        state.planet_source_q[index] = setup.planet_source_q[index];
        state.planet_source_r[index] = setup.planet_source_r[index];
        state.planet_source_ids[index] = setup.planet_source_ids[index];
        state.planet_sectors[index] = setup.planet_sectors[index];
        state.terrains[index] = setup.terrains[index];
    }
    state.planet_source_catalog_length = static_cast<int>(setup.planet_source_catalog.size());
    std::copy(setup.planet_source_catalog.begin(), setup.planet_source_catalog.end(),
              state.planet_source_catalog.begin());
    state.player_to_move = state.placement_order_length > 0 ? state.placement_order[0] : state.first_player;
    return state;
}

bool GaiaState::is_terminal() const noexcept { return round_number > kMaxRounds; }
bool GaiaState::is_starting_placement() const noexcept { return round_number == 0 && placement_step < placement_order_length; }
bool GaiaState::is_booster_selection() const noexcept {
    return round_number == 0 && placement_step >= placement_order_length && booster_selection_step < player_count;
}

std::vector<float> GaiaState::observation() const {
    // This is the standard-v22 flat observation used by the Python training
    // boundary.  Keep the append order in lockstep with
    // game/gaia_state.py::observation; the graph encoder has its own compact
    // representation and does not replace this raw NPZ contract.
    std::vector<float> values;
    values.reserve(observation_size());
    const auto add = [&values](float value) { values.push_back(value); };
    const auto onehot = [&add](int value, int count) {
        for (int candidate = 0; candidate < count; ++candidate)
            add(value == candidate ? 1.0F : 0.0F);
    };
    add(static_cast<float>(round_number) / static_cast<float>(kMaxRounds));
    add(static_cast<float>(player_count) / 4.0F);
    add(is_starting_placement() ? 1.0F : 0.0F);
    add(is_booster_selection() ? 1.0F : 0.0F);
    add(static_cast<float>(booster_selection_step) /
        static_cast<float>(std::max(1, player_count)));
    add(brainstone_selected ? 1.0F : 0.0F);
    for (int p = 0; p < player_count; ++p) add(player_to_move == p ? 1.0F : 0.0F);
    for (int p = 0; p < player_count; ++p) add(first_player == p ? 1.0F : 0.0F);
    for (int action = 0; action < 7; ++action)
        add((used_power_actions & (1 << action)) ? 1.0F : 0.0F);
    for (int action = 0; action < 3; ++action)
        add((used_qic_actions & (1 << action)) ? 1.0F : 0.0F);
    for (const int tile : round_scoring_tiles) onehot(tile, 10);
    for (const int tile : final_scoring_tiles) onehot(tile, 6);
    for (const int tile : standard_tech_tiles) onehot(tile, 9);
    for (const int tile : advanced_tech_tiles) onehot(tile, 15);
    onehot(terraforming_federation_tile, 6);
    for (const int count : federation_tile_supply) add(static_cast<float>(count) / 3.0F);
    add(pending_gaia_conversion_player >= 0 ? 1.0F : 0.0F);
    add(static_cast<float>(pending_gaia_conversion_power) / 15.0F);
    add(pending_itars_gaia_player >= 0 ? 1.0F : 0.0F);
    add(pending_passive_charge_player >= 0 ? 1.0F : 0.0F);
    add(static_cast<float>(pending_passive_charge_amount) / 7.0F);
    add(static_cast<float>(pending_passive_charge_queue_length) / 3.0F);
    add(pending_taklons_charge_player >= 0 ? 1.0F : 0.0F);
    add(static_cast<float>(pending_taklons_charge_amount) / 7.0F);
    add(pending_tech_player >= 0 ? 1.0F : 0.0F);
    add(pending_advanced_tech >= 0 ? 1.0F : 0.0F);
    add(pending_research_player >= 0 ? 1.0F : 0.0F);
    add(pending_lost_planet_player >= 0 ? 1.0F : 0.0F);
    add(pending_power_terraform_player >= 0 ? 1.0F : 0.0F);
    add(static_cast<float>(pending_power_terraform_steps) / 2.0F);
    add(pending_booster_terraform_player >= 0 ? 1.0F : 0.0F);
    add(pending_booster_range_player >= 0 ? 1.0F : 0.0F);
    add(pending_research_optional ? 1.0F : 0.0F);
    for (int track = -1; track < kTrackCount; ++track)
        add(pending_research_track == track ? 1.0F : 0.0F);
    for (int p = 0; p < player_count; ++p) add(pending_gaia_conversion_player == p ? 1.0F : 0.0F);
    for (int p = 0; p < player_count; ++p) add(pending_itars_gaia_player == p ? 1.0F : 0.0F);
    for (int p = 0; p < player_count; ++p) add(pending_passive_charge_player == p ? 1.0F : 0.0F);
    for (int p = 0; p < player_count; ++p) add(pending_passive_charge_acting == p ? 1.0F : 0.0F);
    for (int planet = 0; planet < kMaxPlanets; ++planet)
        add(pending_passive_charge_planet == planet ? 1.0F : 0.0F);
    for (int p = 0; p < player_count; ++p) add(pending_taklons_charge_player == p ? 1.0F : 0.0F);
    for (int p = 0; p < player_count; ++p) add(pending_lost_planet_player == p ? 1.0F : 0.0F);
    for (int tile = 0; tile < 15; ++tile) add(pending_advanced_tech == tile ? 1.0F : 0.0F);
    for (const int owner : booster_owner) {
        add(owner == -2 ? 1.0F : 0.0F);
        add(owner == -1 ? 1.0F : 0.0F);
        for (int p = 0; p < player_count; ++p) add(owner == p ? 1.0F : 0.0F);
    }
    for (int position = 0; position < kMaxSectors; ++position) {
        const bool present = position < sector_count;
        const int tile = present ? sector_tiles[static_cast<std::size_t>(position)] : -1;
        const int rotation = present ? sector_rotations[static_cast<std::size_t>(position)] : -1;
        add(present ? 1.0F : 0.0F);
        onehot(tile, 10);
        onehot(rotation, 6);
    }
    for (int p = 0; p < player_count; ++p) {
        const auto& info = players[static_cast<std::size_t>(p)];
        const int booster = player_booster(*this, p);
        add(static_cast<float>(info.credits) / 30.0F);
        add(static_cast<float>(info.ore) / 15.0F);
        add(static_cast<float>(info.knowledge) / 15.0F);
        add(static_cast<float>(info.qic) / 10.0F);
        add(static_cast<float>(info.vp) / 150.0F);
        const float offset_scale = player_count == 2 ? 30.0F : 50.0F;
        add(static_cast<float>(starting_vp_offsets[static_cast<std::size_t>(p)]) / offset_scale);
        add(static_cast<float>(info.bowl_one) / 15.0F);
        add(static_cast<float>(info.bowl_two) / 15.0F);
        add(static_cast<float>(info.bowl_three) / 15.0F);
        add(static_cast<float>(info.gaia_power) / 15.0F);
        add(static_cast<float>(info.gaiaformers) / 3.0F);
        add(static_cast<float>(info.gaiaformers_in_gaia) / 3.0F);
        add(static_cast<float>(info.federation_tokens) / 6.0F);
        add(static_cast<float>(info.federation_keys) / 3.0F);
        add(static_cast<float>(info.gleens_federation_tokens));
        add(static_cast<float>(info.board_federations) / 6.0F);
        add(static_cast<float>(info.knowledge_academies));
        add(static_cast<float>(info.qic_academies));
        add(info.used_qic_academy_action ? 1.0F : 0.0F);
        add(info.used_standard_tech_action ? 1.0F : 0.0F);
        add(info.used_booster_action ? 1.0F : 0.0F);
        add(info.used_firaks_downgrade_action ? 1.0F : 0.0F);
        add(info.used_bescods_research_action ? 1.0F : 0.0F);
        add(info.used_ivits_space_station_action ? 1.0F : 0.0F);
        for (int tile = 0; tile < 3; ++tile)
            add((info.used_advanced_tech_actions & (1 << tile)) ? 1.0F : 0.0F);
        add(info.passed ? 1.0F : 0.0F);
        for (int bowl = 0; bowl < 5; ++bowl) add(info.brainstone_bowl == bowl ? 1.0F : 0.0F);
        for (const int level : info.tracks) add(static_cast<float>(level) / 5.0F);
        for (int faction = 0; faction < 14; ++faction) add(info.faction == faction ? 1.0F : 0.0F);
        for (int candidate = 0; candidate < 10; ++candidate) add(booster == candidate ? 1.0F : 0.0F);
        for (int tile = 0; tile < 9; ++tile) add((info.tech_tiles & (1U << tile)) ? 1.0F : 0.0F);
        for (int tile = 0; tile < 9; ++tile) add((info.covered_tech_tiles & (1U << tile)) ? 1.0F : 0.0F);
        for (int tile = 0; tile < 15; ++tile) add((info.advanced_tech_tiles & (1U << tile)) ? 1.0F : 0.0F);
        for (const int count : info.federation_tile_counts) add(static_cast<float>(count) / 6.0F);
    }
    for (int planet = 0; planet < kMaxPlanets; ++planet) {
        const auto index = static_cast<std::size_t>(planet);
        add(active_planets[index] ? 1.0F : 0.0F);
        for (int terrain = 0; terrain < 10; ++terrain) add(terrains[index] == terrain ? 1.0F : 0.0F);
        add(owners[index] == -1 ? 1.0F : 0.0F);
        for (int p = 0; p < player_count; ++p) add(owners[index] == p ? 1.0F : 0.0F);
        for (int building = 0; building < 6; ++building) add(buildings[index] == building ? 1.0F : 0.0F);
        add(coexisting_mine_owner[index] == -1 ? 1.0F : 0.0F);
        for (int p = 0; p < player_count; ++p) add(coexisting_mine_owner[index] == p ? 1.0F : 0.0F);
        add(coexisting_mine_federated[index] ? 1.0F : 0.0F);
        add(gaiaformer_owner[index] == -1 ? 1.0F : 0.0F);
        for (int p = 0; p < player_count; ++p) add(gaiaformer_owner[index] == p ? 1.0F : 0.0F);
        add(federated[index] ? 1.0F : 0.0F);
    }
    const auto spaces = board_spaces(*this);
    for (int space = 0; space < kMaxBoardSpaces; ++space) {
        const bool present = space < static_cast<int>(spaces.size());
        const int owner = space_station_owner[static_cast<std::size_t>(space)];
        add(present ? 1.0F : 0.0F);
        add(present && owner == -1 ? 1.0F : 0.0F);
        for (int p = 0; p < player_count; ++p) add(present && owner == p ? 1.0F : 0.0F);
        add(present && space_station_federated[static_cast<std::size_t>(space)] ? 1.0F : 0.0F);
        for (int p = 0; p < player_count; ++p)
            add(present && (satellite_owners[static_cast<std::size_t>(space)] & (1 << p)) ? 1.0F : 0.0F);
    }
    if (values.size() != observation_size())
        throw std::logic_error("standard-v22 observation length mismatch");
    return values;
}

std::string GaiaState::canonical_json() const {
    std::ostringstream out;
    out << '{';
    append_key(out, "active_planets"); append_array(out, active_planets); out << ',';
    append_key(out, "advanced_tech_tiles"); append_array(out, advanced_tech_tiles); out << ',';
    append_key(out, "booster_owner"); append_array(out, booster_owner); out << ',';
    append_key(out, "booster_selection_order"); out << '['; for (int i = 0; i < player_count; ++i) { if (i) out << ','; out << booster_selection_order[static_cast<std::size_t>(i)]; } out << "],";
    append_key(out, "booster_selection_step"); out << booster_selection_step << ',';
    append_key(out, "brainstone_selected"); out << (brainstone_selected ? "true" : "false") << ',';
    append_key(out, "buildings"); append_array(out, buildings); out << ',';
    append_key(out, "coexisting_mine_federated"); append_array(out, coexisting_mine_federated); out << ',';
    append_key(out, "coexisting_mine_owner"); append_array(out, coexisting_mine_owner); out << ',';
    append_key(out, "compensation_version"); append_json_string(out, compensation_version); out << ',';
    append_key(out, "federated"); append_array(out, federated); out << ',';
    append_key(out, "federation_tile_supply"); append_array(out, federation_tile_supply); out << ',';
    append_key(out, "final_scoring_tiles"); append_array(out, final_scoring_tiles); out << ',';
    append_key(out, "first_player"); out << first_player << ',';
    append_key(out, "gaiaformer_owner"); append_array(out, gaiaformer_owner); out << ',';
    append_key(out, "map_mode"); append_json_string(out, map_mode); out << ',';
    append_key(out, "next_first_player"); out << next_first_player << ',';
    append_key(out, "owners"); append_array(out, owners); out << ',';
    append_key(out, "pending_advanced_tech"); out << pending_advanced_tech << ',';
    append_key(out, "pending_booster_range_player"); out << pending_booster_range_player << ',';
    append_key(out, "pending_booster_terraform_player"); out << pending_booster_terraform_player << ',';
    append_key(out, "pending_gaia_conversion_player"); out << pending_gaia_conversion_player << ',';
    append_key(out, "pending_gaia_conversion_power"); out << pending_gaia_conversion_power << ',';
    append_key(out, "pending_itars_gaia_player"); out << pending_itars_gaia_player << ',';
    append_key(out, "pending_lost_planet_player"); out << pending_lost_planet_player << ',';
    append_key(out, "pending_passive_charge_acting"); out << pending_passive_charge_acting << ',';
    append_key(out, "pending_passive_charge_amount"); out << pending_passive_charge_amount << ',';
    append_key(out, "pending_passive_charge_planet"); out << pending_passive_charge_planet << ',';
    append_key(out, "pending_passive_charge_player"); out << pending_passive_charge_player << ',';
    append_key(out, "pending_passive_charge_queue");
    out << '['; for (int i = 0; i < pending_passive_charge_queue_length; ++i) { if (i) out << ','; append_array(out, pending_passive_charge_queue[static_cast<std::size_t>(i)]); } out << "],";
    append_key(out, "pending_power_terraform_player"); out << pending_power_terraform_player << ',';
    append_key(out, "pending_power_terraform_steps"); out << pending_power_terraform_steps << ',';
    append_key(out, "pending_research_optional"); out << (pending_research_optional ? "true" : "false") << ',';
    append_key(out, "pending_research_player"); out << pending_research_player << ',';
    append_key(out, "pending_research_track"); out << pending_research_track << ',';
    append_key(out, "pending_taklons_charge_acting"); out << pending_taklons_charge_acting << ',';
    append_key(out, "pending_taklons_charge_amount"); out << pending_taklons_charge_amount << ',';
    append_key(out, "pending_taklons_charge_player"); out << pending_taklons_charge_player << ',';
    append_key(out, "pending_tech_player"); out << pending_tech_player << ',';
    append_key(out, "placement_order"); out << '['; for (int i = 0; i < placement_order_length; ++i) { if (i) out << ','; out << placement_order[static_cast<std::size_t>(i)]; } out << "],";
    append_key(out, "placement_step"); out << placement_step << ',';
    append_key(out, "planet_q"); append_array(out, planet_q); out << ',';
    append_key(out, "planet_r"); append_array(out, planet_r); out << ',';
    append_key(out, "planet_sectors"); append_array(out, planet_sectors); out << ',';
    append_key(out, "planet_source_catalog"); out << '['; for (int i = 0; i < planet_source_catalog_length; ++i) { if (i) out << ','; append_array(out, planet_source_catalog[static_cast<std::size_t>(i)]); } out << "],";
    append_key(out, "planet_source_ids"); append_array(out, planet_source_ids); out << ',';
    append_key(out, "planet_source_q"); append_array(out, planet_source_q); out << ',';
    append_key(out, "planet_source_r"); append_array(out, planet_source_r); out << ',';
    append_key(out, "player_count"); out << player_count << ',';
    append_key(out, "player_to_move"); out << player_to_move << ',';
    append_key(out, "players"); out << '['; for (int i = 0; i < player_count; ++i) { if (i) out << ','; append_player(out, players[static_cast<std::size_t>(i)]); } out << "],";
    append_key(out, "published_vp_offsets"); out << '['; for (int player = 0; player < player_count; ++player) { if (player) out << ','; out << published_vp_offsets[static_cast<std::size_t>(player)]; } out << "],";
    append_key(out, "round_number"); out << round_number << ',';
    append_key(out, "round_scoring_tiles"); append_array(out, round_scoring_tiles); out << ',';
    append_key(out, "satellite_owners"); append_array(out, satellite_owners); out << ',';
    append_key(out, "sector_centers"); out << '['; for (int i = 0; i < sector_count; ++i) { if (i) out << ','; append_array(out, sector_centers[static_cast<std::size_t>(i)]); } out << "],";
    append_key(out, "sector_rotations"); out << '['; for (int i = 0; i < sector_count; ++i) { if (i) out << ','; out << sector_rotations[static_cast<std::size_t>(i)]; } out << "],";
    append_key(out, "sector_tiles"); out << '['; for (int i = 0; i < sector_count; ++i) { if (i) out << ','; out << sector_tiles[static_cast<std::size_t>(i)]; } out << "],";
    append_key(out, "setup_hash"); append_json_string(out, setup_hash); out << ',';
    append_key(out, "setup_seed"); out << setup_seed << ',';
    append_key(out, "setup_seed_stream_version"); append_json_string(out, setup_seed_stream_version); out << ',';
    append_key(out, "setup_seed_streams"); out << '['; for (std::size_t i = 0; i < setup_seed_streams.size(); ++i) { if (i) out << ','; out << '['; append_json_string(out, setup_seed_streams[i].first); out << ',' << setup_seed_streams[i].second << ']'; } out << "],";
    append_key(out, "space_station_federated"); append_array(out, space_station_federated); out << ',';
    append_key(out, "space_station_owner"); append_array(out, space_station_owner); out << ',';
    append_key(out, "standard_tech_tiles"); append_array(out, standard_tech_tiles); out << ',';
    append_key(out, "starting_planets"); out << '['; for (int player = 0; player < player_count; ++player) { if (player) out << ','; out << '['; for (int i = 0; i < starting_planet_count[static_cast<std::size_t>(player)]; ++i) { if (i) out << ','; out << starting_planets[static_cast<std::size_t>(player)][static_cast<std::size_t>(i)]; } out << ']'; } out << "],";
    append_key(out, "starting_vp_offsets"); out << '['; for (int player = 0; player < player_count; ++player) { if (player) out << ','; out << starting_vp_offsets[static_cast<std::size_t>(player)]; } out << "],";
    append_key(out, "terraforming_federation_tile"); out << terraforming_federation_tile << ',';
    append_key(out, "terrains"); append_array(out, terrains); out << ',';
    append_key(out, "used_power_actions"); out << used_power_actions << ',';
    append_key(out, "used_qic_actions"); out << used_qic_actions << ',';
    append_key(out, "vp_offset_perturbations"); out << '['; for (int player = 0; player < player_count; ++player) { if (player) out << ','; out << vp_offset_perturbations[static_cast<std::size_t>(player)]; } out << ']';
    out << '}';
    return out.str();
}

std::string GaiaState::state_hash() const {
    return state_hash_from_canonical_json(canonical_json());
}

std::vector<ActionTuple> GaiaState::legal_action_tuples() const {
    if (is_terminal()) return {};
    const int player = current_player(*this);
    const auto& p = players[static_cast<std::size_t>(player)];
    auto scalar = [](ActionType type) { return ActionTuple::create(type, {}); };
    if (pending_gaia_conversion_player >= 0) {
        std::vector<ActionTuple> actions{scalar(ActionType::terrans_gaia_finish)};
        if (pending_gaia_conversion_power >= 1 && p.credits < 30) actions.push_back(scalar(ActionType::terrans_gaia_credit));
        if (pending_gaia_conversion_power >= 3 && p.ore < 15) actions.push_back(scalar(ActionType::terrans_gaia_ore));
        if (pending_gaia_conversion_power >= 4 && p.knowledge < 15) actions.push_back(scalar(ActionType::terrans_gaia_knowledge));
        if (pending_gaia_conversion_power >= 4) actions.push_back(scalar(ActionType::terrans_gaia_qic));
        return actions;
    }
    if (pending_passive_charge_player >= 0)
        return {scalar(ActionType::passive_charge_accept), scalar(ActionType::passive_charge_decline)};
    if (pending_taklons_charge_player >= 0)
        return {scalar(ActionType::taklons_passive_before), scalar(ActionType::taklons_passive_after)};
    if (pending_itars_gaia_player >= 0 && pending_advanced_tech < 0 &&
        pending_research_player < 0 && pending_lost_planet_player < 0 && pending_tech_player < 0) {
        std::vector<ActionTuple> actions{scalar(ActionType::itars_gaia_finish)};
        if (p.gaia_power >= 4 && has_tech_choice(*this, player))
            actions.push_back(scalar(ActionType::itars_gaia_technology));
        return actions;
    }
    if (is_starting_placement()) {
        std::vector<ActionTuple> actions;
        for (int planet = 0; planet < kMaxPlanets; ++planet) if (is_home_planet(*this, player, planet)) {
            actions.push_back(ActionTuple::create(ActionType::place_starting_structure, {planet}));
        }
        return actions;
    }
    if (is_booster_selection()) {
        std::vector<ActionTuple> actions;
        for (int booster = 0; booster < kBoosterCount; ++booster) if (booster_owner[static_cast<std::size_t>(booster)] == -1)
            actions.push_back(ActionTuple::create(ActionType::pass_booster, {booster}));
        return actions;
    }
    if (pending_advanced_tech >= 0) {
        std::vector<ActionTuple> actions;
        for (int space = 0; space < 9; ++space) {
            const int tile = standard_tech_tiles[static_cast<std::size_t>(space)];
            const auto mask = std::uint32_t{1} << static_cast<unsigned>(tile);
            if ((p.tech_tiles & mask) && !(p.covered_tech_tiles & mask))
                actions.push_back(ActionTuple::create(ActionType::tech_take, {space}));
        }
        return actions;
    }
    if (pending_research_player >= 0) {
        std::vector<ActionTuple> actions;
        const int begin = pending_research_track >= 0 ? pending_research_track : 0;
        const int end = pending_research_track >= 0 ? pending_research_track + 1 : kTrackCount;
        for (int track = begin; track < end; ++track)
            if (can_advance_research(*this, player, track)) actions.push_back(ActionTuple::create(ActionType::research, {track}));
        if (pending_research_optional) actions.push_back(scalar(ActionType::skip_tech_research));
        return actions;
    }
    auto is_empty_board_space = [this](int space) {
        const auto spaces = board_spaces(*this);
        if (space < 0 || space >= static_cast<int>(spaces.size()) || space_station_owner[static_cast<std::size_t>(space)] >= 0) return false;
        for (int planet = 0; planet < kMaxPlanets; ++planet)
            if (active_planets[static_cast<std::size_t>(planet)] &&
                planet_q[static_cast<std::size_t>(planet)] == spaces[static_cast<std::size_t>(space)][0] &&
                planet_r[static_cast<std::size_t>(planet)] == spaces[static_cast<std::size_t>(space)][1]) return false;
        return true;
    };
    if (pending_lost_planet_player >= 0) {
        std::vector<ActionTuple> actions;
        const auto spaces = board_spaces(*this);
        if (!active_planets[kMaxPlanets - 1]) for (int space = 0; space < static_cast<int>(spaces.size()); ++space)
            if (is_empty_board_space(space) && p.qic >= coordinate_range_qic_cost(*this, player, spaces[static_cast<std::size_t>(space)][0], spaces[static_cast<std::size_t>(space)][1]))
                actions.push_back(ActionTuple::create(ActionType::lost_planet, {space}));
        return actions;
    }
    if (pending_power_terraform_player >= 0 || pending_booster_terraform_player >= 0) {
        const int free_steps = pending_power_terraform_player >= 0 ? pending_power_terraform_steps : 1;
        std::vector<ActionTuple> actions;
        for (int planet = 0; planet < kMaxPlanets; ++planet)
            if (can_build_mine(*this, player, planet, free_steps)) actions.push_back(ActionTuple::create(ActionType::build_mine, {planet}));
        return actions;
    }
    if (pending_booster_range_player >= 0) {
        std::vector<ActionTuple> actions;
        for (int planet = 0; planet < kMaxPlanets; ++planet) {
            if (can_build_mine(*this, player, planet, 0, 3)) actions.push_back(ActionTuple::create(ActionType::build_mine, {planet}));
            if (can_start_gaia(*this, player, planet, 3)) actions.push_back(ActionTuple::create(ActionType::gaia_project, {planet}));
        }
        return actions;
    }
    auto append_tech_actions = [this, player](std::vector<ActionTuple>& actions) {
        const auto& info = players[static_cast<std::size_t>(player)];
        for (int space = 0; space < 9; ++space) {
            const int tile = standard_tech_tiles[static_cast<std::size_t>(space)];
            if ((info.tech_tiles & (std::uint32_t{1} << static_cast<unsigned>(tile))) == 0)
                actions.push_back(ActionTuple::create(ActionType::tech_take, {space}));
        }
        bool cover = false;
        for (const int tile : standard_tech_tiles)
            cover = cover || ((info.tech_tiles & (std::uint32_t{1} << static_cast<unsigned>(tile))) != 0 &&
                              (info.covered_tech_tiles & (std::uint32_t{1} << static_cast<unsigned>(tile))) == 0);
        if (info.federation_keys > 0 && cover) for (int track = 0; track < kTrackCount; ++track) {
            const int tile = advanced_tech_tiles[static_cast<std::size_t>(track)];
            bool taken = false;
            for (int opponent = 0; opponent < player_count; ++opponent)
                taken = taken || (players[static_cast<std::size_t>(opponent)].advanced_tech_tiles & (std::uint32_t{1} << static_cast<unsigned>(tile))) != 0;
            if (info.tracks[static_cast<std::size_t>(track)] >= 4 && !taken)
                actions.push_back(ActionTuple::create(ActionType::tech_take, {9 + track}));
        }
    };
    if (pending_tech_player >= 0) {
        std::vector<ActionTuple> actions;
        append_tech_actions(actions);
        return actions;
    }
    auto append_standard_free = [this, &p, &scalar](std::vector<ActionTuple>& actions, bool use_brainstone) {
        if (use_brainstone) {
            if (p.brainstone_bowl != 3) return;
            if (p.credits <= 27) actions.push_back(scalar(ActionType::power_to_credit));
            if (p.ore < 15) actions.push_back(scalar(ActionType::power_to_ore));
            if (can_spend_power(p, 4, true)) {
                if (p.knowledge < 15) actions.push_back(scalar(ActionType::power_to_knowledge));
                actions.push_back(scalar(ActionType::power_to_qic));
            }
            return;
        }
        const int power = ordinary_power(p);
        if (power >= 1 && p.credits < 30) actions.push_back(scalar(ActionType::power_to_credit));
        if (power >= 3 && p.ore < 15) actions.push_back(scalar(ActionType::power_to_ore));
        if (power >= 4 && p.knowledge < 15) actions.push_back(scalar(ActionType::power_to_knowledge));
        if (power >= 4) actions.push_back(scalar(ActionType::power_to_qic));
        if (p.qic >= 1 && p.ore < 15) actions.push_back(scalar(ActionType::qic_to_ore));
        if (p.ore >= 1 && p.credits < 30) actions.push_back(scalar(ActionType::ore_to_credit));
        if (p.knowledge >= 1 && p.credits < 30) actions.push_back(scalar(ActionType::knowledge_to_credit));
    };
    if (brainstone_selected) {
        std::vector<ActionTuple> actions;
        if (p.brainstone_bowl == 3) {
            for (int power_action = 0; power_action < 7; ++power_action) {
                const int free_steps = power_terraform_steps(power_action);
                bool target = free_steps == 0;
                for (int planet = 0; !target && planet < kMaxPlanets; ++planet) target = can_build_mine(*this, player, planet, free_steps);
                if (can_spend_power(p, power_action_cost(*this, player, power_action), true) &&
                    !(used_power_actions & (1 << power_action)) && target)
                    actions.push_back(ActionTuple::create(ActionType::power_action, {power_action}));
            }
            append_standard_free(actions, true);
        }
        return actions;
    }
    std::vector<ActionTuple> actions;
    for (int planet = 0; planet < kMaxPlanets; ++planet) {
        if (!active_planets[static_cast<std::size_t>(planet)]) continue;
        const int terrain = terrains[static_cast<std::size_t>(planet)];
        if (terrain != static_cast<int>(Terrain::transdim) && can_build_mine(*this, player, planet))
            actions.push_back(ActionTuple::create(ActionType::build_mine, {planet}));
        if (terrain == static_cast<int>(Terrain::transdim) && can_start_gaia(*this, player, planet))
            actions.push_back(ActionTuple::create(ActionType::gaia_project, {planet}));
        if (owners[static_cast<std::size_t>(planet)] != player || terrain == static_cast<int>(Terrain::lost)) continue;
        const auto level = static_cast<Building>(buildings[static_cast<std::size_t>(planet)]);
        if (level == Building::mine && p.faction == 5 && has_pi(*this, player) && !p.used_ambas_swap_action)
            actions.push_back(ActionTuple::create(ActionType::upgrade_planetary_institute, {planet}));
        if (level == Building::mine && building_count(*this, player, Building::trading_station) < 4) {
            const int credits = has_nearby_opponent(*this, player, planet) ? 3 : 6;
            if (p.credits >= credits && p.ore >= 2) actions.push_back(ActionTuple::create(ActionType::upgrade_trading, {planet}));
        } else if (level == Building::trading_station) {
            if (building_count(*this, player, Building::research_lab) < 3 && p.credits >= 5 && p.ore >= 3 && has_tech_choice(*this, player))
                actions.push_back(ActionTuple::create(ActionType::upgrade_lab, {planet}));
            if (p.faction == 11) {
                if (building_count(*this, player, Building::academy) < 2 && p.credits >= 6 && p.ore >= 6 && has_tech_choice(*this, player)) {
                    if (p.knowledge_academies < 1) actions.push_back(ActionTuple::create(ActionType::upgrade_academy, {planet}));
                    if (p.qic_academies < 1) actions.push_back(ActionTuple::create(ActionType::upgrade_qic_academy, {planet}));
                }
            } else if (building_count(*this, player, Building::planetary_institute) < 1 && p.credits >= 6 && p.ore >= 4)
                actions.push_back(ActionTuple::create(ActionType::upgrade_planetary_institute, {planet}));
        } else if (level == Building::research_lab) {
            const int lowest = *std::min_element(p.tracks.begin(), p.tracks.end());
            const bool firaks = p.faction == 10 && has_pi(*this, player) && !p.used_firaks_downgrade_action &&
                                building_count(*this, player, Building::trading_station) < 4 && has_research_choice(*this, player);
            (void)lowest;
            if (firaks) actions.push_back(ActionTuple::create(ActionType::upgrade_trading, {planet}));
            if (p.faction == 11) {
                if (building_count(*this, player, Building::planetary_institute) < 1 && p.credits >= 6 && p.ore >= 4)
                    actions.push_back(ActionTuple::create(ActionType::upgrade_planetary_institute, {planet}));
            } else if (building_count(*this, player, Building::academy) < 2 && p.credits >= 6 && p.ore >= 6 && has_tech_choice(*this, player)) {
                if (p.knowledge_academies < 1) actions.push_back(ActionTuple::create(ActionType::upgrade_academy, {planet}));
                if (p.qic_academies < 1) actions.push_back(ActionTuple::create(ActionType::upgrade_qic_academy, {planet}));
            }
        }
    }
    if (p.knowledge >= 4) {
        for (int track = 0; track < kTrackCount; ++track)
            if (can_advance_research(*this, player, track))
                actions.push_back(ActionTuple::create(ActionType::research, {track}));
    }
    for (int power_action = 0; power_action < 7; ++power_action) {
        const int free_steps = power_terraform_steps(power_action);
        bool target = free_steps == 0;
        for (int planet = 0; !target && planet < kMaxPlanets; ++planet) target = can_build_mine(*this, player, planet, free_steps);
        if (can_spend_power(p, power_action_cost(*this, player, power_action)) &&
            !(used_power_actions & (1 << power_action)) && target)
            actions.push_back(ActionTuple::create(ActionType::power_action, {power_action}));
    }
    // Python's canonical action ordering places federation actions after all
    // power actions.  Keep this order stable because ActionTuple lists are
    // part of the cross-language golden contract.
    FederationPlan federation_plan;
    if (federation_plan_details(*this, player, federation_plan))
        for (int tile = 0; tile < 6; ++tile)
            if (federation_tile_supply[static_cast<std::size_t>(tile)] > 0)
                actions.push_back(ActionTuple::create(ActionType::federation, {tile}));
    if (p.qic_academies && !p.used_qic_academy_action) actions.push_back(scalar(ActionType::qic_academy));
    if (has_active_standard_tech(*this, p, 8) && !p.used_standard_tech_action) actions.push_back(scalar(ActionType::standard_tech));
    for (int tile = 0; tile < 3; ++tile)
        if ((p.advanced_tech_tiles & (std::uint32_t{1} << tile)) && !(p.used_advanced_tech_actions & (1 << tile)))
            actions.push_back(ActionTuple::create(ActionType::advanced_tech, {tile}));
    if (!(used_qic_actions & 1) && p.qic >= 4 && has_tech_choice(*this, player)) actions.push_back(scalar(ActionType::qic_tech));
    if (!(used_qic_actions & 2) && p.qic >= 3) {
        for (int tile = 0; tile < 6; ++tile) if (p.federation_tile_counts[static_cast<std::size_t>(tile)] > 0)
            actions.push_back(ActionTuple::create(ActionType::qic_federation, {tile}));
        if (p.gleens_federation_tokens) actions.push_back(ActionTuple::create(ActionType::qic_federation, {6}));
    }
    if (!(used_qic_actions & 4) && p.qic >= 2) actions.push_back(scalar(ActionType::qic_planet_types));
    const int booster = player_booster(*this, player);
    if (!p.used_booster_action) {
        if (booster == 0) {
            bool target = false;
            for (int planet = 0; !target && planet < kMaxPlanets; ++planet) target = can_build_mine(*this, player, planet, 1);
            if (target) actions.push_back(scalar(ActionType::booster_terraform));
        } else if (booster == 1) {
            bool target = false;
            for (int planet = 0; !target && planet < kMaxPlanets; ++planet)
                target = can_build_mine(*this, player, planet, 0, 3) || can_start_gaia(*this, player, planet, 3);
            if (target) actions.push_back(scalar(ActionType::booster_range));
        }
    }
    if (round_number == kMaxRounds) {
        actions.push_back(ActionTuple::create(ActionType::pass_final, {}));
    } else {
        for (int booster = 0; booster < kBoosterCount; ++booster)
            if (booster_owner[static_cast<std::size_t>(booster)] == -1)
                actions.push_back(ActionTuple::create(ActionType::pass_booster, {booster}));
    }
    if (brainstone_action_available(*this, player)) actions.push_back(scalar(ActionType::brainstone));
    if (p.faction == 6 && has_pi(*this, player)) {
        if (p.credits >= 3 && p.ore < 15) actions.push_back(scalar(ActionType::terrans_gaia_ore));
        if (p.credits >= 4 && p.knowledge < 15) actions.push_back(scalar(ActionType::terrans_gaia_knowledge));
        if (p.credits >= 4) actions.push_back(scalar(ActionType::terrans_gaia_qic));
    }
    if (p.faction == 7 && has_pi(*this, player) && !p.used_ivits_space_station_action) {
        const auto spaces = board_spaces(*this);
        int count = 0;
        for (const int owner : space_station_owner) count += owner == player;
        if (count < kMaxRounds) for (int space = 0; space < static_cast<int>(spaces.size()); ++space)
            if (is_empty_board_space(space) && is_coordinate_reachable(*this, player, spaces[static_cast<std::size_t>(space)][0], spaces[static_cast<std::size_t>(space)][1]))
                actions.push_back(ActionTuple::create(ActionType::ivits_space_station, {space}));
    }
    if (p.faction == 11 && !p.used_bescods_research_action) {
        const int lowest = *std::min_element(p.tracks.begin(), p.tracks.end());
        for (int track = 0; track < kTrackCount; ++track)
            if (p.tracks[static_cast<std::size_t>(track)] == lowest && can_advance_research(*this, player, track))
                actions.push_back(ActionTuple::create(ActionType::bescods_research, {track}));
    }
    if (p.faction == 9 && p.gaiaformers > 0) actions.push_back(scalar(ActionType::bal_taks_gaiaformer_qic));
    if (p.faction == 13 && p.bowl_two >= 2) actions.push_back(scalar(ActionType::itars_burn_power));
    if (p.faction == 12) {
        if (p.bowl_three >= 1 && p.knowledge < 15) actions.push_back(scalar(ActionType::nevlas_power_to_gaia));
        if (has_pi(*this, player)) {
            if (p.bowl_three >= 1 && p.credits < 30) actions.push_back(scalar(ActionType::nevlas_credits));
            if (p.bowl_three >= 2 && (p.credits < 30 || p.ore < 15)) actions.push_back(scalar(ActionType::nevlas_credit_ore));
            if (p.bowl_three >= 3 && p.ore < 15) actions.push_back(scalar(ActionType::nevlas_ore));
            if (p.bowl_three >= 2) actions.push_back(scalar(ActionType::nevlas_qic));
            if (p.bowl_three >= 2 && p.knowledge < 15) actions.push_back(scalar(ActionType::nevlas_knowledge));
        }
    }
    append_standard_free(actions, false);
    return actions;
}

GaiaState GaiaState::apply(const ActionTuple& action) const {
    if (!action.valid()) throw std::invalid_argument("invalid ActionTuple");
    const auto legal = legal_action_tuples();
    if (std::find(legal.begin(), legal.end(), action) == legal.end()) throw std::invalid_argument("illegal action tuple");
    GaiaState next = *this;
    const int actor = player_to_move;
    switch (action.action_type) {
    case ActionType::passive_charge_accept:
    case ActionType::passive_charge_decline: {
        const int charging = next.pending_passive_charge_player;
        const int acting = next.pending_passive_charge_acting;
        const int amount = next.pending_passive_charge_amount;
        if (charging < 0 || acting < 0 || amount <= 0) throw std::invalid_argument("passive charge is not pending");
        next.pending_passive_charge_player = -1;
        next.pending_passive_charge_amount = 0;
        if (action.action_type == ActionType::passive_charge_accept) {
            auto& info = next.players[static_cast<std::size_t>(charging)];
            if (info.faction == 4 && has_pi(next, charging)) {
                next.pending_taklons_charge_player = charging;
                next.pending_taklons_charge_acting = acting;
                next.pending_taklons_charge_amount = amount;
                return next;
            }
            const int charged = charge_power(info, amount);
            info.vp -= std::max(0, charged - 1);
        }
        continue_passive_charge(next);
        return next;
    }
    case ActionType::taklons_passive_before:
    case ActionType::taklons_passive_after: {
        const int charging = next.pending_taklons_charge_player;
        const int amount = next.pending_taklons_charge_amount;
        if (charging < 0 || next.pending_taklons_charge_acting < 0 || amount <= 0)
            throw std::invalid_argument("Taklons passive charge is not pending");
        auto& info = next.players[static_cast<std::size_t>(charging)];
        const bool before = action.action_type == ActionType::taklons_passive_before;
        if (before) ++info.bowl_one;
        const int charged = charge_power(info, amount);
        info.vp -= std::max(0, charged - 1);
        if (!before) ++info.bowl_one;
        next.pending_taklons_charge_player = -1;
        next.pending_taklons_charge_acting = -1;
        next.pending_taklons_charge_amount = 0;
        continue_passive_charge(next);
        return next;
    }
    case ActionType::power_to_credit:
    case ActionType::power_to_ore:
    case ActionType::power_to_knowledge:
    case ActionType::power_to_qic:
    case ActionType::qic_to_ore:
    case ActionType::ore_to_credit:
    case ActionType::knowledge_to_credit: {
        auto& info = next.players[static_cast<std::size_t>(actor)];
        const bool brainstone = next.brainstone_selected;
        if (action.action_type == ActionType::power_to_credit) {
            const int cost = brainstone ? 3 : 1;
            spend_power(info, cost, brainstone);
            info.credits = std::min(30, info.credits + cost);
        } else if (action.action_type == ActionType::power_to_ore) {
            spend_power(info, 3, brainstone); info.ore = std::min(15, info.ore + 1);
        } else if (action.action_type == ActionType::power_to_knowledge) {
            spend_power(info, 4, brainstone); info.knowledge = std::min(15, info.knowledge + 1);
        } else if (action.action_type == ActionType::power_to_qic) {
            spend_power(info, 4, brainstone); gain_qic(info, 1);
        } else if (action.action_type == ActionType::qic_to_ore) {
            --info.qic; ++info.ore;
        } else if (action.action_type == ActionType::ore_to_credit) {
            --info.ore; ++info.credits;
        } else {
            --info.knowledge; ++info.credits;
        }
        next.brainstone_selected = false;
        return next;
    }
    case ActionType::brainstone:
        next.brainstone_selected = true;
        return next;
    case ActionType::terrans_gaia_credit:
    case ActionType::terrans_gaia_ore:
    case ActionType::terrans_gaia_knowledge:
    case ActionType::terrans_gaia_qic:
    case ActionType::terrans_gaia_finish: {
        auto& info = next.players[static_cast<std::size_t>(actor)];
        if (next.pending_gaia_conversion_player >= 0) {
            if (action.action_type == ActionType::terrans_gaia_finish) {
                info.bowl_two += info.gaia_power;
                info.gaia_power = 0;
                next.pending_gaia_conversion_player = -1;
                next.pending_gaia_conversion_power = 0;
                gaia_phase(next);
                return next;
            }
            const int cost = action.action_type == ActionType::terrans_gaia_credit ? 1
                : action.action_type == ActionType::terrans_gaia_ore ? 3 : 4;
            next.pending_gaia_conversion_power -= cost;
            if (action.action_type == ActionType::terrans_gaia_credit) info.credits = std::min(30, info.credits + 1);
            else if (action.action_type == ActionType::terrans_gaia_ore) info.ore = std::min(15, info.ore + 1);
            else if (action.action_type == ActionType::terrans_gaia_knowledge) info.knowledge = std::min(15, info.knowledge + 1);
            else gain_qic(info, 1);
            return next;
        }
        // These three semantic tuples are shared with Hadsch Hallas' PI credit actions.
        if (info.faction != 6 || !has_pi(next, actor)) throw std::invalid_argument("credit conversion is unavailable");
        if (action.action_type == ActionType::terrans_gaia_ore) { info.credits -= 3; ++info.ore; }
        else if (action.action_type == ActionType::terrans_gaia_knowledge) { info.credits -= 4; ++info.knowledge; }
        else if (action.action_type == ActionType::terrans_gaia_qic) { info.credits -= 4; gain_qic(info, 1); }
        else throw std::invalid_argument("invalid Hadsch Hallas conversion");
        return next;
    }
    case ActionType::nevlas_power_to_gaia:
    case ActionType::nevlas_credits:
    case ActionType::nevlas_credit_ore:
    case ActionType::nevlas_ore:
    case ActionType::nevlas_qic:
    case ActionType::nevlas_knowledge: {
        auto& info = next.players[static_cast<std::size_t>(actor)];
        if (action.action_type == ActionType::nevlas_power_to_gaia) {
            --info.bowl_three; ++info.gaia_power; ++info.knowledge;
            return next;
        }
        const int cost = action.action_type == ActionType::nevlas_credits ? 1
            : action.action_type == ActionType::nevlas_ore ? 3 : 2;
        spend_power(info, cost);
        if (action.action_type == ActionType::nevlas_credits) info.credits = std::min(30, info.credits + 2);
        else if (action.action_type == ActionType::nevlas_credit_ore) { info.credits = std::min(30, info.credits + 1); info.ore = std::min(15, info.ore + 1); }
        else if (action.action_type == ActionType::nevlas_ore) info.ore = std::min(15, info.ore + 2);
        else if (action.action_type == ActionType::nevlas_qic) gain_qic(info, 1);
        else info.knowledge = std::min(15, info.knowledge + 1);
        return next;
    }
    case ActionType::bal_taks_gaiaformer_qic: {
        auto& info = next.players[static_cast<std::size_t>(actor)];
        --info.gaiaformers; ++info.gaiaformers_in_gaia; gain_qic(info, 1);
        return next;
    }
    case ActionType::itars_burn_power: {
        auto& info = next.players[static_cast<std::size_t>(actor)];
        info.bowl_two -= 2; ++info.bowl_three; ++info.gaia_power;
        return next;
    }
    case ActionType::itars_gaia_technology: {
        auto& info = next.players[static_cast<std::size_t>(actor)];
        if (next.pending_itars_gaia_player != actor || info.faction != 13 ||
            !has_pi(next, actor) || info.gaia_power < 4 ||
            !has_tech_choice(next, actor))
            throw std::invalid_argument("Itars Gaia technology is unavailable");
        info.gaia_power -= 4;
        // The Gaia power decision remains pending while the technology
        // choice is resolved, exactly as in the Python state machine.
        next.pending_tech_player = actor;
        return next;
    }
    case ActionType::itars_gaia_finish: {
        auto& info = next.players[static_cast<std::size_t>(actor)];
        if (next.pending_itars_gaia_player != actor)
            throw std::invalid_argument("no Itars Gaia technology choice is pending");
        info.bowl_one += info.gaia_power;
        info.gaia_power = 0;
        next.pending_itars_gaia_player = -1;
        gaia_phase(next);
        return next;
    }
    case ActionType::skip_tech_research:
        next.pending_research_player = -1;
        next.pending_research_track = -1;
        next.pending_research_optional = false;
        advance_after_action(next);
        return next;
    case ActionType::gaia_project: {
        const int planet = action.arguments[0];
        const int range_bonus = next.pending_booster_range_player >= 0 ? 3 : 0;
        if (!can_start_gaia(next, actor, planet, range_bonus)) throw std::invalid_argument("cannot start Gaia project");
        auto& info = next.players[static_cast<std::size_t>(actor)];
        info.qic -= range_qic_cost(next, actor, planet, range_bonus);
        move_power_to_gaia(info, gaia_cost(info));
        --info.gaiaformers;
        next.gaiaformer_owner[static_cast<std::size_t>(planet)] = actor;
        next.pending_booster_range_player = -1;
        advance_after_action(next);
        return next;
    }
    case ActionType::upgrade_trading:
    case ActionType::upgrade_lab:
    case ActionType::upgrade_planetary_institute:
    case ActionType::upgrade_academy:
    case ActionType::upgrade_qic_academy: {
        const int planet = action.arguments[0];
        const auto index = static_cast<std::size_t>(planet);
        auto& info = next.players[static_cast<std::size_t>(actor)];
        const auto old = static_cast<Building>(next.buildings[index]);
        if (action.action_type == ActionType::upgrade_planetary_institute &&
            info.faction == 5 && old == Building::mine && has_pi(next, actor)) {
            int pi = -1;
            for (int candidate = 0; candidate < kMaxPlanets; ++candidate)
                if (next.owners[static_cast<std::size_t>(candidate)] == actor &&
                    next.buildings[static_cast<std::size_t>(candidate)] == static_cast<int>(Building::planetary_institute)) pi = candidate;
            next.buildings[static_cast<std::size_t>(pi)] = static_cast<int>(Building::mine);
            next.buildings[index] = static_cast<int>(Building::planetary_institute);
            info.used_ambas_swap_action = true;
            advance_after_action(next);
            return next;
        }
        if (action.action_type == ActionType::upgrade_trading && info.faction == 10 &&
            old == Building::research_lab && has_pi(next, actor)) {
            next.buildings[index] = static_cast<int>(Building::trading_station);
            info.used_firaks_downgrade_action = true;
            score(next, actor, 4);
            if (info.advanced_tech_tiles & (std::uint32_t{1} << 14)) info.vp += 3;
            next.pending_research_player = actor;
            next.pending_research_track = -1;
            next.pending_research_optional = false;
            trigger_passive_charge(next, actor, planet);
            advance_after_action(next);
            return next;
        }
        Building target = Building::empty;
        if (action.action_type == ActionType::upgrade_trading) {
            target = Building::trading_station;
            info.credits -= has_nearby_opponent(next, actor, planet) ? 3 : 6;
            info.ore -= 2;
            score(next, actor, 4);
            if (info.advanced_tech_tiles & (std::uint32_t{1} << 14)) info.vp += 3;
        } else if (action.action_type == ActionType::upgrade_lab) {
            target = Building::research_lab; info.credits -= 5; info.ore -= 3;
        } else if (action.action_type == ActionType::upgrade_planetary_institute) {
            target = Building::planetary_institute; info.credits -= 6; info.ore -= 4; score(next, actor, 6);
            if (info.faction == 3) {
                info.credits = std::min(30, info.credits + 2);
                info.ore = std::min(15, info.ore + 1);
                info.knowledge = std::min(15, info.knowledge + 1);
                ++info.federation_tokens; ++info.federation_keys; ++info.gleens_federation_tokens;
                score(next, actor, 3);
            }
        } else {
            target = Building::academy; info.credits -= 6; info.ore -= 6; score(next, actor, 6);
            if (action.action_type == ActionType::upgrade_academy) ++info.knowledge_academies;
            else ++info.qic_academies;
        }
        next.buildings[index] = static_cast<int>(target);
        if (target == Building::research_lab || target == Building::academy) next.pending_tech_player = actor;
        trigger_passive_charge(next, actor, planet);
        advance_after_action(next);
        return next;
    }
    case ActionType::tech_take: {
        const int space = action.arguments[0];
        auto& info = next.players[static_cast<std::size_t>(actor)];
        if (next.pending_advanced_tech >= 0) {
            const int standard = next.standard_tech_tiles[static_cast<std::size_t>(space)];
            const int advanced = next.pending_advanced_tech;
            info.covered_tech_tiles |= std::uint32_t{1} << static_cast<unsigned>(standard);
            info.advanced_tech_tiles |= std::uint32_t{1} << static_cast<unsigned>(advanced);
            --info.federation_keys;
            if (advanced == 3) info.vp += 2 * building_count(next, actor, Building::mine);
            else if (advanced == 4 || advanced == 5) {
                std::array<bool, kMaxSectors + 1> seen{};
                int sectors = 0;
                for (int planet = 0; planet < kMaxPlanets; ++planet) if (player_has_structure(next, actor, planet)) {
                    const int sector = next.planet_sectors[static_cast<std::size_t>(planet)];
                    if (sector >= 0 && sector <= kMaxSectors && !seen[static_cast<std::size_t>(sector)]) { seen[static_cast<std::size_t>(sector)] = true; ++sectors; }
                }
                if (advanced == 4) info.ore = std::min(15, info.ore + sectors); else info.vp += 2 * sectors;
            } else if (advanced == 6) {
                int gaia = 0;
                for (int planet = 0; planet < kMaxPlanets; ++planet)
                    gaia += next.owners[static_cast<std::size_t>(planet)] == actor && next.terrains[static_cast<std::size_t>(planet)] == static_cast<int>(Terrain::gaia);
                info.vp += 2 * gaia;
            } else if (advanced == 7) info.vp += 5 * info.federation_tokens;
            else if (advanced == 8) info.vp += 4 * building_count(next, actor, Building::trading_station);
            next.pending_advanced_tech = -1;
            if (has_research_choice(next, actor)) {
                next.pending_research_player = actor;
                next.pending_research_track = -1;
                next.pending_research_optional = true;
            }
            advance_after_action(next);
            return next;
        }
        if (space >= 9) {
            next.pending_tech_player = -1;
            next.pending_advanced_tech = next.advanced_tech_tiles[static_cast<std::size_t>(space - 9)];
            return next;
        }
        const int tile = next.standard_tech_tiles[static_cast<std::size_t>(space)];
        info.tech_tiles |= std::uint32_t{1} << static_cast<unsigned>(tile);
        if (tile == 0) { info.ore = std::min(15, info.ore + 1); gain_qic(info, 1); }
        else if (tile == 1) info.knowledge = std::min(15, info.knowledge + static_cast<int>(std::popcount(info.colonized_types)));
        else if (tile == 2) info.vp += 7;
        next.pending_tech_player = -1;
        if (space < kTrackCount && can_advance_research(next, actor, space)) {
            next.pending_research_player = actor; next.pending_research_track = space; next.pending_research_optional = true;
        } else if (space >= kTrackCount && has_research_choice(next, actor)) {
            next.pending_research_player = actor; next.pending_research_track = -1; next.pending_research_optional = true;
        }
        advance_after_action(next);
        return next;
    }
    case ActionType::power_action: {
        const int selected = action.arguments[0];
        auto& info = next.players[static_cast<std::size_t>(actor)];
        spend_power(info, power_action_cost(next, actor, selected), next.brainstone_selected);
        const int free_steps = power_terraform_steps(selected);
        if (selected == 0) info.knowledge = std::min(15, info.knowledge + 3);
        else if (free_steps) { next.pending_power_terraform_player = actor; next.pending_power_terraform_steps = free_steps; }
        else if (selected == 2) info.ore = std::min(15, info.ore + 2);
        else if (selected == 3) info.credits = std::min(30, info.credits + 7);
        else if (selected == 4) info.knowledge = std::min(15, info.knowledge + 2);
        else info.bowl_one += 2;
        next.used_power_actions |= 1 << selected;
        next.brainstone_selected = false;
        advance_after_action(next);
        return next;
    }
    case ActionType::qic_academy: {
        auto& info = next.players[static_cast<std::size_t>(actor)];
        if (info.faction == 9) info.credits = std::min(30, info.credits + 4); else gain_qic(info, 1);
        info.used_qic_academy_action = true;
        advance_after_action(next);
        return next;
    }
    case ActionType::standard_tech: {
        auto& info = next.players[static_cast<std::size_t>(actor)];
        charge_power(info, 4); info.used_standard_tech_action = true;
        advance_after_action(next);
        return next;
    }
    case ActionType::advanced_tech: {
        const int tile = action.arguments[0];
        auto& info = next.players[static_cast<std::size_t>(actor)];
        if (tile == 0) { info.credits = std::min(30, info.credits + 5); gain_qic(info, 1); }
        else if (tile == 1) info.ore = std::min(15, info.ore + 3);
        else info.knowledge = std::min(15, info.knowledge + 3);
        info.used_advanced_tech_actions |= 1 << tile;
        advance_after_action(next);
        return next;
    }
    case ActionType::qic_tech: {
        auto& info = next.players[static_cast<std::size_t>(actor)];
        info.qic -= 4; next.used_qic_actions |= 1; next.pending_tech_player = actor;
        return next;
    }
    case ActionType::qic_federation: {
        const int tile = action.arguments[0];
        auto& info = next.players[static_cast<std::size_t>(actor)];
        if (tile < 0 || tile > 6 || info.qic < 3)
            throw std::invalid_argument("QIC federation action is unavailable");
        info.qic -= 3;
        if (tile < 6) {
            if (info.federation_tile_counts[static_cast<std::size_t>(tile)] <= 0)
                throw std::invalid_argument("player does not own that federation tile");
            gain_federation_reward(info, tile);
        } else {
            if (info.gleens_federation_tokens <= 0)
                throw std::invalid_argument("player does not own the Gleens federation tile");
            gain_gleens_federation_reward(info);
        }
        next.used_qic_actions |= 2;
        advance_after_action(next);
        return next;
    }
    case ActionType::qic_planet_types: {
        auto& info = next.players[static_cast<std::size_t>(actor)];
        info.qic -= 2; info.vp += 3 + static_cast<int>(std::popcount(info.colonized_types));
        next.used_qic_actions |= 4;
        advance_after_action(next);
        return next;
    }
    case ActionType::booster_terraform:
        next.players[static_cast<std::size_t>(actor)].used_booster_action = true;
        next.pending_booster_terraform_player = actor;
        return next;
    case ActionType::booster_range:
        next.players[static_cast<std::size_t>(actor)].used_booster_action = true;
        next.pending_booster_range_player = actor;
        return next;
    case ActionType::bescods_research: {
        auto& info = next.players[static_cast<std::size_t>(actor)];
        const int track = action.arguments[0];
        info.used_bescods_research_action = true;
        const bool lost = track == 1 && info.tracks[1] == 4;
        advance_research(next, actor, track, true);
        if (lost) next.pending_lost_planet_player = actor;
        advance_after_action(next);
        return next;
    }
    case ActionType::ivits_space_station: {
        const int space = action.arguments[0];
        next.space_station_owner[static_cast<std::size_t>(space)] = actor;
        next.players[static_cast<std::size_t>(actor)].used_ivits_space_station_action = true;
        mark_adjacent_federated(next, actor, 2 * kMaxPlanets + space);
        advance_after_action(next);
        return next;
    }
    case ActionType::lost_planet: {
        const int space = action.arguments[0];
        const auto spaces = board_spaces(next);
        const int planet = kMaxPlanets - 1;
        auto& info = next.players[static_cast<std::size_t>(actor)];
        const int q = spaces[static_cast<std::size_t>(space)][0];
        const int r = spaces[static_cast<std::size_t>(space)][1];
        info.qic -= coordinate_range_qic_cost(next, actor, q, r);
        score(next, actor, 2);
        if (info.advanced_tech_tiles & (std::uint32_t{1} << 13)) info.vp += 3;
        if (info.faction == 8 && has_pi(next, actor) && !(info.colonized_types & (std::uint32_t{1} << static_cast<unsigned>(Terrain::lost))))
            info.knowledge = std::min(15, info.knowledge + 3);
        info.colonized_types |= std::uint32_t{1} << static_cast<unsigned>(Terrain::lost);
        const auto index = static_cast<std::size_t>(planet);
        next.active_planets[index] = true; next.planet_q[index] = q; next.planet_r[index] = r;
        next.planet_source_q[index] = q; next.planet_source_r[index] = r;
        next.owners[index] = actor; next.buildings[index] = static_cast<int>(Building::mine);
        next.terrains[index] = static_cast<int>(Terrain::lost); next.pending_lost_planet_player = -1;
        int sector_id = -1;
        for (int position = 0; position < next.sector_count; ++position) {
            const int local_q = q - next.sector_centers[static_cast<std::size_t>(position)][0];
            const int local_r = r - next.sector_centers[static_cast<std::size_t>(position)][1];
            if (std::max({std::abs(local_q), std::abs(local_r), std::abs(local_q + local_r)}) <= 2) {
                sector_id = next.sector_tiles[static_cast<std::size_t>(position)] + 1; break;
            }
        }
        next.planet_sectors[index] = sector_id;
        mark_adjacent_federated(next, actor, planet);
        trigger_passive_charge(next, actor, planet);
        advance_after_action(next);
        return next;
    }
    case ActionType::federation: {
        const int reward = action.arguments[0];
        FederationPlan plan;
        if (!federation_plan_details(next, actor, plan)) throw std::invalid_argument("no legal federation plan");
        auto& info = next.players[static_cast<std::size_t>(actor)];
        if (info.faction == 7) info.qic -= static_cast<int>(plan.satellites.size());
        else discard_power(info, static_cast<int>(plan.satellites.size()));
        gain_federation_reward(info, reward);
        ++info.federation_tile_counts[static_cast<std::size_t>(reward)];
        ++info.federation_tokens;
        info.federation_keys += reward != 5;
        ++info.board_federations;
        info.satellites += static_cast<int>(plan.satellites.size());
        score(next, actor, 3);
        for (const int location : plan.locations) {
            if (location < kMaxPlanets) next.federated[static_cast<std::size_t>(location)] = true;
            else if (location < 2 * kMaxPlanets) next.coexisting_mine_federated[static_cast<std::size_t>(location - kMaxPlanets)] = true;
            else next.space_station_federated[static_cast<std::size_t>(location - 2 * kMaxPlanets)] = true;
        }
        for (const int space : plan.satellites) next.satellite_owners[static_cast<std::size_t>(space)] |= 1 << actor;
        --next.federation_tile_supply[static_cast<std::size_t>(reward)];
        advance_after_action(next);
        return next;
    }
    case ActionType::place_starting_structure: {
        const int planet = action.arguments[0];
        if (!is_home_planet(next, actor, planet)) throw std::invalid_argument("starting structure must be on an available home planet");
        const auto& f = kFactions[static_cast<std::size_t>(next.players[static_cast<std::size_t>(actor)].faction)];
        next.owners[static_cast<std::size_t>(planet)] = actor;
        next.buildings[static_cast<std::size_t>(planet)] = static_cast<int>(next.starting_planet_count[static_cast<std::size_t>(actor)] == 0 && f.starts_with_pi ? Building::planetary_institute : Building::mine);
        auto& count = next.starting_planet_count[static_cast<std::size_t>(actor)];
        if (count < 3) next.starting_planets[static_cast<std::size_t>(actor)][static_cast<std::size_t>(count)] = planet;
        ++count;
        next.players[static_cast<std::size_t>(actor)].colonized_types |= 1u << static_cast<unsigned>(next.terrains[static_cast<std::size_t>(planet)]);
        ++next.placement_step;
        next.player_to_move = next.placement_step < next.placement_order_length ? next.placement_order[static_cast<std::size_t>(next.placement_step)] : next.booster_selection_order[0];
        return next;
    }
    case ActionType::pass_booster: {
        if (is_booster_selection()) {
            next.booster_owner[static_cast<std::size_t>(action.arguments[0])] = actor;
            ++next.booster_selection_step;
            if (next.booster_selection_step >= next.player_count) {
                next.round_number = 1;
                next.player_to_move = next.first_player;
                grant_income(next);
            } else next.player_to_move = next.booster_selection_order[static_cast<std::size_t>(next.booster_selection_step)];
            return next;
        }
        if (next.round_number >= kMaxRounds) throw std::invalid_argument("final-round pass cannot select a booster");
        const int previous = player_booster(next, actor);
        next.players[static_cast<std::size_t>(actor)].vp += booster_pass_points(next, actor, previous);
        if (previous >= 0)
            next.booster_owner[static_cast<std::size_t>(previous)] = -1;
        next.booster_owner[static_cast<std::size_t>(action.arguments[0])] = actor;
        next.players[static_cast<std::size_t>(actor)].passed = true;
        if (next.next_first_player < 0) next.next_first_player = actor;
        advance_after_action(next);
        return next;
    }
    case ActionType::build_mine: {
        const int planet = action.arguments[0];
        const int free_steps = next.pending_power_terraform_player >= 0
            ? next.pending_power_terraform_steps
            : next.pending_booster_terraform_player >= 0 ? 1 : 0;
        const int range_bonus = next.pending_booster_range_player >= 0 ? 3 : 0;
        if (!can_build_mine(next, actor, planet, free_steps, range_bonus)) throw std::invalid_argument("cannot build mine on target planet");
        const auto index = static_cast<std::size_t>(planet);
        auto& p = next.players[static_cast<std::size_t>(actor)];
        const auto cost = mine_cost(next, actor, planet, free_steps, range_bonus);
        p.credits -= cost.credits; p.ore -= cost.ore; p.qic -= cost.qic;
        const bool coexisting = can_lantids_coexist(next, actor, planet);
        const int terrain = next.terrains[index];
        const int steps = coexisting || terrain == static_cast<int>(Terrain::gaia)
            ? 0 : terrain_steps(kFactions[static_cast<std::size_t>(p.faction)].home,
                                static_cast<Terrain>(terrain));
        score(next, actor, 2);
        if (steps) score(next, actor, 0, steps);
        if (terrain == static_cast<int>(Terrain::gaia) && !coexisting) {
            score(next, actor, 5);
            if (has_active_standard_tech(next, p, 3)) p.vp += 3;
            if (p.faction == 3) p.vp += 2;
        }
        if (p.advanced_tech_tiles & (std::uint32_t{1} << 13)) p.vp += 3;
        const bool geodens = p.faction == 8 && has_pi(next, actor) &&
            terrain != static_cast<int>(Terrain::transdim) &&
            !(p.colonized_types & (std::uint32_t{1} << static_cast<unsigned>(terrain)));
        if (!coexisting) p.colonized_types |= 1u << static_cast<unsigned>(terrain);
        if (geodens) p.knowledge = std::min(15, p.knowledge + 3);
        if (coexisting && has_pi(next, actor)) p.knowledge = std::min(15, p.knowledge + 2);
        if (coexisting) next.coexisting_mine_owner[index] = actor;
        else { next.owners[index] = actor; next.buildings[index] = static_cast<int>(Building::mine); }
        if (next.gaiaformer_owner[index] == actor) { next.gaiaformer_owner[index] = -1; ++p.gaiaformers; }
        next.pending_power_terraform_player = -1;
        next.pending_power_terraform_steps = 0;
        next.pending_booster_terraform_player = -1;
        next.pending_booster_range_player = -1;
        mark_adjacent_federated(next, actor, coexisting ? kMaxPlanets + planet : planet);
        trigger_passive_charge(next, actor, planet);
        advance_after_action(next);
        return next;
    }
    case ActionType::research: {
        auto& p = next.players[static_cast<std::size_t>(actor)];
        const bool free = next.pending_research_player >= 0;
        if (!free && p.knowledge < 4) throw std::invalid_argument("research requires four knowledge");
        if (!free) p.knowledge -= 4;
        const int track = action.arguments[0];
        const bool lost = track == 1 && p.tracks[1] == 4;
        advance_research(next, actor, action.arguments[0], true);
        if (free) {
            next.pending_research_player = -1;
            next.pending_research_track = -1;
            next.pending_research_optional = false;
        }
        if (lost) next.pending_lost_planet_player = actor;
        advance_after_action(next);
        return next;
    }
    case ActionType::pass_final: {
        // Final-round passing still awards the current booster’s pass VP;
        // unlike a normal pass it does not take a replacement booster.
        const int current_booster = player_booster(next, actor);
        next.players[static_cast<std::size_t>(actor)].vp += booster_pass_points(next, actor, current_booster);
        next.players[static_cast<std::size_t>(actor)].passed = true;
        if (next.next_first_player < 0) next.next_first_player = actor;
        advance_after_action(next);
        return next;
    }
    default:
        throw std::invalid_argument("C++ baseline does not yet implement this action type");
    }
}

std::array<double, kMaxPlayers> GaiaState::final_scores() const {
    std::array<double, kMaxPlayers> scores{};
    auto scoring_metric = [this](int player, int tile) {
        if (tile == 0) {
            int total = 0;
            for (int planet = 0; planet < kMaxPlanets; ++planet) {
                const auto index = static_cast<std::size_t>(planet);
                total += owners[index] == player && federated[index];
                total += coexisting_mine_owner[index] == player && coexisting_mine_federated[index];
            }
            return total;
        }
        if (tile == 1) {
            int total = 0;
            for (int planet = 0; planet < kMaxPlanets; ++planet) {
                const auto index = static_cast<std::size_t>(planet);
                total += owners[index] == player;
                total += coexisting_mine_owner[index] == player;
            }
            return total;
        }
        if (tile == 2) return static_cast<int>(std::popcount(players[static_cast<std::size_t>(player)].colonized_types));
        if (tile == 3) {
            int total = 0;
            for (int planet = 0; planet < kMaxPlanets; ++planet)
                total += owners[static_cast<std::size_t>(planet)] == player && terrains[static_cast<std::size_t>(planet)] == static_cast<int>(Terrain::gaia);
            return total;
        }
        if (tile == 4) {
            std::array<bool, kMaxSectors> seen{};
            int total = 0;
            for (int planet = 0; planet < kMaxPlanets; ++planet) {
                const auto index = static_cast<std::size_t>(planet);
                if ((owners[index] == player || coexisting_mine_owner[index] == player) && planet_sectors[index] >= 0) {
                    const int sector = planet_sectors[index];
                    if (sector < kMaxSectors && !seen[static_cast<std::size_t>(sector)]) {
                        seen[static_cast<std::size_t>(sector)] = true;
                        ++total;
                    }
                }
            }
            return total;
        }
        int stations = 0;
        for (const int owner : space_station_owner) stations += owner == player;
        return players[static_cast<std::size_t>(player)].satellites + stations;
    };

    for (int player = 0; player < player_count; ++player) {
        const auto& p = players[static_cast<std::size_t>(player)];
        int research = 0;
        for (const int level : p.tracks) research += std::max(0, level - 2) * 4;
        int ordinary_power = p.bowl_three - (p.brainstone_bowl == 3 ? 1 : 0);
        if (p.faction == 12 && has_pi(*this, player)) ordinary_power *= 2;
        const int brainstone_power = p.brainstone_bowl == 3 ? 3 : 0;
        const int resources = (p.credits + p.ore + p.knowledge + p.qic + ordinary_power + brainstone_power) / 3;
        scores[static_cast<std::size_t>(player)] = static_cast<double>(p.vp + research + resources);
    }

    constexpr std::array<double, 4> awards{{18.0, 12.0, 6.0, 0.0}};
    constexpr std::array<int, 6> neutral{{10, 11, 5, 4, 6, 8}};
    for (const int tile : final_scoring_tiles) {
        std::vector<std::pair<int, int>> ranked;
        for (int player = 0; player < player_count; ++player) ranked.emplace_back(scoring_metric(player, tile), player);
        if (player_count == 2) ranked.emplace_back(neutral[static_cast<std::size_t>(tile)], player_count);
        std::stable_sort(ranked.begin(), ranked.end(), [](const auto& left, const auto& right) { return left.first > right.first; });
        for (std::size_t begin = 0; begin < ranked.size();) {
            std::size_t end = begin + 1;
            while (end < ranked.size() && ranked[end].first == ranked[begin].first) ++end;
            double shared = 0.0;
            for (std::size_t place = begin; place < end; ++place) shared += awards[place];
            shared /= static_cast<double>(end - begin);
            for (std::size_t index = begin; index < end; ++index)
                if (ranked[index].second < player_count) scores[static_cast<std::size_t>(ranked[index].second)] += shared;
            begin = end;
        }
    }
    return scores;
}

std::string GaiaState::debug_federation_plan() const {
    FederationPlan plan;
    if (!federation_plan_details(*this, player_to_move, plan)) return "none";
    std::ostringstream out;
    out << "locations=";
    for (std::size_t i = 0; i < plan.locations.size(); ++i) {
        if (i) out << ',';
        out << plan.locations[i];
    }
    out << ";satellites=";
    for (std::size_t i = 0; i < plan.satellites.size(); ++i) {
        if (i) out << ',';
        out << plan.satellites[i];
    }
    out << ";clusters=";
    const auto clusters = location_clusters(*this, structure_locations(*this, player_to_move, false));
    for (std::size_t ci = 0; ci < clusters.size(); ++ci) {
        if (ci) out << '|';
        for (std::size_t li = 0; li < clusters[ci].size(); ++li) {
            if (li) out << ',';
            out << clusters[ci][li];
        }
    }
    out << ";coords=";
    const auto spaces = board_spaces(*this);
    for (std::size_t i = 0; i < plan.satellites.size(); ++i) {
        if (i) out << '|';
        const auto coordinate = spaces[static_cast<std::size_t>(plan.satellites[i])];
        out << coordinate[0] << ':' << coordinate[1];
    }
    return out.str();
}

} // namespace gaiazero
