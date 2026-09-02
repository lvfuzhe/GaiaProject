#include "gaiazero/gaia_state.hpp"

#include "gaiazero/sha256.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdlib>
#include <cstdint>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>

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

std::uint64_t stream_seed(std::int64_t root, std::string_view name) {
    const auto digest = sha256_hex(std::to_string(root) + "|setup-seed-stream-v1|" + std::string(name));
    // Python: int.from_bytes(digest[:8], byteorder="little", signed=False).
    std::uint64_t result = 0;
    for (int i = 0; i < 8; ++i) {
        result |= static_cast<std::uint64_t>(std::stoul(digest.substr(static_cast<std::size_t>(2 * i), 2), nullptr, 16)) << (8 * i);
    }
    return result;
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

constexpr std::array<std::array<int, 2>, 7> kSectorCenters2p{{
    {0, 0}, {3, -5}, {5, -2}, {2, 3}, {-3, 5}, {-5, 2}, {-2, -3},
}};
constexpr std::array<std::array<int, 2>, 10> kSectorCenters34p{{
    {-4, -2}, {1, -4}, {6, -6}, {-7, 3}, {-2, 1},
    {3, -1}, {8, -3}, {-5, 6}, {0, 4}, {5, 2},
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

int range_qic_cost(const GaiaState& state, int player, int destination) {
    if (state.gaiaformer_owner[static_cast<std::size_t>(destination)] == player) return 0;
    int distance = kMaxPlanets;
    for (int source = 0; source < kMaxPlanets; ++source) {
        const auto index = static_cast<std::size_t>(source);
        if (state.owners[index] == player || state.coexisting_mine_owner[index] == player) {
            distance = std::min(distance, hex_distance(state.planet_q[index], state.planet_r[index],
                                                       state.planet_q[static_cast<std::size_t>(destination)],
                                                       state.planet_r[static_cast<std::size_t>(destination)]));
        }
    }
    if (distance == kMaxPlanets) return kMaxPlanets;
    constexpr std::array<int, 6> ranges{{1, 1, 2, 2, 3, 4}};
    const int reach = ranges[static_cast<std::size_t>(std::clamp(state.players[static_cast<std::size_t>(player)].tracks[1], 0, 5))];
    return (std::max(0, distance - reach) + 1) / 2;
}

struct ResourceCost { int credits; int ore; int qic; };

ResourceCost mine_cost(const GaiaState& state, int player, int planet) {
    const auto index = static_cast<std::size_t>(planet);
    const auto& p = state.players[static_cast<std::size_t>(player)];
    const bool coexisting = can_lantids_coexist(state, player, planet);
    const int range_qic = range_qic_cost(state, player, planet);
    if (coexisting) return {2, 1, range_qic};
    const auto terrain = static_cast<Terrain>(state.terrains[index]);
    if (terrain == Terrain::gaia) {
        const int gaia_qic = state.gaiaformer_owner[index] == player ? 0 : 1;
        return p.faction == 3 && gaia_qic != 0 ? ResourceCost{2, 2, range_qic}
                                               : ResourceCost{2, 1, gaia_qic + range_qic};
    }
    constexpr std::array<int, 6> ore_per_step{{3, 3, 2, 1, 1, 1}};
    const int steps = terrain_steps(kFactions[static_cast<std::size_t>(p.faction)].home, terrain);
    return {2, 1 + steps * ore_per_step[static_cast<std::size_t>(std::clamp(p.tracks[0], 0, 5))], range_qic};
}

bool can_build_mine(const GaiaState& state, int player, int planet) {
    if (planet < 0 || planet >= kMaxPlanets) return false;
    const auto index = static_cast<std::size_t>(planet);
    const bool coexisting = can_lantids_coexist(state, player, planet);
    if (!state.active_planets[index] || (state.owners[index] >= 0 && !coexisting) ||
        state.terrains[index] == static_cast<int>(Terrain::transdim) ||
        building_count(state, player, Building::mine) >= 8 ||
        (state.gaiaformer_owner[index] >= 0 && state.gaiaformer_owner[index] != player)) return false;
    const auto cost = mine_cost(state, player, planet);
    const auto& p = state.players[static_cast<std::size_t>(player)];
    return p.credits >= cost.credits && p.ore >= cost.ore && p.qic >= cost.qic;
}

void charge_power(PlayerState& p, int amount) {
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
    }
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
    if (score_round && state.round_number >= 1 && state.round_number <= kMaxRounds && state.round_scoring_tiles[static_cast<std::size_t>(state.round_number - 1)] == 1) p.vp += 2;
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
        int knowledge = 1 + f.income_knowledge + science[static_cast<std::size_t>(std::clamp(p.tracks[5], 0, 5))] + booster_income[2];
        int qic = f.income_qic + booster_income[3];
        int power_tokens = f.income_power + booster_income[4];
        int power_charge = institutes * 4 + economy_income[2] + booster_income[5];
        if (p.faction == 11) {
            knowledge += trading;
            for (int i = 0; i < std::min(labs, 3); ++i) credits += bescods_lab_credits[static_cast<std::size_t>(i)];
        } else {
            for (int i = 0; i < std::min(trading, 4); ++i) credits += trading_credits[static_cast<std::size_t>(i)];
            knowledge += labs;
        }
        knowledge += p.knowledge_academies * (p.faction == 13 ? 3 : 2);
        if (p.faction == 12) power_charge += labs * 2;
        if (institutes) {
            if (p.faction == 2) qic += institutes;
            else if (p.faction == 3) ore += institutes;
            else if (p.faction == 5 || p.faction == 11) power_tokens += 2 * institutes;
            else if (p.faction != 1) power_tokens += institutes;
        }
        if (p.faction == 3 && p.qic_academies == 0) { ore += qic; qic = 0; }
        p.credits = std::min(30, p.credits + credits);
        p.ore = std::min(15, p.ore + ore);
        p.knowledge = std::min(15, p.knowledge + knowledge);
        p.qic += qic;
        p.bowl_one += power_tokens;
        charge_power(p, power_charge);
    }
}

int booster_pass_points(const GaiaState& state, int player, int booster) {
    if (booster == 5) return building_count(state, player, Building::mine);
    if (booster == 6) return 2 * building_count(state, player, Building::trading_station);
    if (booster == 7) return 3 * building_count(state, player, Building::research_lab);
    if (booster == 8) return 4 * (building_count(state, player, Building::planetary_institute) + building_count(state, player, Building::academy));
    if (booster == 9) {
        int gaia = 0;
        for (int planet = 0; planet < kMaxPlanets; ++planet)
            gaia += state.owners[static_cast<std::size_t>(planet)] == player && state.terrains[static_cast<std::size_t>(planet)] == static_cast<int>(Terrain::gaia);
        return gaia;
    }
    return 0;
}

void advance_after_action(GaiaState& state) {
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
}

} // namespace

GaiaState GaiaState::initial(std::int32_t players_count, std::int64_t seed) {
    if (players_count < 2 || players_count > kMaxPlayers) throw std::invalid_argument("GaiaState supports two to four players");
    if (seed < 0) throw std::invalid_argument("setup seed must be non-negative");
    GaiaState state;
    state.player_count = players_count;
    state.setup_seed = seed;
    state.first_player = static_cast<int>((seed % players_count + players_count) % players_count);
    state.setup_seed_streams = {{"map", static_cast<std::uint64_t>(seed)},
                                {"factions", stream_seed(seed, "factions")},
                                {"boosters", stream_seed(seed, "boosters")},
                                {"round_scoring", stream_seed(seed, "round_scoring")},
                                {"final_scoring", stream_seed(seed, "final_scoring")},
                                {"standard_technology", stream_seed(seed, "standard_technology")},
                                {"advanced_technology", stream_seed(seed, "advanced_technology")},
                                {"terraforming_federation", stream_seed(seed, "terraforming_federation")}};
    state.owners.fill(-1);
    state.buildings.fill(static_cast<int>(Building::empty));
    state.terrains.fill(static_cast<int>(Terrain::terra));
    state.planet_sectors.fill(-1);
    state.planet_source_ids.fill(-1);
    state.gaiaformer_owner.fill(-1);
    state.coexisting_mine_owner.fill(-1);
    state.satellite_owners.fill(0);
    state.space_station_owner.fill(-1);
    state.sector_tiles.fill(-1);
    state.booster_owner.fill(-2);

    auto shuffled_prefix = [](auto& values, std::uint64_t seed_value) {
        std::mt19937_64 rng(seed_value);
        std::shuffle(values.begin(), values.end(), rng);
    };
    std::array<int, kBoosterCount> boosters{};
    for (int i = 0; i < kBoosterCount; ++i) boosters[static_cast<std::size_t>(i)] = i;
    shuffled_prefix(boosters, stream_seed(seed, "boosters"));
    for (int i = 0; i < players_count + 3; ++i) state.booster_owner[static_cast<std::size_t>(boosters[static_cast<std::size_t>(i)])] = -1;

    std::array<int, 10> round_tiles{};
    for (int i = 0; i < 10; ++i) round_tiles[static_cast<std::size_t>(i)] = i;
    shuffled_prefix(round_tiles, stream_seed(seed, "round_scoring"));
    std::copy_n(round_tiles.begin(), state.round_scoring_tiles.size(), state.round_scoring_tiles.begin());
    std::array<int, 6> final_tiles{};
    for (int i = 0; i < 6; ++i) final_tiles[static_cast<std::size_t>(i)] = i;
    shuffled_prefix(final_tiles, stream_seed(seed, "final_scoring"));
    std::copy_n(final_tiles.begin(), state.final_scoring_tiles.size(), state.final_scoring_tiles.begin());
    for (int i = 0; i < 9; ++i) state.standard_tech_tiles[static_cast<std::size_t>(i)] = i;
    shuffled_prefix(state.standard_tech_tiles, stream_seed(seed, "standard_technology"));
    std::array<int, 15> advanced_pool{};
    for (int i = 0; i < 15; ++i) advanced_pool[static_cast<std::size_t>(i)] = i;
    shuffled_prefix(advanced_pool, stream_seed(seed, "advanced_technology"));
    std::copy_n(advanced_pool.begin(), state.advanced_tech_tiles.size(), state.advanced_tech_tiles.begin());
    state.terraforming_federation_tile = static_cast<int>(stream_seed(seed, "terraforming_federation") % 6u);
    for (int i = 0; i < 6; ++i) state.federation_tile_supply[static_cast<std::size_t>(i)] = i == state.terraforming_federation_tile ? 2 : 3;

    std::array<int, 7> board_pool{};
    for (int i = 0; i < 7; ++i) board_pool[static_cast<std::size_t>(i)] = i;
    std::mt19937_64 faction_rng(stream_seed(seed, "factions"));
    std::shuffle(board_pool.begin(), board_pool.end(), faction_rng);
    std::array<int, kMaxPlayers> faction_pool{};
    std::uniform_int_distribution<int> side(0, 1);
    for (int player = 0; player < players_count; ++player)
        faction_pool[static_cast<std::size_t>(player)] = 2 * board_pool[static_cast<std::size_t>(player)] + side(faction_rng);
    state.placement_order_length = 0;
    // Match the Python/BGA convention: regular factions snake by layer; Ivits is appended last.
    std::array<bool, kMaxPlayers> places_last{};
    for (int player = 0; player < players_count; ++player) {
        places_last[static_cast<std::size_t>(player)] = kFactions[static_cast<std::size_t>(faction_pool[static_cast<std::size_t>(player)])].starts_with_pi;
    }
    std::array<int, kMaxPlayers> forward{};
    for (int i = 0; i < players_count; ++i)
        forward[static_cast<std::size_t>(i)] = (state.first_player + i) % players_count;
    for (int layer = 0; layer < 3; ++layer) {
        for (int offset = 0; offset < players_count; ++offset) {
            const int index = layer % 2 == 0 ? offset : players_count - 1 - offset;
            const int player = forward[static_cast<std::size_t>(index)];
            const auto& f = kFactions[static_cast<std::size_t>(faction_pool[static_cast<std::size_t>(player)])];
            if (!places_last[static_cast<std::size_t>(player)] && f.starting_structures > layer)
                state.placement_order[static_cast<std::size_t>(state.placement_order_length++)] = player;
        }
    }
    for (int player = 0; player < players_count; ++player) {
        if (places_last[static_cast<std::size_t>(player)])
            state.placement_order[static_cast<std::size_t>(state.placement_order_length++)] = player;
    }
    for (int player = 0; player < players_count; ++player) {
        auto& p = state.players[static_cast<std::size_t>(player)];
        const auto& f = kFactions[static_cast<std::size_t>(faction_pool[static_cast<std::size_t>(player)])];
        p.faction = faction_pool[static_cast<std::size_t>(player)];
        p.credits = f.credits; p.ore = f.ore; p.knowledge = f.knowledge; p.qic = f.qic;
        p.bowl_one = f.power[0]; p.bowl_two = f.power[1]; p.bowl_three = f.power[2];
        if (f.brainstone) { p.brainstone_bowl = 1; ++p.bowl_one; }
        if (f.start_track >= 0) advance_research(state, player, f.start_track, false);
        state.booster_selection_order[static_cast<std::size_t>(player)] =
            (state.first_player - player - 1 + players_count * 2) % players_count;
    }
    state.sector_count = players_count == 2 ? 7 : 10;
    std::array<int, kMaxSectors> sector_pool{};
    for (int i = 0; i < kMaxSectors; ++i) sector_pool[static_cast<std::size_t>(i)] = i;
    std::mt19937_64 map_rng(static_cast<std::uint64_t>(seed));
    std::shuffle(sector_pool.begin(), sector_pool.end(), map_rng);
    std::uniform_int_distribution<int> rotation(0, 5);
    for (int i = 0; i < state.sector_count; ++i) {
        state.sector_tiles[static_cast<std::size_t>(i)] = sector_pool[static_cast<std::size_t>(i)];
        state.sector_rotations[static_cast<std::size_t>(i)] = rotation(map_rng);
        state.sector_centers[static_cast<std::size_t>(i)] = players_count == 2
            ? kSectorCenters2p[static_cast<std::size_t>(i)]
            : kSectorCenters34p[static_cast<std::size_t>(i)];
    }
    state.planet_source_catalog_length = 0;
    // A stable coordinate scaffold. The Python/BGA map generator can replace these arrays at the adapter boundary.
    for (int planet = 0; planet < kMaxPlanets; ++planet) {
        state.active_planets[static_cast<std::size_t>(planet)] = planet < 42;
        state.planet_q[static_cast<std::size_t>(planet)] = (planet % 10) * 2 + (planet / 10 % 2);
        state.planet_r[static_cast<std::size_t>(planet)] = planet / 10;
        state.planet_source_q[static_cast<std::size_t>(planet)] = state.planet_q[static_cast<std::size_t>(planet)];
        state.planet_source_r[static_cast<std::size_t>(planet)] = state.planet_r[static_cast<std::size_t>(planet)];
        state.planet_source_ids[static_cast<std::size_t>(planet)] = planet < 70 ? planet : -1;
        state.planet_sectors[static_cast<std::size_t>(planet)] = planet < 42 ? planet / 7 : -1;
        state.terrains[static_cast<std::size_t>(planet)] = planet < 70 ? planet % 8 : static_cast<int>(Terrain::lost);
        if (planet < 42) {
            state.planet_source_catalog[static_cast<std::size_t>(state.planet_source_catalog_length++)] =
                {planet, state.planet_source_q[static_cast<std::size_t>(planet)], state.planet_source_r[static_cast<std::size_t>(planet)], state.terrains[static_cast<std::size_t>(planet)], state.planet_sectors[static_cast<std::size_t>(planet)]};
        }
    }
    state.player_to_move = state.placement_order_length > 0 ? state.placement_order[0] : state.first_player;
    // Hash the materialized setup, not just its seed. A setup algorithm change
    // must therefore produce a different audit key even for the same root seed.
    state.setup_hash = sha256_hex(state.canonical_json());
    return state;
}

bool GaiaState::is_terminal() const noexcept { return round_number > kMaxRounds; }
bool GaiaState::is_starting_placement() const noexcept { return round_number == 0 && placement_step < placement_order_length; }
bool GaiaState::is_booster_selection() const noexcept {
    return round_number == 0 && placement_step >= placement_order_length && booster_selection_step < player_count;
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
    append_key(out, "terraforming_federation_tile"); out << terraforming_federation_tile << ',';
    append_key(out, "terrains"); append_array(out, terrains); out << ',';
    append_key(out, "used_power_actions"); out << used_power_actions << ',';
    append_key(out, "used_qic_actions"); out << used_qic_actions << '}';
    return out.str();
}

std::string GaiaState::state_hash() const {
    return state_hash_from_canonical_json(canonical_json());
}

std::vector<ActionTuple> GaiaState::legal_action_tuples() const {
    if (is_terminal()) return {};
    if (is_starting_placement()) {
        const int player = placement_order[static_cast<std::size_t>(placement_step)];
        std::vector<ActionTuple> actions;
        for (int planet = 0; planet < kMaxPlanets; ++planet) if (is_home_planet(*this, player, planet)) {
            actions.push_back(ActionTuple::create(ActionType::place_starting_structure, {planet}));
        }
        return actions;
    }
    if (is_booster_selection()) {
        std::vector<ActionTuple> actions;
        for (int booster = 0; booster < kBoosterCount; ++booster) if (booster_owner[static_cast<std::size_t>(booster)] < 0)
            actions.push_back(ActionTuple::create(ActionType::pass_booster, {booster}));
        return actions;
    }
    const int player = current_player(*this);
    std::vector<ActionTuple> actions;
    const auto& p = players[static_cast<std::size_t>(player)];
    for (int planet = 0; planet < kMaxPlanets; ++planet) {
        if (can_build_mine(*this, player, planet))
            actions.push_back(ActionTuple::create(ActionType::build_mine, {planet}));
    }
    if (p.knowledge >= 4) {
        for (int track = 0; track < kTrackCount; ++track)
            if (can_advance_research(*this, player, track))
                actions.push_back(ActionTuple::create(ActionType::research, {track}));
    }
    if (round_number == kMaxRounds) {
        actions.push_back(ActionTuple::create(ActionType::pass_final, {}));
    } else {
        for (int booster = 0; booster < kBoosterCount; ++booster)
            if (booster_owner[static_cast<std::size_t>(booster)] == -1)
                actions.push_back(ActionTuple::create(ActionType::pass_booster, {booster}));
    }
    return actions;
}

GaiaState GaiaState::apply(const ActionTuple& action) const {
    if (!action.valid()) throw std::invalid_argument("invalid ActionTuple");
    const auto legal = legal_action_tuples();
    if (std::find(legal.begin(), legal.end(), action) == legal.end()) throw std::invalid_argument("illegal action tuple");
    GaiaState next = *this;
    const int actor = player_to_move;
    switch (action.action_type) {
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
        if (!can_build_mine(next, actor, planet)) throw std::invalid_argument("cannot build mine on target planet");
        const auto index = static_cast<std::size_t>(planet);
        auto& p = next.players[static_cast<std::size_t>(actor)];
        const auto cost = mine_cost(next, actor, planet);
        p.credits -= cost.credits; p.ore -= cost.ore; p.qic -= cost.qic;
        const bool coexisting = can_lantids_coexist(next, actor, planet);
        if (!coexisting) p.colonized_types |= 1u << static_cast<unsigned>(next.terrains[index]);
        if (coexisting) next.coexisting_mine_owner[index] = actor;
        else { next.owners[index] = actor; next.buildings[index] = static_cast<int>(Building::mine); }
        if (next.gaiaformer_owner[index] == actor) { next.gaiaformer_owner[index] = -1; ++p.gaiaformers; }
        if (p.faction == 3 && next.terrains[index] == static_cast<int>(Terrain::gaia)) p.vp += 2;
        score_mine(next, actor, next.terrains[index]);
        advance_after_action(next);
        return next;
    }
    case ActionType::research: {
        auto& p = next.players[static_cast<std::size_t>(actor)];
        if (p.knowledge < 4) throw std::invalid_argument("research requires four knowledge");
        p.knowledge -= 4;
        advance_research(next, actor, action.arguments[0], true);
        advance_after_action(next);
        return next;
    }
    case ActionType::pass_final:
        next.players[static_cast<std::size_t>(actor)].passed = true;
        if (next.next_first_player < 0) next.next_first_player = actor;
        advance_after_action(next);
        return next;
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

} // namespace gaiazero
