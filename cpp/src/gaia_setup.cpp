#include "gaiazero/gaia_setup.hpp"

#include "gaiazero/numpy_random.hpp"
#include "gaiazero/sha256.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <unordered_set>

namespace gaiazero {
namespace {

using Planet = std::array<int, 3>;
using Sector = std::vector<Planet>;

const std::array<Sector, 10> kSolidSectors{{
    {{{1, -2, 7}}, {{0, -1, 0}}, {{2, -1, 4}}, {{2, 0, 3}}, {{-1, 1, 2}}, {{-1, 2, 1}}},
    {{{1, -2, 1}}, {{0, -1, 6}}, {{2, -1, 7}}, {{-2, 0, 5}}, {{-2, 1, 4}}, {{0, 1, 2}}, {{1, 1, 3}}},
    {{{1, -2, 5}}, {{1, -1, 6}}, {{-2, 0, 7}}, {{2, 0, 1}}, {{-1, 1, 8}}, {{1, 1, 0}}},
    {{{2, -2, 0}}, {{1, -1, 2}}, {{-2, 0, 5}}, {{-1, 0, 3}}, {{0, 1, 4}}, {{-1, 2, 6}}},
    {{{0, -2, 7}}, {{1, -2, 3}}, {{-2, 0, 6}}, {{2, 0, 1}}, {{-1, 1, 8}}, {{1, 1, 4}}},
    {{{2, -2, 1}}, {{-1, -1, 7}}, {{0, -1, 0}}, {{2, -1, 7}}, {{1, 0, 8}}, {{-1, 1, 2}}},
    {{{-1, -1, 2}}, {{1, -1, 8}}, {{-1, 0, 3}}, {{2, 0, 5}}, {{0, 1, 8}}, {{-2, 2, 7}}},
    {{{0, -2, 7}}, {{1, -1, 5}}, {{-2, 0, 0}}, {{-1, 0, 6}}, {{0, 1, 4}}, {{1, 1, 7}}},
    {{{0, -2, 6}}, {{-1, -1, 7}}, {{1, -1, 8}}, {{-2, 1, 4}}, {{0, 1, 5}}, {{0, 2, 2}}},
    {{{0, -2, 7}}, {{-1, -1, 7}}, {{1, -1, 8}}, {{-1, 1, 1}}, {{1, 1, 3}}, {{0, 2, 0}}},
}};

const std::array<Sector, 3> kOutlinedSectors{{
    {{{0, -2, 7}}, {{1, -2, 3}}, {{-2, 0, 6}}, {{-1, 1, 8}}, {{1, 1, 4}}},
    {{{2, -2, 1}}, {{-1, -1, 7}}, {{0, -1, 0}}, {{2, -1, 7}}, {{1, 0, 8}}},
    {{{1, -1, 2}}, {{-1, 0, 8}}, {{2, 0, 5}}, {{0, 1, 8}}, {{-2, 2, 7}}},
}};

constexpr std::array<std::array<int, 2>, 7> kCenters2p{{
    {{0, 0}}, {{3, -5}}, {{5, -2}}, {{2, 3}}, {{-3, 5}}, {{-5, 2}}, {{-2, -3}},
}};
constexpr std::array<std::array<int, 2>, 10> kCentersNormal{{
    {{-4, -2}}, {{1, -4}}, {{6, -6}}, {{-7, 3}}, {{-2, 1}},
    {{3, -1}}, {{8, -3}}, {{-5, 6}}, {{0, 4}}, {{5, 2}},
}};
constexpr std::array<int, 8> kReduced3pCenterIndices{{0, 1, 2, 4, 5, 7, 8, 9}};
constexpr std::array<int, 14> kStartingStructures{{2, 2, 3, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2}};

std::uint64_t derived_seed(std::int64_t root, std::string_view name) {
    const auto digest = sha256_hex(std::to_string(root) +
                                   "|setup-seed-stream-v1|" +
                                   std::string(name));
    std::uint64_t result = 0;
    for (int index = 0; index < 8; ++index) {
        result |= static_cast<std::uint64_t>(std::stoul(
                      digest.substr(static_cast<std::size_t>(index * 2), 2),
                      nullptr, 16))
                  << (index * 8);
    }
    return result;
}

std::array<int, 2> rotate(int q, int r, int steps) {
    for (int index = 0; index < steps % 6; ++index) {
        const int next_q = -r;
        r = q + r;
        q = next_q;
    }
    return {q, r};
}

int hex_distance(int aq, int ar, int bq, int br) {
    const int dq = aq - bq;
    const int dr = ar - br;
    return (std::abs(dq) + std::abs(dr) + std::abs(dq + dr)) / 2;
}

void assemble_map(GaiaSetupData& setup, bool outlined) {
    setup.active_planets.fill(false);
    setup.planet_q.fill(0);
    setup.planet_r.fill(0);
    setup.terrains.fill(0);
    setup.planet_sectors.fill(-1);
    for (std::size_t position = 0; position < setup.sector_centers.size(); ++position) {
        const int tile = setup.sector_tiles[position];
        const auto& planets = outlined && tile >= 4 && tile <= 6
                                  ? kOutlinedSectors[static_cast<std::size_t>(tile - 4)]
                                  : kSolidSectors[static_cast<std::size_t>(tile)];
        for (std::size_t local = 0; local < planets.size(); ++local) {
            const auto slot = position * 7 + local;
            const auto rotated = rotate(planets[local][0], planets[local][1],
                                        setup.sector_rotations[position]);
            setup.active_planets[slot] = true;
            setup.planet_q[slot] = setup.sector_centers[position][0] + rotated[0];
            setup.planet_r[slot] = setup.sector_centers[position][1] + rotated[1];
            setup.terrains[slot] = planets[local][2];
            setup.planet_sectors[slot] = tile + 1;
        }
    }
}

bool map_is_valid(const GaiaSetupData& setup) {
    std::unordered_set<std::int64_t> positions;
    std::vector<int> active;
    for (int planet = 0; planet < kPrintedPlanetSlots; ++planet) {
        const auto index = static_cast<std::size_t>(planet);
        if (!setup.active_planets[index]) continue;
        const auto key = (static_cast<std::uint64_t>(
                              static_cast<std::uint32_t>(setup.planet_q[index]))
                          << 32U) |
                         static_cast<std::uint32_t>(setup.planet_r[index]);
        if (!positions.insert(key).second) return false;
        active.push_back(planet);
    }
    for (std::size_t left_index = 0; left_index < active.size(); ++left_index) {
        const auto left = static_cast<std::size_t>(active[left_index]);
        for (std::size_t right_index = left_index + 1; right_index < active.size(); ++right_index) {
            const auto right = static_cast<std::size_t>(active[right_index]);
            if (setup.terrains[left] == setup.terrains[right] &&
                setup.terrains[left] < 7 &&
                hex_distance(setup.planet_q[left], setup.planet_r[left],
                             setup.planet_q[right], setup.planet_r[right]) == 1) {
                return false;
            }
        }
    }
    return true;
}

template <typename Range>
void append_array(std::ostringstream& out, const Range& values) {
    out << '[';
    bool first = true;
    for (const auto& value : values) {
        if (!first) out << ',';
        first = false;
        using Value = std::remove_cvref_t<decltype(value)>;
        if constexpr (std::is_same_v<Value, bool>) out << (value ? "true" : "false");
        else out << value;
    }
    out << ']';
}

template <typename Range>
void append_nested_array(std::ostringstream& out, const Range& values) {
    out << '[';
    bool first = true;
    for (const auto& value : values) {
        if (!first) out << ',';
        first = false;
        append_array(out, value);
    }
    out << ']';
}

void append_string(std::ostringstream& out, std::string_view value) {
    out << '"' << value << '"';
}

std::string setup_json(const GaiaSetupData& setup,
                       const std::vector<int>& selected_boosters) {
    std::ostringstream out;
    out << '{';
    append_string(out, "active_planets"); out << ':'; append_array(out, setup.active_planets); out << ',';
    append_string(out, "advanced_tech_tiles"); out << ':'; append_array(out, setup.advanced_tech_tiles); out << ',';
    append_string(out, "booster_tiles"); out << ':'; append_array(out, selected_boosters); out << ',';
    append_string(out, "faction_indices"); out << ':';
    append_array(out, std::vector<int>(setup.factions.begin(),
                                       setup.factions.begin() + setup.player_count));
    out << ',';
    append_string(out, "final_scoring_tiles"); out << ':'; append_array(out, setup.final_scoring_tiles); out << ',';
    append_string(out, "first_player"); out << ':' << setup.first_player << ',';
    append_string(out, "map_mode"); out << ':'; append_string(out, setup.map_mode); out << ',';
    append_string(out, "placement_order"); out << ':'; append_array(out, setup.placement_order); out << ',';
    append_string(out, "planet_q"); out << ':'; append_array(out, setup.planet_q); out << ',';
    append_string(out, "planet_r"); out << ':'; append_array(out, setup.planet_r); out << ',';
    append_string(out, "planet_sectors"); out << ':'; append_array(out, setup.planet_sectors); out << ',';
    append_string(out, "planet_source_catalog"); out << ':'; append_nested_array(out, setup.planet_source_catalog); out << ',';
    append_string(out, "planet_source_ids"); out << ':'; append_array(out, setup.planet_source_ids); out << ',';
    append_string(out, "planet_source_q"); out << ':'; append_array(out, setup.planet_source_q); out << ',';
    append_string(out, "planet_source_r"); out << ':'; append_array(out, setup.planet_source_r); out << ',';
    append_string(out, "round_scoring_tiles"); out << ':'; append_array(out, setup.round_scoring_tiles); out << ',';
    append_string(out, "sector_centers"); out << ':'; append_nested_array(out, setup.sector_centers); out << ',';
    append_string(out, "sector_rotations"); out << ':'; append_array(out, setup.sector_rotations); out << ',';
    append_string(out, "sector_tiles"); out << ':'; append_array(out, setup.sector_tiles); out << ',';
    append_string(out, "seed"); out << ':' << setup.seed << ',';
    append_string(out, "seed_stream_version"); out << ':'; append_string(out, setup.seed_stream_version); out << ',';
    append_string(out, "seed_streams"); out << ":[";
    for (std::size_t index = 0; index < setup.seed_streams.size(); ++index) {
        if (index) out << ',';
        out << '['; append_string(out, setup.seed_streams[index].first); out << ',' << setup.seed_streams[index].second << ']';
    }
    out << "],";
    append_string(out, "standard_tech_tiles"); out << ':'; append_array(out, setup.standard_tech_tiles); out << ',';
    append_string(out, "terraforming_federation_tile"); out << ':' << setup.terraforming_federation_tile << ',';
    append_string(out, "terrains"); out << ':'; append_array(out, setup.terrains);
    out << '}';
    return out.str();
}

}  // namespace

GaiaSetupData generate_gaia_setup(std::int32_t player_count,
                                  std::int64_t seed,
                                  std::string map_size) {
    if (player_count < 2 || player_count > 4) {
        throw std::invalid_argument("player_count must be between two and four");
    }
    if (seed < 0) throw std::invalid_argument("setup seed must be non-negative");
    if (map_size.empty()) map_size = player_count == 2 ? "reduced" : "normal";
    if (map_size != "normal" && map_size != "reduced") {
        throw std::invalid_argument("map_size must be normal or reduced");
    }
    if (player_count == 2 && map_size != "reduced") {
        throw std::invalid_argument("two-player games must use the reduced map");
    }
    if (player_count == 4 && map_size != "normal") {
        throw std::invalid_argument("four-player games must use the normal map");
    }

    GaiaSetupData setup;
    setup.player_count = player_count;
    setup.seed = seed;
    setup.first_player = static_cast<int>(seed % player_count);
    setup.seed_streams = {{"map", static_cast<std::uint64_t>(seed)}};
    for (const std::string_view name : {"factions", "boosters", "round_scoring",
                                        "final_scoring", "standard_technology",
                                        "advanced_technology", "terraforming_federation"}) {
        setup.seed_streams.emplace_back(std::string(name), derived_seed(seed, name));
    }
    setup.planet_sectors.fill(-1);
    setup.planet_source_ids.fill(-1);
    setup.booster_owner.fill(-2);
    if (player_count == 2) {
        setup.sector_centers.assign(kCenters2p.begin(), kCenters2p.end());
    } else if (map_size == "reduced") {
        for (const int index : kReduced3pCenterIndices) {
            setup.sector_centers.push_back(kCentersNormal[static_cast<std::size_t>(index)]);
        }
    } else {
        setup.sector_centers.assign(kCentersNormal.begin(), kCentersNormal.end());
    }

    NumpyRandom map_rng(static_cast<std::uint64_t>(seed));
    bool valid = false;
    for (int attempt = 0; attempt < 2000 && !valid; ++attempt) {
        setup.sector_tiles = map_rng.permutation(
            static_cast<int>(setup.sector_centers.size()));
        setup.sector_rotations.clear();
        for (std::size_t index = 0; index < setup.sector_centers.size(); ++index) {
            setup.sector_rotations.push_back(static_cast<int>(map_rng.integers(6)));
        }
        assemble_map(setup, player_count == 2);
        valid = map_is_valid(setup);
    }
    if (!valid) throw std::runtime_error("unable to assemble a valid random sector map");

    setup.planet_source_q = setup.planet_q;
    setup.planet_source_r = setup.planet_r;
    setup.planet_source_catalog.clear();
    for (int planet = 0; planet < kPrintedPlanetSlots; ++planet) {
        const auto index = static_cast<std::size_t>(planet);
        setup.planet_source_ids[index] = planet;
        if (setup.active_planets[index]) {
            setup.planet_source_catalog.push_back({planet, setup.planet_q[index],
                                                   setup.planet_r[index], setup.terrains[index],
                                                   setup.planet_sectors[index]});
        }
    }

    NumpyRandom faction_rng(derived_seed(seed, "factions"));
    const auto boards = faction_rng.choice_without_replacement(7, player_count);
    for (int player = 0; player < player_count; ++player) {
        setup.factions[static_cast<std::size_t>(player)] =
            boards[static_cast<std::size_t>(player)] * 2 +
            static_cast<int>(faction_rng.integers(2));
    }
    std::vector<int> forward;
    for (int offset = 0; offset < player_count; ++offset) {
        forward.push_back((setup.first_player + offset) % player_count);
    }
    for (int layer = 0; layer < 3; ++layer) {
        for (int offset = 0; offset < player_count; ++offset) {
            const int position = layer % 2 == 0 ? offset : player_count - offset - 1;
            const int player = forward[static_cast<std::size_t>(position)];
            const int faction = setup.factions[static_cast<std::size_t>(player)];
            if (faction != 7 && kStartingStructures[static_cast<std::size_t>(faction)] > layer) {
                setup.placement_order.push_back(player);
            }
        }
    }
    for (const int player : forward) {
        if (setup.factions[static_cast<std::size_t>(player)] == 7) {
            setup.placement_order.push_back(player);
        }
    }

    NumpyRandom booster_rng(derived_seed(seed, "boosters"));
    const auto boosters = booster_rng.choice_without_replacement(10, player_count + 3);
    for (const int booster : boosters) setup.booster_owner[static_cast<std::size_t>(booster)] = -1;
    NumpyRandom round_rng(derived_seed(seed, "round_scoring"));
    const auto round = round_rng.choice_without_replacement(10, 6);
    std::copy(round.begin(), round.end(), setup.round_scoring_tiles.begin());
    NumpyRandom final_rng(derived_seed(seed, "final_scoring"));
    const auto final = final_rng.choice_without_replacement(6, 2);
    std::copy(final.begin(), final.end(), setup.final_scoring_tiles.begin());
    NumpyRandom standard_rng(derived_seed(seed, "standard_technology"));
    const auto standard = standard_rng.permutation(9);
    std::copy(standard.begin(), standard.end(), setup.standard_tech_tiles.begin());
    NumpyRandom advanced_rng(derived_seed(seed, "advanced_technology"));
    const auto advanced = advanced_rng.choice_without_replacement(15, 6);
    std::copy(advanced.begin(), advanced.end(), setup.advanced_tech_tiles.begin());
    NumpyRandom federation_rng(derived_seed(seed, "terraforming_federation"));
    setup.terraforming_federation_tile = static_cast<int>(federation_rng.integers(6));
    setup.setup_hash = sha256_hex(setup_json(setup, boosters));
    return setup;
}

}  // namespace gaiazero
