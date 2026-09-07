#include "gaiazero/gaia_setup.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {

template <typename Range>
void print_values(const Range& values) {
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index) std::cout << ',';
        std::cout << values[index];
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 3 || argc > 4) {
            throw std::invalid_argument(
                "usage: gaiazero_setup_probe <players> <seed> [normal|reduced]");
        }
        const int players = std::stoi(argv[1]);
        const auto seed = std::stoll(argv[2]);
        const std::string map_size = argc == 4 ? argv[3] : "";
        const auto setup = gaiazero::generate_gaia_setup(players, seed, map_size);
        std::cout << "setup_hash=" << setup.setup_hash << '\n';
        std::cout << "first_player=" << setup.first_player << '\n';
        std::cout << "factions=";
        for (int index = 0; index < players; ++index) {
            if (index) std::cout << ',';
            std::cout << setup.factions[static_cast<std::size_t>(index)];
        }
        std::cout << "\nsector_tiles="; print_values(setup.sector_tiles);
        std::cout << "\nsector_rotations="; print_values(setup.sector_rotations);
        std::cout << "\nplacement_order="; print_values(setup.placement_order);
        std::cout << "\nround_scoring="; print_values(setup.round_scoring_tiles);
        std::cout << "\nfinal_scoring="; print_values(setup.final_scoring_tiles);
        std::cout << "\nstandard_tech="; print_values(setup.standard_tech_tiles);
        std::cout << "\nadvanced_tech="; print_values(setup.advanced_tech_tiles);
        std::cout << "\nterraforming_federation=" << setup.terraforming_federation_tile;
        std::cout << "\nactive=";
        for (const bool active : setup.active_planets) std::cout << (active ? '1' : '0');
        std::cout << "\nplanet_q="; print_values(setup.planet_q);
        std::cout << "\nplanet_r="; print_values(setup.planet_r);
        std::cout << "\nterrains="; print_values(setup.terrains);
        std::cout << "\nplanet_sectors="; print_values(setup.planet_sectors);
        std::cout << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
