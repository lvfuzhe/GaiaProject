import unittest
from dataclasses import replace

import numpy as np

from gaiazero.game.gaia_state import (
    BOOSTER_COUNT,
    FEDERATION_ACTION,
    FACTIONS,
    MAX_BUILDINGS,
    Building,
    GaiaHeuristicEvaluator,
    GaiaState,
    PlayerState,
    Terrain,
    Track,
)
from gaiazero.game.gaia_setup import hex_distance
from gaiazero.mcts import PUCTSearch, SearchConfig


class StandardGaiaRulesTests(unittest.TestCase):
    def test_setup_has_full_research_and_building_model(self) -> None:
        state = GaiaState.initial(4, seed=3)

        self.assertEqual(state.action_size, 267)
        self.assertEqual(len(state.players[0].tracks), 6)
        self.assertEqual(state.current_player, 3)
        self.assertEqual(sum(state.active_planets), 40)
        self.assertEqual(len(state.sector_tiles), 10)
        self.assertEqual(
            sum(owner >= 0 for owner in state.owners),
            sum(len(planets) for planets in state.starting_planets),
        )
        self.assertEqual(state.snapshot()["ruleset"], "standard-v3")

    def test_random_setup_is_seeded_and_respects_component_counts(self) -> None:
        first = GaiaState.initial(2, seed=19)
        repeated = GaiaState.initial(2, seed=19)
        different = GaiaState.initial(2, seed=20)

        self.assertEqual(first.sector_tiles, repeated.sector_tiles)
        self.assertEqual(first.sector_rotations, repeated.sector_rotations)
        self.assertEqual(first.round_scoring_tiles, repeated.round_scoring_tiles)
        self.assertNotEqual(
            (first.sector_tiles, first.sector_rotations, first.players),
            (different.sector_tiles, different.sector_rotations, different.players),
        )
        self.assertEqual(sum(first.active_planets), 28)
        self.assertEqual(len(first.sector_tiles), 7)
        self.assertEqual(len(set(first.round_scoring_tiles)), 6)
        self.assertEqual(len(set(first.final_scoring_tiles)), 2)
        self.assertEqual(len(set(first.standard_tech_tiles)), 9)
        self.assertEqual(len(set(first.advanced_tech_tiles)), 6)
        self.assertEqual(sum(owner != -2 for owner in first.booster_owner), first.num_players + 3)
        self.assertEqual(len(first.booster_owner), BOOSTER_COUNT)

    def test_snapshot_preserves_complete_initial_setup(self) -> None:
        snapshot = GaiaState.initial(3, seed=41).snapshot()
        setup = snapshot["setup"]

        self.assertEqual(snapshot["ruleset"], "standard-v3")
        self.assertEqual(setup["seed"], 41)
        self.assertEqual(setup["map"]["sector_count"], 10)
        self.assertEqual(len(setup["map"]["sectors"]), 10)
        self.assertEqual(len(setup["factions"]), 3)
        self.assertEqual(len(setup["faction_catalog"]), 14)
        self.assertEqual(
            {faction["name"] for faction in setup["faction_catalog"]},
            {faction.name for faction in FACTIONS},
        )
        self.assertEqual(
            len({(faction["board"], faction["side"]) for faction in setup["faction_catalog"]}),
            14,
        )
        self.assertTrue(all(
            faction["side"] in ("A", "B")
            for faction in setup["factions"]
        ))
        self.assertEqual(len(setup["boosters"]), 6)
        self.assertEqual(len(setup["round_scoring"]), 6)
        self.assertEqual(len(setup["final_scoring"]), 2)
        self.assertEqual(len(setup["standard_tech"]), 9)
        self.assertEqual(len(setup["advanced_tech"]), 6)
        self.assertIn("terraforming_federation", setup)
        self.assertTrue(all(sector["side"] == "solid" for sector in setup["map"]["sectors"]))
        self.assertTrue(all(isinstance(player["satellites"], int) for player in snapshot["players"]))
        self.assertTrue(all(isinstance(player["colonized_types"], int) for player in snapshot["players"]))
        for player in snapshot["players"]:
            self.assertIsInstance(player["gaiaformers_on_board"], int)
            for building, maximum in MAX_BUILDINGS.items():
                inventory = player["structures"][building.name.lower()]
                self.assertEqual(inventory["built"] + inventory["supply"], maximum)
                self.assertEqual(
                    inventory["built"],
                    sum(
                        planet["owner"] == player["id"]
                        and planet["building"] == building.name.lower()
                        for planet in snapshot["planets"]
                    ),
                )

        two_player_sectors = GaiaState.initial(2, seed=41).snapshot()["setup"]["map"]["sectors"]
        self.assertTrue(all(
            sector["side"] == ("outlined" if sector["tile"] in (5, 6, 7) else "solid")
            for sector in two_player_sectors
        ))

    def test_observation_encodes_public_booster_pool(self) -> None:
        state = GaiaState.initial(2, seed=17)
        available = next(
            booster for booster, owner in enumerate(state.booster_owner) if owner == -1
        )
        excluded = next(
            booster for booster, owner in enumerate(state.booster_owner) if owner == -2
        )
        owners = list(state.booster_owner)
        owners[available], owners[excluded] = owners[excluded], owners[available]
        changed = replace(state, booster_owner=tuple(owners))

        self.assertFalse(np.array_equal(state.observation(), changed.observation()))

    def test_random_factions_use_different_boards_and_home_planets(self) -> None:
        for seed in range(20):
            state = GaiaState.initial(4, seed)
            factions = [FACTIONS[player.faction] for player in state.players]
            self.assertEqual(len({faction.board for faction in factions}), 4)
            for player, faction in enumerate(factions):
                for planet in state.starting_planets[player]:
                    self.assertEqual(state.terrains[planet], faction.home)
                    self.assertEqual(state.owners[planet], player)

    def test_manual_setup_applies_factions_first_player_and_home_planets(self) -> None:
        state = GaiaState.initial(
            3,
            seed=17,
            faction_indices=(0, 2, 4),
            first_player=2,
        )

        self.assertEqual(state.first_player, 2)
        self.assertEqual(state.current_player, 2)
        self.assertEqual(tuple(player.faction for player in state.players), (0, 2, 4))
        for player, faction_id in enumerate((0, 2, 4)):
            faction = FACTIONS[faction_id]
            for planet in state.starting_planets[player]:
                self.assertEqual(state.terrains[planet], faction.home)
                self.assertEqual(state.owners[planet], player)

    def test_manual_setup_rejects_two_sides_of_the_same_faction_board(self) -> None:
        with self.assertRaisesRegex(ValueError, "different double-sided boards"):
            GaiaState.initial(2, faction_indices=(0, 1), first_player=0)

    def test_random_map_never_places_equal_home_types_adjacent(self) -> None:
        for seed in range(20):
            state = GaiaState.initial(2 + seed % 3, seed)
            active = [index for index, value in enumerate(state.active_planets) if value]
            for offset, left in enumerate(active):
                for right in active[offset + 1 :]:
                    if state.terrains[left] == state.terrains[right] and state.terrains[left] < 7:
                        self.assertNotEqual(state._distance(left, right), 1)

    def test_sector_artwork_footprints_do_not_overlap(self) -> None:
        local_spaces = {
            (q, r)
            for q in range(-2, 3)
            for r in range(-2, 3)
            if max(abs(q), abs(r), abs(q + r)) <= 2
        }
        self.assertEqual(len(local_spaces), 19)
        for players in (2, 3, 4):
            state = GaiaState.initial(players, seed=9)
            occupied: set[tuple[int, int]] = set()
            footprints: list[set[tuple[int, int]]] = []
            for center_q, center_r in state.sector_centers:
                footprint = {
                    (center_q + q, center_r + r)
                    for q, r in local_spaces
                }
                self.assertTrue(occupied.isdisjoint(footprint))
                occupied.update(footprint)
                footprints.append(footprint)

            seams = 0
            for left_index, left in enumerate(footprints):
                for right in footprints[left_index + 1:]:
                    contacts = sum(
                        hex_distance(left_q, left_r, right_q, right_r) == 1
                        for left_q, left_r in left
                        for right_q, right_r in right
                    )
                    if contacts:
                        self.assertGreaterEqual(contacts, 3)
                        seams += 1
            self.assertEqual(seams, 12 if players == 2 else 19)

    def test_power_charges_bowl_one_before_bowl_two(self) -> None:
        info = PlayerState(faction=0, bowl_one=1, bowl_two=2, bowl_three=0)

        charged, amount = GaiaState._charge_power(info, 4)

        self.assertEqual(amount, 4)
        self.assertEqual((charged.bowl_one, charged.bowl_two, charged.bowl_three), (0, 0, 3))
        spent = GaiaState._spend_power(charged, 2)
        self.assertEqual((spent.bowl_one, spent.bowl_two, spent.bowl_three), (2, 0, 1))

    def test_gaia_project_transforms_then_returns_gaiaformer(self) -> None:
        state = GaiaState.initial(2)
        targets = [
            planet
            for planet, active in enumerate(state.active_planets)
            if active and state.terrains[planet] == Terrain.TRANSDIM
        ]
        target = targets[0]
        source = min(
            (
                planet
                for planet, active in enumerate(state.active_planets)
                if active and planet != target
            ),
            key=lambda planet: state._distance(planet, target),
        )
        owners = list(state.owners)
        buildings = list(state.buildings)
        owners[source] = 0
        buildings[source] = Building.MINE
        players = list(state.players)
        tracks = list(players[0].tracks)
        tracks[Track.GAIA_PROJECT] = 1
        tracks[Track.NAVIGATION] = 5
        players[0] = replace(
            players[0],
            faction=0,
            tracks=tuple(tracks),
            gaiaformers=1,
            bowl_one=6,
            bowl_two=0,
            bowl_three=0,
        )
        state = replace(
            state,
            player_to_move=0,
            owners=tuple(owners),
            buildings=tuple(int(value) for value in buildings),
            players=tuple(players),
        )
        action = state.gaia_action(target)
        self.assertIn(action, state.legal_actions())

        started = state.apply(action)
        self.assertEqual(started.gaiaformer_owner[target], 0)
        self.assertEqual(started.players[0].gaia_power, 6)
        transformed = started._gaia_phase()
        self.assertEqual(transformed.terrains[target], Terrain.GAIA)
        self.assertEqual(transformed.players[0].gaia_power, 0)
        self.assertEqual(transformed.players[0].bowl_two, 6)

        transformed = replace(transformed, player_to_move=0)
        built = transformed.apply(transformed.build_action(target))
        self.assertEqual(built.owners[target], 0)
        self.assertEqual(built.gaiaformer_owner[target], -1)
        self.assertEqual(built.players[0].gaiaformers, 1)

    def test_research_lab_requires_immediate_tech_choice(self) -> None:
        state = GaiaState.initial(2)
        mine = next(
            planet
            for planet, owner in enumerate(state.owners)
            if owner == 0 and state.buildings[planet] == Building.MINE
        )
        state = replace(state, player_to_move=0)
        state = state.apply(state.upgrade_trading_action(mine))
        players = list(state.players)
        players[0] = replace(players[0], credits=30, ore=15)
        state = replace(state, player_to_move=0, players=tuple(players))

        pending = state.apply(state.upgrade_lab_action(mine))

        self.assertEqual(pending.current_player, 0)
        self.assertEqual(pending.pending_tech_player, 0)
        self.assertTrue(pending.legal_actions())
        self.assertTrue(all(action >= pending.tech_action(Track.TERRAFORMING) for action in pending.legal_actions()))
        resolved = pending.apply(pending.tech_action(Track.TERRAFORMING))
        self.assertEqual(resolved.pending_tech_player, -1)
        self.assertEqual(resolved.current_player, 1)
        self.assertEqual(resolved.players[0].tracks[Track.TERRAFORMING], 1)

    def test_upgrade_requiring_tech_is_hidden_when_no_choice_remains(self) -> None:
        state = GaiaState.initial(2)
        planet = next(index for index, active in enumerate(state.active_planets) if active)
        owners = list(state.owners)
        buildings = list(state.buildings)
        owners[planet] = 0
        buildings[planet] = Building.TRADING_STATION
        players = list(state.players)
        players[0] = replace(players[0], credits=30, ore=15, tracks=(5, 5, 5, 5, 5, 5))
        state = replace(
            state,
            player_to_move=0,
            players=tuple(players),
            owners=tuple(owners),
            buildings=tuple(int(value) for value in buildings),
        )

        self.assertNotIn(state.upgrade_lab_action(planet), state.legal_actions())

    def test_canonical_federation_awards_token(self) -> None:
        state = GaiaState.initial(2)
        planets = [index for index, active in enumerate(state.active_planets) if active][:3]
        owners = list(state.owners)
        buildings = list(state.buildings)
        for planet in planets:
            owners[planet] = 0
        buildings[planets[0]] = Building.PLANETARY_INSTITUTE
        buildings[planets[1]] = Building.ACADEMY
        buildings[planets[2]] = Building.MINE
        players = list(state.players)
        players[0] = replace(players[0], bowl_one=15, bowl_two=10, bowl_three=0)
        state = replace(
            state,
            player_to_move=0,
            owners=tuple(owners),
            buildings=tuple(int(value) for value in buildings),
            players=tuple(players),
        )

        self.assertIn(FEDERATION_ACTION, state.legal_actions())
        formed = state.apply(FEDERATION_ACTION)
        self.assertEqual(formed.players[0].federation_tokens, 1)
        self.assertEqual(formed.players[0].federation_keys, 1)
        self.assertTrue(all(formed.federated[planet] for planet in planets))

    def test_passing_returns_old_booster_and_takes_new_one(self) -> None:
        state = GaiaState.initial(2)
        player = state.current_player
        old_booster = state._player_booster(player)
        new_booster = next(
            booster for booster, owner in enumerate(state.booster_owner) if owner == -1
        )
        passed = state.apply(state.pass_booster_action(new_booster))

        self.assertEqual(passed.booster_owner[old_booster], -1)
        self.assertEqual(passed.booster_owner[new_booster], player)
        self.assertTrue(passed.players[player].passed)

    def test_random_legal_playouts_reach_terminal_state(self) -> None:
        for seed in range(12):
            rng = np.random.default_rng(seed)
            state = GaiaState.initial(2 + seed % 3, seed)
            moves = 0
            while not state.is_terminal and moves < 500:
                legal = state.legal_actions()
                self.assertTrue(legal)
                state = state.apply(int(rng.choice(legal)))
                moves += 1
            self.assertTrue(state.is_terminal)
            self.assertLess(moves, 500)
            self.assertAlmostEqual(float(state.returns().sum()), 0.0, places=5)

    def test_puct_operates_on_standard_rules(self) -> None:
        state = GaiaState.initial(3)
        result = PUCTSearch(
            GaiaHeuristicEvaluator(),
            SearchConfig(simulations=12, seed=4),
        ).run(state)

        self.assertEqual(int(result.visits.sum()), 12)
        self.assertAlmostEqual(float(result.policy.sum()), 1.0, places=6)
        self.assertEqual(result.root_value.shape, (3,))
        self.assertTrue(np.all(result.policy[~state.legal_action_mask()] == 0))


if __name__ == "__main__":
    unittest.main()
