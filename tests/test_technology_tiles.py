import unittest
from dataclasses import replace

from gaiazero.game.gaia_state import (
    ADVANCED_TECH_ACTION_OFFSET,
    ADVANCED_TECH_TILES,
    SKIP_TECH_RESEARCH_ACTION,
    STANDARD_TECH_ACTION,
    STANDARD_TECH_COUNT,
    STANDARD_TECH_TILES,
    Building,
    GaiaState,
    Terrain,
    Track,
)


def finish_setup(state: GaiaState) -> GaiaState:
    while state.is_starting_placement or state.is_booster_selection:
        legal = state.legal_actions()
        if not legal:
            raise AssertionError("setup has no legal action")
        state = state.apply(legal[0])
    return state


class TechnologyTileRulesTests(unittest.TestCase):
    def _base_state(self, *, factions: tuple[int, int] = (0, 2)) -> GaiaState:
        return replace(
            finish_setup(
                GaiaState.initial(
                    2,
                    seed=71,
                    faction_indices=factions,
                    first_player=0,
                    standard_tech_tiles=tuple(range(STANDARD_TECH_COUNT)),
                )
            ),
            player_to_move=0,
            round_scoring_tiles=(0, 1, 2, 3, 4, 5),
        )

    def _technology_layout_state(self) -> GaiaState:
        state = self._base_state(factions=(1, 2))
        planets = [
            planet
            for planet, active in enumerate(state.active_planets)
            if active
        ][:9]
        structures = (
            Building.MINE,
            Building.MINE,
            Building.MINE,
            Building.TRADING_STATION,
            Building.TRADING_STATION,
            Building.RESEARCH_LAB,
            Building.PLANETARY_INSTITUTE,
            Building.ACADEMY,
        )
        owners = [-1] * len(state.owners)
        buildings = [Building.EMPTY] * len(state.buildings)
        coexisting = [-1] * len(state.coexisting_mine_owner)
        sectors = list(state.planet_sectors)
        terrains = list(state.terrains)
        for sector, (planet, building) in enumerate(
            zip(planets[:8], structures, strict=True)
        ):
            owners[planet] = 0
            buildings[planet] = building
            sectors[planet] = sector
            terrains[planet] = Terrain.TERRA
        # A Lantids coexisting mine is a mine and occupies a sector, but does not
        # count for effects based on colonized planet types or Gaia planets.
        owners[planets[8]] = 1
        buildings[planets[8]] = Building.MINE
        coexisting[planets[8]] = 0
        sectors[planets[8]] = 8
        terrains[planets[0]] = Terrain.GAIA
        terrains[planets[1]] = Terrain.GAIA
        terrains[planets[8]] = Terrain.GAIA

        players = list(state.players)
        players[0] = replace(
            players[0],
            ore=2,
            vp=10,
            colonized_types=(
                (1 << int(Terrain.TERRA))
                | (1 << int(Terrain.DESERT))
                | (1 << int(Terrain.GAIA))
                | (1 << int(Terrain.LOST))
            ),
            federation_tokens=2,
        )
        return replace(
            state,
            players=tuple(players),
            owners=tuple(owners),
            buildings=tuple(int(building) for building in buildings),
            coexisting_mine_owner=tuple(coexisting),
            planet_sectors=tuple(sectors),
            terrains=tuple(int(terrain) for terrain in terrains),
        )

    def _gaia_build_state(
        self,
        *,
        standard_tiles: int = 0,
        covered_tiles: int = 0,
        advanced_tiles: int = 0,
    ) -> tuple[GaiaState, int]:
        state = self._base_state()
        players = list(state.players)
        players[0] = replace(
            players[0],
            credits=30,
            ore=15,
            qic=10,
            tech_tiles=standard_tiles,
            covered_tech_tiles=covered_tiles,
            advanced_tech_tiles=advanced_tiles,
        )
        state = replace(state, players=tuple(players))
        for target, active in enumerate(state.active_planets):
            if not active or state.owners[target] != -1:
                continue
            terrains = list(state.terrains)
            terrains[target] = Terrain.GAIA
            candidate = replace(
                state,
                terrains=tuple(int(terrain) for terrain in terrains),
            )
            if candidate.build_action(target) in candidate.legal_actions():
                return candidate, target
        raise AssertionError("no legal Gaia planet was available for the tile test")

    def test_technology_catalog_matches_all_physical_tiles(self) -> None:
        self.assertEqual(
            tuple(tile.key for tile in STANDARD_TECH_TILES),
            (
                "ore-qic",
                "planet-type-knowledge",
                "vp-7",
                "gaia-mine-vp",
                "structure-power",
                "ore-power-income",
                "knowledge-credit-income",
                "credits-income",
                "power-action",
            ),
        )
        self.assertEqual(
            tuple(tile.label for tile in ADVANCED_TECH_TILES),
            (
                "Action: gain 1 Q.I.C. and 5 credits",
                "Action: gain 3 ore",
                "Action: gain 3 knowledge",
                "2 VP per mine",
                "1 ore per colonized sector",
                "2 VP per colonized sector",
                "2 VP per Gaia planet",
                "5 VP per federation token",
                "4 VP per trading station",
                "Pass: 3 VP per federation token",
                "Pass: 3 VP per research lab",
                "Pass: 1 VP per planet type",
                "2 VP per research advance",
                "3 VP per mine built",
                "3 VP per trading station built",
            ),
        )

    def test_all_standard_immediate_tiles(self) -> None:
        state = self._base_state()
        info = replace(
            state.players[0],
            ore=4,
            qic=1,
            knowledge=3,
            vp=10,
            tech_tiles=0,
            colonized_types=0b1111,
        )

        ore_qic = state._gain_standard_tech(info, 0)
        knowledge = state._gain_standard_tech(info, 1)
        victory_points = state._gain_standard_tech(info, 2)

        self.assertEqual((ore_qic.ore, ore_qic.qic), (5, 2))
        self.assertEqual(knowledge.knowledge, 7)
        self.assertEqual(victory_points.vp, 17)
        for tile, gained in enumerate((ore_qic, knowledge, victory_points)):
            with self.subTest(tile=tile):
                self.assertTrue(gained.tech_tiles & (1 << tile))

    def test_all_standard_income_tiles_and_covering(self) -> None:
        state = self._base_state()
        base_players = list(state.players)
        base_players[0] = replace(base_players[0], tech_tiles=0)
        base = replace(state, players=tuple(base_players))._income_preview(0)
        deltas = {
            5: {"ore": 1, "power_charge": 1},
            6: {"credits": 1, "knowledge": 1},
            7: {"credits": 4},
        }

        for tile, expected in deltas.items():
            with self.subTest(tile=tile):
                players = list(state.players)
                players[0] = replace(players[0], tech_tiles=1 << tile)
                active = replace(state, players=tuple(players))._income_preview(0)
                for resource in base:
                    self.assertEqual(
                        active[resource] - base[resource],
                        expected.get(resource, 0),
                    )

                players[0] = replace(players[0], covered_tech_tiles=1 << tile)
                covered = replace(state, players=tuple(players))._income_preview(0)
                self.assertEqual(covered, base)

    def test_standard_gaia_mine_tile_stops_after_covering(self) -> None:
        active, target = self._gaia_build_state(standard_tiles=1 << 3)
        covered, _ = self._gaia_build_state(
            standard_tiles=1 << 3,
            covered_tiles=1 << 3,
        )

        active_result = active._apply_build(target).players[0]
        covered_result = covered._apply_build(target).players[0]

        self.assertEqual(active_result.vp, covered_result.vp + 3)

    def test_standard_structure_power_tile_stops_after_covering(self) -> None:
        state = self._base_state()
        players = list(state.players)
        players[0] = replace(players[0], tech_tiles=1 << 4)
        active = replace(state, players=tuple(players))
        players[0] = replace(players[0], covered_tech_tiles=1 << 4)
        covered = replace(state, players=tuple(players))

        for building in (Building.PLANETARY_INSTITUTE, Building.ACADEMY):
            with self.subTest(building=building):
                self.assertEqual(active._structure_power(0, building), 4)
                self.assertEqual(covered._structure_power(0, building), 3)

    def test_standard_power_action_charges_four_and_stops_after_covering(self) -> None:
        state = self._base_state()
        players = list(state.players)
        players[0] = replace(
            players[0],
            tech_tiles=1 << 8,
            bowl_one=2,
            bowl_two=2,
            bowl_three=0,
        )
        state = replace(state, players=tuple(players))

        used = state.apply(STANDARD_TECH_ACTION)

        self.assertEqual(
            (used.players[0].bowl_one, used.players[0].bowl_two, used.players[0].bowl_three),
            (0, 2, 2),
        )
        self.assertTrue(used.players[0].used_standard_tech_action)
        players[0] = replace(players[0], covered_tech_tiles=1 << 8)
        covered = replace(state, players=tuple(players))
        self.assertNotIn(STANDARD_TECH_ACTION, covered.legal_actions())

    def test_all_advanced_special_action_tiles(self) -> None:
        expected = {
            0: {"credits": 5, "qic": 1},
            1: {"ore": 3},
            2: {"knowledge": 3},
        }
        for tile, deltas in expected.items():
            with self.subTest(tile=tile):
                state = self._base_state()
                players = list(state.players)
                players[0] = replace(
                    players[0],
                    credits=5,
                    ore=2,
                    knowledge=3,
                    qic=1,
                    advanced_tech_tiles=1 << tile,
                )
                state = replace(state, players=tuple(players))
                before = state.players[0]
                action = ADVANCED_TECH_ACTION_OFFSET + tile

                self.assertIn(action, state.legal_actions())
                used = state.apply(action)
                for resource, delta in deltas.items():
                    self.assertEqual(
                        getattr(used.players[0], resource),
                        getattr(before, resource) + delta,
                    )
                self.assertTrue(used.players[0].used_advanced_tech_actions & (1 << tile))
                self.assertNotIn(action, replace(used, player_to_move=0).legal_actions())

    def test_all_advanced_immediate_board_rewards(self) -> None:
        state = self._technology_layout_state()
        info = state.players[0]
        expected = {
            3: ("vp", 8),
            4: ("ore", 9),
            5: ("vp", 18),
            6: ("vp", 4),
            7: ("vp", 10),
            8: ("vp", 8),
        }

        for tile, (resource, delta) in expected.items():
            with self.subTest(tile=tile):
                gained = state._gain_advanced_tech_reward(0, info, tile)
                self.assertEqual(
                    getattr(gained, resource),
                    getattr(info, resource) + delta,
                )

    def test_all_advanced_pass_tiles(self) -> None:
        state = self._technology_layout_state()
        expected = {9: 6, 10: 3, 11: 4}

        for tile, points in expected.items():
            with self.subTest(tile=tile):
                players = list(state.players)
                players[0] = replace(players[0], advanced_tech_tiles=1 << tile)
                candidate = replace(state, players=tuple(players))
                self.assertEqual(candidate._booster_pass_points(0, -1), points)

    def test_advanced_research_tile_scores_every_advance(self) -> None:
        state = self._base_state()
        info = replace(
            state.players[0],
            vp=10,
            tracks=(0, 0, 0, 0, 0, 0),
            advanced_tech_tiles=1 << 12,
        )

        advanced = state._advance_research(
            0,
            info,
            Track.SCIENCE,
            score_round=False,
        )

        self.assertEqual(advanced.tracks[Track.SCIENCE], 1)
        self.assertEqual(advanced.vp, 12)

    def test_advanced_mine_tile_scores_normal_builds(self) -> None:
        state, target = self._gaia_build_state(advanced_tiles=1 << 13)
        players = list(state.players)
        players[0] = replace(players[0], advanced_tech_tiles=0)
        without_tile = replace(state, players=tuple(players))

        scored = state._apply_build(target).players[0]
        baseline = without_tile._apply_build(target).players[0]

        self.assertEqual(scored.vp, baseline.vp + 3)

    def test_advanced_trading_station_tile_scores_upgrade(self) -> None:
        state = self._base_state()
        mine = state.starting_planets[0][0]
        players = list(state.players)
        players[0] = replace(
            players[0],
            credits=30,
            ore=15,
            advanced_tech_tiles=1 << 14,
        )
        state = replace(state, players=tuple(players))
        players[0] = replace(players[0], advanced_tech_tiles=0)
        without_tile = replace(state, players=tuple(players))

        scored = state._apply_upgrade(mine, Building.TRADING_STATION).players[0]
        baseline = without_tile._apply_upgrade(mine, Building.TRADING_STATION).players[0]

        self.assertEqual(scored.vp, baseline.vp + 3)

    def test_advanced_acquisition_covers_tile_spends_key_and_scores_free_research(self) -> None:
        state = replace(
            self._base_state(),
            advanced_tech_tiles=(12, 0, 1, 2, 3, 4),
            pending_tech_player=0,
        )
        players = list(state.players)
        players[0] = replace(
            players[0],
            tracks=(4, 0, 0, 0, 0, 0),
            tech_tiles=1 << 5,
            federation_keys=1,
            vp=10,
        )
        state = replace(state, players=tuple(players))

        selected = state.apply(
            state.tech_action(STANDARD_TECH_COUNT + Track.TERRAFORMING)
        )
        covered = selected.apply(selected.tech_action(5))

        self.assertTrue(covered.players[0].advanced_tech_tiles & (1 << 12))
        self.assertTrue(covered.players[0].covered_tech_tiles & (1 << 5))
        self.assertEqual(covered.players[0].federation_keys, 0)
        self.assertEqual(covered.pending_research_player, 0)
        self.assertTrue(covered.pending_research_optional)
        self.assertIn(SKIP_TECH_RESEARCH_ACTION, covered.legal_actions())
        resolved = covered.apply(covered.research_action(Track.SCIENCE))
        self.assertEqual(resolved.players[0].tracks[Track.SCIENCE], 1)
        self.assertEqual(resolved.players[0].vp, 12)

    def test_standard_track_tile_research_advance_is_optional_and_track_locked(self) -> None:
        state = replace(self._base_state(), pending_tech_player=0)
        players = list(state.players)
        players[0] = replace(
            players[0],
            tracks=(0, 0, 0, 0, 0, 0),
            tech_tiles=0,
        )
        state = replace(state, players=tuple(players))

        choosing = state.apply(state.tech_action(Track.TERRAFORMING))

        self.assertEqual(choosing.players[0].tracks[Track.TERRAFORMING], 0)
        self.assertEqual(choosing.pending_research_track, Track.TERRAFORMING)
        self.assertTrue(choosing.pending_research_optional)
        self.assertEqual(
            set(choosing.legal_actions()),
            {
                choosing.research_action(Track.TERRAFORMING),
                SKIP_TECH_RESEARCH_ACTION,
            },
        )

        skipped = choosing.apply(SKIP_TECH_RESEARCH_ACTION)
        self.assertEqual(skipped.players[0].tracks[Track.TERRAFORMING], 0)
        self.assertEqual(skipped.pending_research_player, -1)
        self.assertEqual(skipped.pending_research_track, -1)
        self.assertFalse(skipped.pending_research_optional)
        self.assertEqual(skipped.player_to_move, 1)

    def test_optional_research_details_are_encoded_in_snapshot_and_observation(self) -> None:
        state = self._base_state()
        any_track = replace(
            state,
            pending_research_player=0,
            pending_research_track=-1,
            pending_research_optional=True,
        )
        fixed_track = replace(
            any_track,
            pending_research_track=Track.SCIENCE,
        )

        details = fixed_track.snapshot()["technology_research"]
        self.assertEqual(
            details,
            {
                "active": True,
                "player": 0,
                "track": Track.SCIENCE,
                "optional": True,
                "skip_action": SKIP_TECH_RESEARCH_ACTION,
            },
        )
        self.assertEqual(any_track.observation().shape, fixed_track.observation().shape)
        self.assertFalse(
            (any_track.observation() == fixed_track.observation()).all()
        )

    def test_mandatory_firaks_research_cannot_be_skipped(self) -> None:
        state = self._base_state()
        mandatory = replace(
            state,
            pending_research_player=0,
            pending_research_track=-1,
            pending_research_optional=False,
        )

        self.assertNotIn(SKIP_TECH_RESEARCH_ACTION, mandatory.legal_actions())

    def test_advanced_acquisition_requires_level_key_and_uncovered_standard_tile(self) -> None:
        state = replace(
            self._base_state(),
            advanced_tech_tiles=(0, 1, 2, 3, 4, 5),
            pending_tech_player=0,
        )
        advanced_action = state.tech_action(
            STANDARD_TECH_COUNT + Track.TERRAFORMING
        )
        cases = (
            ("level", (3, 0, 0, 0, 0, 0), 1, 1 << 5, 0),
            ("key", (4, 0, 0, 0, 0, 0), 0, 1 << 5, 0),
            ("cover", (4, 0, 0, 0, 0, 0), 1, 1 << 5, 1 << 5),
        )
        for name, tracks, keys, tech_tiles, covered_tiles in cases:
            with self.subTest(requirement=name):
                players = list(state.players)
                players[0] = replace(
                    players[0],
                    tracks=tracks,
                    federation_keys=keys,
                    tech_tiles=tech_tiles,
                    covered_tech_tiles=covered_tiles,
                )
                candidate = replace(state, players=tuple(players))
                self.assertNotIn(
                    advanced_action,
                    candidate._legal_technology_actions(0),
                )

    def test_covered_standard_tile_cannot_be_taken_again(self) -> None:
        state = replace(self._base_state(), pending_tech_player=0)
        players = list(state.players)
        players[0] = replace(
            players[0],
            tech_tiles=1 << 5,
            covered_tech_tiles=1 << 5,
        )
        state = replace(state, players=tuple(players))

        self.assertNotIn(state.tech_action(5), state._legal_technology_actions(0))


if __name__ == "__main__":
    unittest.main()
