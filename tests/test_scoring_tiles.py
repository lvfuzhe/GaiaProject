import unittest
from dataclasses import replace

from gaiazero.game.gaia_state import (
    FINAL_SCORING_NEUTRAL,
    FINAL_SCORING_TILES,
    LOST_PLANET_OFFSET,
    LOST_PLANET_SLOT,
    ROUND_SCORING_TILES,
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


class ScoringTileRulesTests(unittest.TestCase):
    def _base_state(
        self,
        *,
        players: int = 2,
        factions: tuple[int, ...] | None = None,
        round_tile: int = 0,
    ) -> GaiaState:
        state = finish_setup(
            GaiaState.initial(
                players,
                seed=83,
                faction_indices=factions,
                first_player=0,
            )
        )
        round_tiles = list(state.round_scoring_tiles)
        round_tiles[0] = round_tile
        return replace(
            state,
            player_to_move=0,
            round_number=1,
            round_scoring_tiles=tuple(round_tiles),
        )

    def _build_state(
        self,
        terrain: Terrain,
        round_tile: int,
    ) -> tuple[GaiaState, int]:
        state = self._base_state(
            factions=(0, 2),
            round_tile=round_tile,
        )
        players = list(state.players)
        players[0] = replace(
            players[0],
            credits=30,
            ore=15,
            qic=10,
            vp=10,
            tech_tiles=0,
            advanced_tech_tiles=0,
        )
        state = replace(state, players=tuple(players))
        target = next(
            planet
            for planet, active in enumerate(state.active_planets)
            if active and state.owners[planet] == -1
        )
        terrains = list(state.terrains)
        terrains[target] = terrain
        return replace(
            state,
            terrains=tuple(int(value) for value in terrains),
        ), target

    def test_scoring_catalog_matches_all_physical_tiles(self) -> None:
        self.assertEqual(
            tuple((tile.key, tile.kind, tile.points) for tile in ROUND_SCORING_TILES),
            (
                ("terraform-2", "terraform", 2),
                ("research-2", "research", 2),
                ("mine-2", "mine", 2),
                ("federation-5", "federation", 5),
                ("trading-3", "trading", 3),
                ("trading-4", "trading", 4),
                ("gaia-mine-3", "gaia", 3),
                ("gaia-mine-4", "gaia", 4),
                ("big-5a", "big", 5),
                ("big-5b", "big", 5),
            ),
        )
        self.assertEqual(
            tuple(tile.key for tile in FINAL_SCORING_TILES),
            (
                "federation-structures",
                "structures",
                "planet-types",
                "gaia-planets",
                "sectors",
                "satellites",
            ),
        )
        self.assertEqual(FINAL_SCORING_NEUTRAL, (10, 11, 5, 4, 6, 8))

    def test_all_round_scoring_tiles_award_the_printed_amount(self) -> None:
        state = self._base_state()
        for tile_id, tile in enumerate(ROUND_SCORING_TILES):
            with self.subTest(tile=tile_id):
                round_tiles = list(state.round_scoring_tiles)
                round_tiles[0] = tile_id
                candidate = replace(state, round_scoring_tiles=tuple(round_tiles))
                info = replace(candidate.players[0], vp=10)

                scored = candidate._score(info, tile.kind, amount=2)
                ignored = candidate._score(info, "not-the-current-condition", amount=2)

                self.assertEqual(scored.vp, 10 + 2 * tile.points)
                self.assertEqual(ignored.vp, 10)

    def test_terraform_scoring_counts_free_and_paid_steps(self) -> None:
        state, target = self._build_state(Terrain.VOLCANIC, round_tile=0)

        built = state._apply_build(target, free_steps=2)

        self.assertEqual(
            state._terrain_steps(Terrain.TERRA, Terrain.VOLCANIC),
            3,
        )
        self.assertEqual(built.players[0].vp, state.players[0].vp + 6)

    def test_mine_and_gaia_round_tiles_use_the_build_event(self) -> None:
        cases = (
            (Terrain.TERRA, 2, 2),
            (Terrain.GAIA, 6, 3),
            (Terrain.GAIA, 7, 4),
        )
        for terrain, tile, points in cases:
            with self.subTest(tile=tile):
                state, target = self._build_state(terrain, round_tile=tile)
                built = state._apply_build(target)
                self.assertEqual(built.players[0].vp, state.players[0].vp + points)

    def test_lantids_coexisting_and_lost_planet_mines_score_the_mine_tile(self) -> None:
        lantids = self._base_state(
            factions=(1, 2),
            round_tile=2,
        )
        target = lantids.starting_planets[1][0]
        players = list(lantids.players)
        players[0] = replace(
            players[0],
            credits=30,
            ore=15,
            qic=10,
            vp=10,
        )
        lantids = replace(lantids, players=tuple(players))

        coexisting = lantids._apply_build(target)

        self.assertEqual(coexisting.players[0].vp, 12)
        self.assertEqual(coexisting.coexisting_mine_owner[target], 0)

        lost = self._base_state(round_tile=2)
        players = list(lost.players)
        players[0] = replace(
            players[0],
            qic=20,
            vp=10,
            advanced_tech_tiles=0,
        )
        lost = replace(
            lost,
            players=tuple(players),
            pending_lost_planet_player=0,
        )
        action = next(
            action
            for action in lost.legal_actions()
            if action >= LOST_PLANET_OFFSET
        )

        placed = lost.apply(action)

        self.assertEqual(placed.players[0].vp, 12)
        self.assertEqual(placed.buildings[LOST_PLANET_SLOT], Building.MINE)

    def test_research_trading_and_big_building_round_events(self) -> None:
        research = self._base_state(round_tile=1)
        info = replace(
            research.players[0],
            vp=10,
            tracks=(0, 0, 0, 0, 0, 0),
            advanced_tech_tiles=0,
        )
        researched = research._advance_research(
            0,
            info,
            Track.SCIENCE,
        )
        self.assertEqual(researched.vp, 12)

        trading = self._base_state(round_tile=4)
        mine = trading.starting_planets[0][0]
        players = list(trading.players)
        players[0] = replace(players[0], credits=30, ore=15, vp=10)
        trading = replace(trading, players=tuple(players))
        upgraded = trading._apply_upgrade(mine, Building.TRADING_STATION)
        self.assertEqual(upgraded.players[0].vp, 13)

        big = self._base_state(round_tile=8)
        planet = big.starting_planets[0][0]
        buildings = list(big.buildings)
        buildings[planet] = Building.TRADING_STATION
        players = list(big.players)
        players[0] = replace(players[0], credits=30, ore=15, vp=10)
        big = replace(
            big,
            players=tuple(players),
            buildings=tuple(int(value) for value in buildings),
        )
        institute = big._apply_upgrade(planet, Building.PLANETARY_INSTITUTE)
        self.assertEqual(institute.players[0].vp, 15)

    def test_every_way_of_gaining_a_federation_token_scores_the_round_tile(self) -> None:
        research = self._base_state(round_tile=3)
        players = list(research.players)
        players[0] = replace(
            players[0],
            vp=10,
            tracks=(4, 0, 0, 0, 0, 0),
            federation_keys=2,
        )
        research = replace(research, players=tuple(players))
        without_round = replace(
            research,
            round_scoring_tiles=(0, *research.round_scoring_tiles[1:]),
        )

        scored = research._advance_research(
            0,
            research.players[0],
            Track.TERRAFORMING,
        )
        baseline = without_round._advance_research(
            0,
            without_round.players[0],
            Track.TERRAFORMING,
        )

        self.assertEqual(scored.federation_tokens, baseline.federation_tokens)
        self.assertEqual(scored.vp, baseline.vp + 5)

        gleens = self._base_state(
            factions=(3, 0),
            round_tile=3,
        )
        planet = gleens.starting_planets[0][0]
        buildings = list(gleens.buildings)
        buildings[planet] = Building.TRADING_STATION
        players = list(gleens.players)
        players[0] = replace(players[0], credits=30, ore=15, vp=10)
        gleens = replace(
            gleens,
            players=tuple(players),
            buildings=tuple(int(value) for value in buildings),
        )
        gleens_without_round = replace(
            gleens,
            round_scoring_tiles=(0, *gleens.round_scoring_tiles[1:]),
        )

        scored_pi = gleens._apply_upgrade(planet, Building.PLANETARY_INSTITUTE)
        baseline_pi = gleens_without_round._apply_upgrade(
            planet,
            Building.PLANETARY_INSTITUTE,
        )

        self.assertEqual(scored_pi.players[0].federation_tokens, 1)
        self.assertEqual(scored_pi.players[0].vp, baseline_pi.players[0].vp + 5)

    def test_all_final_scoring_metrics_cover_special_pieces(self) -> None:
        state = self._base_state(factions=(1, 2))
        planets = [
            planet
            for planet, active in enumerate(state.active_planets)
            if active and planet != LOST_PLANET_SLOT
        ][:3]
        owners = [-1] * len(state.owners)
        buildings = [Building.EMPTY] * len(state.buildings)
        coexisting = [-1] * len(state.coexisting_mine_owner)
        federated = [False] * len(state.federated)
        coexisting_federated = [False] * len(state.coexisting_mine_federated)
        active = list(state.active_planets)
        terrains = list(state.terrains)
        sectors = list(state.planet_sectors)

        for planet, building, terrain, sector in (
            (planets[0], Building.MINE, Terrain.GAIA, 0),
            (planets[1], Building.TRADING_STATION, Terrain.TERRA, 1),
            (LOST_PLANET_SLOT, Building.MINE, Terrain.LOST, 1),
        ):
            owners[planet] = 0
            buildings[planet] = building
            terrains[planet] = terrain
            sectors[planet] = sector
            federated[planet] = True
        active[LOST_PLANET_SLOT] = True
        owners[planets[2]] = 1
        buildings[planets[2]] = Building.MINE
        terrains[planets[2]] = Terrain.GAIA
        sectors[planets[2]] = 2
        coexisting[planets[2]] = 0
        coexisting_federated[planets[2]] = True

        station_owners = list(state.space_station_owner)
        station_federated = list(state.space_station_federated)
        station_owners[0] = 0
        station_federated[0] = True
        players = list(state.players)
        players[0] = replace(
            players[0],
            colonized_types=(
                (1 << int(Terrain.TERRA))
                | (1 << int(Terrain.GAIA))
                | (1 << int(Terrain.LOST))
            ),
            satellites=5,
        )
        state = replace(
            state,
            players=tuple(players),
            active_planets=tuple(active),
            owners=tuple(owners),
            buildings=tuple(int(value) for value in buildings),
            coexisting_mine_owner=tuple(coexisting),
            federated=tuple(federated),
            coexisting_mine_federated=tuple(coexisting_federated),
            terrains=tuple(int(value) for value in terrains),
            planet_sectors=tuple(sectors),
            space_station_owner=tuple(station_owners),
            space_station_federated=tuple(station_federated),
        )

        self.assertEqual(
            tuple(state._final_scoring_metric(0, tile) for tile in range(6)),
            (4.0, 4.0, 3.0, 1.0, 3.0, 6.0),
        )

    def test_final_ranking_handles_distinct_positions_and_all_ties(self) -> None:
        state = GaiaState.initial(4, seed=91)

        self.assertEqual(
            state._ranking_awards([9, 8, 7, 6]),
            [18.0, 12.0, 6.0, 0.0],
        )
        self.assertEqual(
            state._ranking_awards([9, 9, 7, 6]),
            [15.0, 15.0, 6.0, 0.0],
        )
        self.assertEqual(
            state._ranking_awards([9, 7, 7, 6]),
            [18.0, 9.0, 9.0, 0.0],
        )
        self.assertEqual(
            state._ranking_awards([7, 7, 7, 7]),
            [9.0, 9.0, 9.0, 9.0],
        )

    def test_two_player_neutral_marker_uses_every_printed_value(self) -> None:
        state = GaiaState.initial(2, seed=97)
        for tile, neutral in enumerate(FINAL_SCORING_NEUTRAL):
            with self.subTest(tile=tile):
                self.assertEqual(
                    state._ranking_awards(
                        [neutral + 1, neutral - 1],
                        neutral,
                    ),
                    [18.0, 6.0],
                )
                self.assertEqual(
                    state._ranking_awards(
                        [neutral, neutral - 1],
                        neutral,
                    ),
                    [15.0, 6.0],
                )

    def test_final_research_and_combined_resource_scoring(self) -> None:
        state = GaiaState.initial(2, seed=101)
        players = list(state.players)
        players[0] = replace(
            players[0],
            credits=0,
            ore=0,
            knowledge=0,
            qic=0,
            tracks=(0, 0, 0, 0, 0, 0),
        )
        baseline = replace(state, players=tuple(players)).final_scores()[0]
        players[0] = replace(
            players[0],
            credits=2,
            ore=3,
            knowledge=4,
            qic=9,
            tracks=(5, 4, 3, 2, 1, 0),
        )
        scored = replace(state, players=tuple(players)).final_scores()[0]

        self.assertEqual(scored - baseline, 27)


if __name__ == "__main__":
    unittest.main()
