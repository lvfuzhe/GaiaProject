import unittest
from dataclasses import replace

import numpy as np

from gaiazero.game.gaia_state import (
    FEDERATION_ACTION,
    Building,
    GaiaHeuristicEvaluator,
    GaiaState,
    PlayerState,
    Terrain,
    Track,
)
from gaiazero.mcts import PUCTSearch, SearchConfig


class StandardGaiaRulesTests(unittest.TestCase):
    def test_setup_has_full_research_and_building_model(self) -> None:
        state = GaiaState.initial(4, seed=3)

        self.assertEqual(state.action_size, 138)
        self.assertEqual(len(state.players[0].tracks), 6)
        self.assertEqual(state.current_player, 3)
        self.assertEqual(sum(owner >= 0 for owner in state.owners), 8)
        self.assertEqual(state.snapshot()["ruleset"], "standard-v2")

    def test_power_charges_bowl_one_before_bowl_two(self) -> None:
        info = PlayerState(faction=0, bowl_one=1, bowl_two=2, bowl_three=0)

        charged, amount = GaiaState._charge_power(info, 4)

        self.assertEqual(amount, 4)
        self.assertEqual((charged.bowl_one, charged.bowl_two, charged.bowl_three), (0, 0, 3))
        spent = GaiaState._spend_power(charged, 2)
        self.assertEqual((spent.bowl_one, spent.bowl_two, spent.bowl_three), (2, 0, 1))

    def test_gaia_project_transforms_then_returns_gaiaformer(self) -> None:
        state = GaiaState.initial(2)
        action = state.gaia_action(6)
        self.assertIn(action, state.legal_actions())

        started = state.apply(action)
        self.assertEqual(started.gaiaformer_owner[6], 0)
        self.assertEqual(started.players[0].gaia_power, 6)
        transformed = started._gaia_phase()
        self.assertEqual(transformed.terrains[6], Terrain.GAIA)
        self.assertEqual(transformed.players[0].gaia_power, 0)
        self.assertEqual(transformed.players[0].bowl_two, 8)

        transformed = replace(transformed, player_to_move=0)
        built = transformed.apply(transformed.build_action(6))
        self.assertEqual(built.owners[6], 0)
        self.assertEqual(built.gaiaformer_owner[6], -1)
        self.assertEqual(built.players[0].gaiaformers, 1)

    def test_research_lab_requires_immediate_tech_choice(self) -> None:
        state = GaiaState.initial(2)
        state = state.apply(state.upgrade_trading_action(0))
        players = list(state.players)
        players[0] = replace(players[0], credits=30, ore=15)
        state = replace(state, player_to_move=0, players=tuple(players))

        pending = state.apply(state.upgrade_lab_action(0))

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
        buildings = list(state.buildings)
        buildings[0] = Building.TRADING_STATION
        players = list(state.players)
        players[0] = replace(players[0], credits=30, ore=15, tracks=(5, 5, 5, 5, 5, 5))
        state = replace(
            state,
            player_to_move=0,
            players=tuple(players),
            buildings=tuple(int(value) for value in buildings),
        )

        self.assertNotIn(state.upgrade_lab_action(0), state.legal_actions())

    def test_canonical_federation_awards_token(self) -> None:
        state = GaiaState.initial(2)
        owners = list(state.owners)
        buildings = list(state.buildings)
        owners[1] = 0
        buildings[0] = Building.PLANETARY_INSTITUTE
        buildings[7] = Building.ACADEMY
        buildings[1] = Building.MINE
        state = replace(state, owners=tuple(owners), buildings=tuple(int(value) for value in buildings))

        self.assertIn(FEDERATION_ACTION, state.legal_actions())
        formed = state.apply(FEDERATION_ACTION)
        self.assertEqual(formed.players[0].federation_tokens, 1)
        self.assertEqual(formed.players[0].federation_keys, 1)
        self.assertTrue(formed.federated[0])
        self.assertTrue(formed.federated[1])
        self.assertTrue(formed.federated[7])

    def test_passing_returns_old_booster_and_takes_new_one(self) -> None:
        state = GaiaState.initial(2)
        passed = state.apply(state.pass_booster_action(2))

        self.assertEqual(passed.booster_owner[0], -1)
        self.assertEqual(passed.booster_owner[2], 0)
        self.assertTrue(passed.players[0].passed)

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
