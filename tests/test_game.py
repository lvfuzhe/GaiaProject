import unittest

import numpy as np

from gaiazero.game.mini_gaia import MAX_ROUNDS, MiniGaiaState


class MiniGaiaStateTests(unittest.TestCase):
    def test_initial_state_contract(self) -> None:
        state = MiniGaiaState.initial(num_players=2, seed=1)

        self.assertEqual(state.current_player, 1)
        self.assertFalse(state.is_terminal)
        self.assertIn(state.pass_action, state.legal_actions())
        self.assertEqual(state.legal_action_mask().shape, (state.action_size,))
        self.assertEqual(state.observation().shape, (state.observation_size,))
        self.assertEqual(int(state.legal_action_mask().sum()), len(state.legal_actions()))

    def test_build_is_immutable_and_advances_turn(self) -> None:
        state = MiniGaiaState.initial(num_players=2)
        action = next(action for action in state.legal_actions() if action < 19)
        destination = action

        next_state = state.apply(action)

        self.assertEqual(state.owners[destination], -1)
        self.assertEqual(next_state.owners[destination], state.current_player)
        self.assertNotEqual(next_state.current_player, state.current_player)

    def test_first_player_to_pass_starts_next_round(self) -> None:
        state = MiniGaiaState.initial(num_players=2, seed=0)
        state = state.apply(state.pass_action)
        self.assertEqual(state.current_player, 1)
        state = state.apply(state.pass_action)

        self.assertEqual(state.round_number, 2)
        self.assertEqual(state.first_player, 0)
        self.assertEqual(state.current_player, 0)
        self.assertTrue(all(not player.passed for player in state.players))

    def test_passing_ends_the_game_after_six_rounds(self) -> None:
        state = MiniGaiaState.initial(num_players=3)
        while not state.is_terminal:
            state = state.apply(state.pass_action)

        self.assertEqual(state.round_number, MAX_ROUNDS + 1)
        returns = state.returns()
        self.assertEqual(returns.shape, (3,))
        self.assertAlmostEqual(float(returns.sum()), 0.0, places=6)
        self.assertTrue(np.all(returns >= -1.0))
        self.assertTrue(np.all(returns <= 1.0))

    def test_same_seed_has_same_encoding(self) -> None:
        first = MiniGaiaState.initial(num_players=4, seed=9)
        second = MiniGaiaState.initial(num_players=4, seed=9)
        np.testing.assert_array_equal(first.observation(), second.observation())
        self.assertEqual(first.legal_actions(), second.legal_actions())


if __name__ == "__main__":
    unittest.main()

