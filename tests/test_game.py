import unittest

import numpy as np

from gaiazero.game import GaiaState


class GaiaStateTests(unittest.TestCase):
    def test_initial_state_contract(self) -> None:
        state = GaiaState.initial(num_players=2, seed=1)

        self.assertFalse(state.is_terminal)
        self.assertTrue(state.is_starting_placement)
        self.assertEqual(state.legal_action_mask().shape, (state.action_size,))
        self.assertEqual(state.observation().shape, (state.observation_size,))
        self.assertEqual(int(state.legal_action_mask().sum()), len(state.legal_actions()))
        self.assertEqual(state.snapshot()["ruleset"], "standard-v22")

    def test_legal_action_application_is_immutable(self) -> None:
        state = GaiaState.initial(num_players=2, seed=2)
        action = state.legal_actions()[0]
        next_state = state.apply(action)

        self.assertEqual(state.snapshot()["round"], 0)
        self.assertEqual(next_state.snapshot()["ruleset"], "standard-v22")
        self.assertFalse(np.array_equal(state.observation(), next_state.observation()))

    def test_same_seed_has_same_setup_and_encoding(self) -> None:
        first = GaiaState.initial(num_players=4, seed=9)
        second = GaiaState.initial(num_players=4, seed=9)

        self.assertEqual(first.sector_tiles, second.sector_tiles)
        self.assertEqual(first.sector_rotations, second.sector_rotations)
        np.testing.assert_array_equal(first.observation(), second.observation())


if __name__ == "__main__":
    unittest.main()
