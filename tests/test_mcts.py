import unittest

import numpy as np

from gaiazero.game import MiniGaiaHeuristicEvaluator, MiniGaiaState
from gaiazero.mcts import PUCTSearch, SearchConfig


class PUCTSearchTests(unittest.TestCase):
    def test_policy_contains_only_legal_actions(self) -> None:
        state = MiniGaiaState.initial(num_players=2)
        result = PUCTSearch(
            MiniGaiaHeuristicEvaluator(),
            SearchConfig(simulations=24, seed=4),
        ).run(state)

        self.assertAlmostEqual(float(result.policy.sum()), 1.0, places=6)
        self.assertEqual(int(result.visits.sum()), 24)
        self.assertEqual(result.root_value.shape, (2,))
        self.assertTrue(np.all(result.policy[~state.legal_action_mask()] == 0))

    def test_root_noise_preserves_legality_and_normalization(self) -> None:
        state = MiniGaiaState.initial(num_players=3)
        result = PUCTSearch(
            MiniGaiaHeuristicEvaluator(),
            SearchConfig(simulations=12, root_noise_fraction=0.5, seed=8),
        ).run(state, add_root_noise=True)

        self.assertAlmostEqual(float(result.policy.sum()), 1.0, places=6)
        self.assertTrue(np.all(result.policy[~state.legal_action_mask()] == 0))
        self.assertEqual(result.root_value.shape, (3,))

    def test_zero_temperature_is_greedy(self) -> None:
        state = MiniGaiaState.initial(num_players=2)
        result = PUCTSearch(
            MiniGaiaHeuristicEvaluator(),
            SearchConfig(simulations=16),
        ).run(state, temperature=0.0)

        self.assertEqual(int(np.count_nonzero(result.policy)), 1)
        self.assertEqual(int(np.argmax(result.policy)), int(np.argmax(result.visits)))


if __name__ == "__main__":
    unittest.main()

