import unittest
from pathlib import Path

import numpy as np

from gaiazero.game import MiniGaiaHeuristicEvaluator, MiniGaiaState
from gaiazero.mcts import SearchConfig
from gaiazero.model import (
    NetworkConfig,
    NetworkEvaluator,
    PolicyValueNetwork,
    load_checkpoint,
    save_checkpoint,
)
from gaiazero.replay import ReplayBuffer
from gaiazero.selfplay import SelfPlayConfig, play_self_game
from gaiazero.training import AlphaZeroTrainer, TrainerConfig


class TrainingPipelineTests(unittest.TestCase):
    def test_self_play_train_and_checkpoint_round_trip(self) -> None:
        state = MiniGaiaState.initial(num_players=2)
        game = play_self_game(
            state,
            MiniGaiaHeuristicEvaluator(),
            SearchConfig(simulations=2, seed=2),
            SelfPlayConfig(temperature_moves=4, seed=2),
        )
        self.assertTrue(game.final_state.is_terminal)
        self.assertGreater(len(game.examples), 0)

        replay = ReplayBuffer(capacity=2_000, seed=2)
        replay.extend(game.examples)
        model = PolicyValueNetwork(
            NetworkConfig(
                observation_size=state.observation_size,
                action_size=state.action_size,
                num_players=state.num_players,
                hidden_size=32,
                residual_blocks=1,
            )
        )
        trainer = AlphaZeroTrainer(model, TrainerConfig(batch_size=8, device="cpu"))
        metrics = trainer.train_updates(replay, updates=1)
        self.assertTrue(np.isfinite(metrics.loss))

        evaluator = NetworkEvaluator(model, "cpu")
        policy, value = evaluator.evaluate(state)
        self.assertAlmostEqual(float(policy.sum()), 1.0, places=5)
        self.assertEqual(value.shape, (2,))
        self.assertTrue(np.all(policy[~state.legal_action_mask()] == 0))

        checkpoint = Path(__file__).parent / ".artifacts" / "model.pt"
        try:
            save_checkpoint(checkpoint, model, metadata={"test": True})
            restored, metadata = load_checkpoint(checkpoint, "cpu")
            self.assertEqual(restored.config, model.config)
            self.assertEqual(metadata, {"test": True})
        finally:
            checkpoint.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
