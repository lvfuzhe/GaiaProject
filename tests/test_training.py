from dataclasses import asdict
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from gaiazero.game import MiniGaiaHeuristicEvaluator, MiniGaiaState
from gaiazero.mcts import SearchConfig
from gaiazero.model import (
    KATAGO_ARCHITECTURE,
    LEGACY_ARCHITECTURE,
    NNUE_ARCHITECTURE,
    KataGoPolicyValueBackbone,
    LegacyPolicyValueBackbone,
    NetworkConfig,
    NetworkEvaluator,
    NNUEPolicyValueBackbone,
    PolicyValueNetwork,
    architecture_for_players,
    load_checkpoint,
    save_checkpoint,
)
from gaiazero.replay import ReplayBuffer
from gaiazero.selfplay import SelfPlayConfig, play_self_game
from gaiazero.training import AlphaZeroTrainer, TrainerConfig


class TrainingPipelineTests(unittest.TestCase):
    def test_player_count_selects_nnue_or_katago_architecture(self) -> None:
        expected = {
            2: (NNUE_ARCHITECTURE, NNUEPolicyValueBackbone),
            3: (KATAGO_ARCHITECTURE, KataGoPolicyValueBackbone),
            4: (KATAGO_ARCHITECTURE, KataGoPolicyValueBackbone),
        }
        for players, (architecture, backbone) in expected.items():
            with self.subTest(players=players):
                config = NetworkConfig(
                    observation_size=24,
                    action_size=11,
                    num_players=players,
                    hidden_size=16,
                    residual_blocks=1,
                )
                model = PolicyValueNetwork(config)
                logits, values = model(torch.zeros((3, 24), dtype=torch.float32))

                self.assertEqual(architecture_for_players(players), architecture)
                self.assertEqual(model.architecture, architecture)
                self.assertIsInstance(model.network, backbone)
                self.assertEqual(logits.shape, (3, 11))
                self.assertEqual(values.shape, (3, players))
                if players == 2:
                    self.assertTrue(torch.allclose(values[:, 0], -values[:, 1]))

    def test_wrong_architecture_for_player_count_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "2-player models require nnue"):
            NetworkConfig(
                observation_size=24,
                action_size=11,
                num_players=2,
                hidden_size=16,
                residual_blocks=1,
                architecture=KATAGO_ARCHITECTURE,
            )

    def test_legacy_checkpoint_remains_readable_but_marked_legacy(self) -> None:
        config = NetworkConfig(
            observation_size=24,
            action_size=11,
            num_players=2,
            hidden_size=16,
            residual_blocks=1,
            architecture=LEGACY_ARCHITECTURE,
        )
        legacy = LegacyPolicyValueBackbone(config)
        stored_config = asdict(config)
        stored_config.pop("architecture")
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "legacy.pt"
            torch.save(
                {
                    "network_config": stored_config,
                    "model_state": legacy.state_dict(),
                    "metadata": {"generation": "legacy"},
                },
                checkpoint,
            )
            restored, metadata = load_checkpoint(checkpoint, "cpu")

        self.assertEqual(restored.architecture, LEGACY_ARCHITECTURE)
        self.assertEqual(metadata, {"generation": "legacy"})

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
