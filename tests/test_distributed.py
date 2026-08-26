import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from gaiazero.distributed import (
    EXPORT_MAGIC,
    PipelineConfig,
    export_checkpoint,
    load_exported_model,
    pipeline_paths,
    read_export_manifest,
    read_npz_shard,
    run_shuffle,
    write_npz_shard,
)
from gaiazero.model import NetworkConfig, PolicyValueNetwork, save_checkpoint
from gaiazero.npz_history import convert_npz_to_history, delete_training_history
from gaiazero.pipeline_monitor import PipelineSupervisor, WORKER_NAMES
from gaiazero.replay import TrainingExample
from gaiazero.telemetry import build_local_history_index, read_local_game_trace


class DistributedPipelineTests(unittest.TestCase):
    def _example(self, value: float) -> TrainingExample:
        return TrainingExample(
            observation=np.asarray([value, value + 1], dtype=np.float32),
            legal_mask=np.asarray([True, False, True], dtype=np.bool_),
            policy_target=np.asarray([0.25, 0.0, 0.75], dtype=np.float32),
            value_target=np.asarray([value, -value, 0.0], dtype=np.float32),
        )

    def test_shuffle_writes_native_npz_packs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = pipeline_paths(root)
            paths["raw"].mkdir(parents=True)
            write_npz_shard(
                paths["raw"] / "game-1.npz",
                [self._example(1), self._example(2)],
            )
            write_npz_shard(
                paths["raw"] / "game-2.npz",
                [self._example(3), self._example(4)],
            )
            config = PipelineConfig(
                root=root,
                players=3,
                ruleset="mini",
                simulations=1,
                shuffle_pack_size=3,
                min_replay=1,
                batch_size=1,
                updates_per_cycle=1,
                hidden_size=16,
                residual_blocks=1,
                gate_games=1,
            )

            self.assertEqual(run_shuffle(config, once=True), 4)
            shards = sorted(paths["shuffled"].glob("*.npz"))
            self.assertEqual(len(shards), 2)
            examples = []
            for shard in shards:
                loaded, metadata = read_npz_shard(shard)
                examples.extend(loaded)
                self.assertEqual(metadata["kind"], "shuffled-training-pack")
            self.assertEqual(len(examples), 4)
            self.assertFalse(any(root.rglob("*.tfrecord")))
            with self.assertRaisesRegex(ValueError, "complete replay trace"):
                convert_npz_to_history(shards[0], root / "history")

    def test_gaiazero_bin_round_trip(self) -> None:
        config = NetworkConfig(
            observation_size=6,
            action_size=4,
            num_players=3,
            hidden_size=16,
            residual_blocks=1,
        )
        model = PolicyValueNetwork(config)
        observation = torch.randn(2, 6)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "model.pt"
            exported = root / "model.bin"
            save_checkpoint(checkpoint, model, metadata={"generation": 7})
            export_checkpoint(checkpoint, exported)
            content = exported.read_bytes()
            restored, metadata = load_exported_model(exported, "cpu")
            manifest = read_export_manifest(exported)

        self.assertTrue(content.startswith(EXPORT_MAGIC))
        self.assertEqual(manifest["format"], "gaiazero-multiplayer-bin-v1")
        self.assertEqual(metadata["generation"], 7)
        model.eval()
        restored.eval()
        with torch.inference_mode():
            expected = model(observation)
            actual = restored(observation)
        self.assertTrue(torch.allclose(expected[0], actual[0]))
        self.assertTrue(torch.allclose(expected[1], actual[1]))

    def test_npz_history_conversion_is_explicit_and_deletable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "game.npz"
            history = root / "history"
            write_npz_shard(
                source,
                [self._example(1), self._example(2)],
                {
                    "players": 3,
                    "ruleset": "mini",
                    "history": {
                        "steps": [
                            {
                                "move": 0,
                                "player": 0,
                                "action": None,
                                "state": {"ruleset": "mini", "round": 0},
                            },
                            {
                                "move": 1,
                                "player": 0,
                                "action": 3,
                                "state": {"ruleset": "mini", "round": 1},
                            },
                        ],
                        "summary": {"moves": 1, "positions": 2, "scores": [1, 0, 0]},
                    },
                },
            )

            output = convert_npz_to_history(source, history, run_id="npz-test")
            payload = output.read_text(encoding="utf-8")
            self.assertIn('"source":"training_npz"', payload)
            self.assertIn('"move":1', payload)
            index = build_local_history_index(history)
            self.assertEqual(index["runs"][0]["source"], "training_npz")
            trace = read_local_game_trace(history, run_id="npz-test")
            self.assertIsNotNone(trace)
            self.assertEqual(trace["steps"][0]["move"], 0)
            self.assertTrue(delete_training_history(history, "npz-test"))
            self.assertFalse(output.exists())

    def test_dashboard_supervisor_starts_and_stops_all_five_workers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "pipeline"
            processes = []

            def create_process(*args, **kwargs):
                process = MagicMock()
                process.pid = 1000 + len(processes)
                process.poll.return_value = None
                processes.append((args, kwargs, process))
                return process

            supervisor = PipelineSupervisor(root)
            with patch("gaiazero.pipeline_monitor.subprocess.Popen", side_effect=create_process):
                status = supervisor.start({
                    "root": str(root),
                    "players": 3,
                    "ruleset": "mini",
                    "simulations": 1,
                    "device": "cpu",
                })

            self.assertEqual(status["status"], "running")
            self.assertEqual(len(processes), 5)
            commands = [item[0][0] for item in processes]
            self.assertEqual([command[3] for command in commands], list(WORKER_NAMES))
            self.assertTrue((root / "pipeline.json").is_file())

            stopped = supervisor.stop()
            self.assertEqual(stopped["status"], "stopping")
            self.assertTrue((root / "STOP").is_file())
            for _args, _kwargs, process in processes:
                process.terminate.assert_called_once()
                process.poll.return_value = 0
            supervisor.close()


if __name__ == "__main__":
    unittest.main()
