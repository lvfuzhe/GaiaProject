import tempfile
import unittest
from pathlib import Path

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
from gaiazero.replay import TrainingExample


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


if __name__ == "__main__":
    unittest.main()
