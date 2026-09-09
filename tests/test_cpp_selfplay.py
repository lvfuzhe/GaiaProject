from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from gaiazero.distributed import read_npz_shard, read_npz_trajectory


class CppSelfplayTests(unittest.TestCase):
    def test_cpp_selfplay_npz_is_python_compatible(self) -> None:
        executable = Path(
            os.environ.get(
                "GAIAZERO_CPP_SELFPLAY",
                "build/cpp-msvc/gaiazero_selfplay.exe",
            )
        )
        if not executable.is_file():
            self.skipTest(f"C++ selfplay executable is not built: {executable}")
        with tempfile.TemporaryDirectory() as temporary:
            subprocess.run(
                [
                    str(executable.resolve()),
                    "--players",
                    "2",
                    "--games",
                    "1",
                    "--simulations",
                    "1",
                    "--leaf-batch-size",
                    "4",
                    "--max-moves",
                    "180",
                    "--output",
                    temporary,
                    "--no-root-noise",
                    "--once",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
            )
            shard = next(Path(temporary).glob("*.npz"))
            examples, metadata = read_npz_shard(shard)
            trajectory = read_npz_trajectory(shard)
            with np.load(shard, allow_pickle=False) as values:
                self.assertEqual(
                    values["pairwise_wdl_targets"].shape,
                    (len(examples), 2, 2, 3),
                )
                self.assertEqual(
                    values["final_utility_targets"].shape,
                    (len(examples), 2),
                )
                self.assertEqual(
                    values["final_vp_belief_targets"].shape,
                    (len(examples), 2, 403),
                )
                self.assertTrue(
                    np.allclose(values["pairwise_wdl_targets"][:, 0, 1].sum(axis=-1), 1.0)
                )
                self.assertTrue(
                    np.allclose(values["final_vp_belief_targets"].sum(axis=-1), 1.0)
                )

        self.assertTrue(examples)
        self.assertEqual(metadata["schema_version"], "npz-trajectory-v1")
        self.assertEqual(len(trajectory["position_index"]), len(examples) + 1)
        self.assertTrue(trajectory["terminal_valid"])
        self.assertEqual(examples[0].observation.shape, (4127,))
        self.assertEqual(examples[0].legal_mask.shape, (971,))
        self.assertAlmostEqual(float(np.sum(examples[0].policy_target)), 1.0)

    def test_cpp_selfplay_thread_pool_writes_one_shard_per_game(self) -> None:
        executable = Path(
            os.environ.get(
                "GAIAZERO_CPP_SELFPLAY",
                "build/cpp-msvc/gaiazero_selfplay.exe",
            )
        )
        if not executable.is_file():
            self.skipTest(f"C++ selfplay executable is not built: {executable}")
        with tempfile.TemporaryDirectory() as temporary:
            subprocess.run(
                [
                    str(executable.resolve()),
                    "--players", "2",
                    "--games", "2",
                    "--threads", "2",
                    "--simulations", "1",
                    "--leaf-batch-size", "2",
                    "--max-moves", "180",
                    "--output", temporary,
                    "--no-root-noise",
                    "--once",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=90,
            )
            shards = list(Path(temporary).glob("*.npz"))
            self.assertEqual(len(shards), 2)
            for shard in shards:
                examples, metadata = read_npz_shard(shard)
                self.assertTrue(examples)
                self.assertEqual(metadata["schema_version"], "npz-trajectory-v1")


if __name__ == "__main__":
    unittest.main()
