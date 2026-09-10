from __future__ import annotations

import hashlib
import json
import subprocess
import unittest
from pathlib import Path

import numpy as np

from gaiazero.config import GaiaTrainingConfig
from gaiazero.contracts import (
    ACTION_TUPLE_SCHEMA_VERSION,
    ACTION_TYPE_IDS,
    MAX_ACTION_ARGUMENTS,
    NPZ_TRAJECTORY_SCHEMA_VERSION,
    RULES_VERSION,
    STATE_HASH_VERSION,
)
from gaiazero.game import GaiaState
from gaiazero.gnn import graph_inputs_from_state


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "build" / "cpp-msvc" / "gaiazero_graph_probe.exe"
FIXTURE = ROOT / "tests" / "fixtures" / "standard-v22-contract.json"
BASELINE = ROOT / "tests" / "fixtures" / "step0_cpu_baseline.json"
ACCEPTANCE = ROOT / "tests" / "fixtures" / "step0-parity-acceptance.json"


def _digest(tensors: tuple) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        array = tensor.detach().cpu().numpy()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


class Step0ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
        cls.config = GaiaTrainingConfig.load()

    def test_contract_manifest_matches_source_and_training_config(self) -> None:
        fixture = self.fixture
        self.assertEqual(fixture["schema_version"], "standard-v22-contract-v1")
        self.assertEqual(fixture["rules_version"], RULES_VERSION)
        self.assertEqual(fixture["action"]["schema_version"], ACTION_TUPLE_SCHEMA_VERSION)
        self.assertEqual(fixture["action"]["action_type_count"], len(ACTION_TYPE_IDS))
        self.assertEqual(fixture["action"]["max_parameter_slots"], MAX_ACTION_ARGUMENTS)
        self.assertEqual(fixture["trajectory"]["schema_version"], NPZ_TRAJECTORY_SCHEMA_VERSION)
        self.assertEqual(fixture["trajectory"]["state_hash_version"], STATE_HASH_VERSION)
        self.assertEqual(
            self.config.data["network"]["heads"],
            ["parameterized_policy", "pairwise_wdl", "vp_belief"],
        )
        self.assertEqual(self.config.data["action_schema"]["version"], ACTION_TUPLE_SCHEMA_VERSION)
        self.assertEqual(self.config.data["observation_schema"]["version"], RULES_VERSION)
        self.assertEqual(self.config.data["action_schema"]["parameter_slot_count"], MAX_ACTION_ARGUMENTS)

    def test_observation_and_network_shapes_are_frozen_for_each_player_count(self) -> None:
        expected = self.fixture["observation"]["shape_by_player_count"]
        for players in (2, 3, 4):
            with self.subTest(players=players):
                state = GaiaState.initial(players, seed=0)
                self.assertEqual([state.observation_size], expected[str(players)])
                graph = self.config.graph_network_config(players)
                self.assertEqual(graph.action_type_count, 54)
                self.assertEqual(graph.parameter_slot_count, 8)
                self.assertEqual(graph.vp_buckets, 403)

    def test_cpu_baseline_is_recorded_without_claiming_gpu_parity(self) -> None:
        baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
        self.assertEqual(baseline["schema_version"], "step0-cpu-baseline-v1")
        self.assertEqual(baseline["environment"]["device"], "cpu")
        measurements = baseline["measurements"]
        for key in (
            "mcts8_seconds",
            "selfplay_one_game_seconds",
            "gnn_batch1_seconds",
            "gnn_batch4_seconds",
        ):
            self.assertGreater(float(measurements[key]), 0.0)
        self.assertIsNone(baseline["environment"]["gpu_memory_bytes"])

    def test_full_parity_acceptance_result_is_recorded(self) -> None:
        acceptance = json.loads(ACCEPTANCE.read_text(encoding="utf-8"))
        self.assertEqual(acceptance["schema_version"], "cpp-python-parity-acceptance-v1")
        self.assertEqual(acceptance["rules_version"], RULES_VERSION)
        self.assertEqual(acceptance["setup_fixtures"], 1024)
        self.assertEqual(acceptance["trace_limit"], 2000)
        self.assertEqual(
            [(row["players"], row["seed"], row["steps"]) for row in acceptance["traces"]],
            [(2, 0, 57), (2, 17, 60), (3, 0, 102), (3, 4, 97), (4, 17, 127), (4, 8, 160)],
        )

    @unittest.skipUnless(PROBE.exists(), "C++ graph probe has not been built")
    def test_cpp_python_graph_batch_matches_byte_for_byte_golden(self) -> None:
        expected = self.fixture["graph_golden"]["digests"]
        seeds = self.fixture["graph_golden"]["seeds"]
        for players in (2, 3, 4):
            graph_config = self.config.graph_network_config(players)
            for seed in seeds:
                with self.subTest(players=players, seed=seed):
                    state = GaiaState.initial(players, seed)
                    python_digest = _digest(graph_inputs_from_state(state, graph_config))
                    self.assertEqual(python_digest, expected[f"{players}/{seed}"])
                    completed = subprocess.run(
                        [str(PROBE), str(players), str(seed)],
                        cwd=ROOT,
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    values = dict(line.split("=", 1) for line in completed.stdout.splitlines())
                    self.assertEqual(values["digest"], python_digest)
                    self.assertEqual(values["shape"].split(","), ["1", "128", "512", str(players), "16", "16", "16"])


if __name__ == "__main__":
    unittest.main()
