import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from gaiazero.gnn import GraphHybridNetwork, GraphNetworkConfig
from gaiazero.gnn import graph_inputs_from_state, load_graph_checkpoint, save_graph_checkpoint
from gaiazero.game import GaiaState
from gaiazero.onnx_export import export_graph_onnx, verify_onnx_cpu_golden
from gaiazero.swa import SWAAccumulator, SWAConfig
from gaiazero.model import load_checkpoint_swa, save_checkpoint


def _digest(tensors: tuple[torch.Tensor, ...]) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        array = tensor.detach().cpu().numpy()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


class GnnAndOnnxTests(unittest.TestCase):
    def _golden(self) -> tuple[dict, GraphHybridNetwork, tuple[torch.Tensor, ...]]:
        fixture = json.loads(
            (Path(__file__).parent / "fixtures" / "gnn_cpu_golden.json").read_text(
                encoding="utf-8"
            )
        )
        config = GraphNetworkConfig(**fixture["config"])
        torch.manual_seed(fixture["torch_model_seed"])
        model = GraphHybridNetwork(config).eval()
        torch.manual_seed(fixture["torch_input_seed"])
        inputs = (
            torch.randn(2, config.max_graph_nodes, config.node_feature_size),
            torch.randint(
                0,
                config.max_graph_nodes,
                (2, config.max_graph_edges, 2),
                dtype=torch.int64,
            ),
            torch.randint(
                0,
                config.relation_type_count,
                (2, config.max_graph_edges),
                dtype=torch.int64,
            ),
            torch.ones(2, config.max_graph_edges),
            torch.ones(2, config.max_graph_nodes),
            torch.randn(2, config.global_feature_size),
            torch.randn(2, config.num_players, config.player_feature_size),
            torch.ones(2, config.num_players),
        )
        return fixture, model, inputs

    def test_cpu_gnn_matches_golden_fixture(self) -> None:
        fixture, model, inputs = self._golden()
        with torch.inference_mode():
            outputs = tuple(model(*inputs))
        self.assertEqual([list(value.shape) for value in outputs], fixture["output_shapes"])
        self.assertEqual(_digest(outputs), fixture["torch_output_digest"])

    def test_onnx_export_matches_cpu_golden(self) -> None:
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            self.skipTest("onnxruntime is not installed")
        fixture, model, inputs = self._golden()
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "graph.onnx"
            export_graph_onnx(model, destination, inputs)
            result = verify_onnx_cpu_golden(
                model,
                destination,
                inputs,
                atol=fixture["max_abs_error"],
                rtol=fixture["max_relative_error"],
            )
        self.assertTrue(result["passed"], result)
        self.assertLessEqual(result["max_abs_error"], fixture["max_abs_error"])

    def test_swa_rolling_average_and_state_round_trip(self) -> None:
        model = torch.nn.Linear(3, 2)
        config = SWAConfig(
            enabled=True,
            start_after_samples=2,
            update_every_samples=2,
            snapshot_count=2,
            device="cpu",
        )
        accumulator = SWAAccumulator(config)
        with torch.no_grad():
            model.weight.fill_(1.0)
        self.assertFalse(accumulator.maybe_update(model, 1))
        self.assertTrue(accumulator.maybe_update(model, 2))
        with torch.no_grad():
            model.weight.fill_(3.0)
        self.assertTrue(accumulator.maybe_update(model, 4))
        target = torch.nn.Linear(3, 2)
        accumulator.copy_to(target)
        self.assertTrue(torch.allclose(target.weight, torch.full_like(target.weight, 2.0)))
        restored = SWAAccumulator(config)
        restored.load_state_dict(accumulator.state_dict())
        self.assertEqual(restored.snapshot_count, 2)
        self.assertTrue(torch.allclose(restored.averaged_state_dict()["weight"], accumulator.averaged_state_dict()["weight"]))

    def test_policy_checkpoint_can_persist_swa_state(self) -> None:
        from gaiazero.model import NetworkConfig, PolicyValueNetwork

        model = PolicyValueNetwork(
            NetworkConfig(observation_size=4, action_size=3, num_players=2, hidden_size=16, residual_blocks=1)
        )
        accumulator = SWAAccumulator(SWAConfig(start_after_samples=0, snapshot_count=1))
        accumulator.maybe_update(model, 0)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.pt"
            save_checkpoint(path, model, swa=accumulator, metadata={"line": "golden"})
            restored, metadata, swa_state = load_checkpoint_swa(path, "cpu")
        self.assertEqual(restored.config, model.config)
        self.assertEqual(metadata["line"], "golden")
        self.assertIsNotNone(swa_state)
        self.assertEqual(len(swa_state["snapshots"]), 1)

    def test_gaia_state_adapter_and_graph_checkpoint_round_trip(self) -> None:
        state = GaiaState.initial(num_players=3, seed=4)
        config = GraphNetworkConfig(
            num_players=3,
            node_feature_size=16,
            global_feature_size=16,
            player_feature_size=16,
            max_graph_nodes=128,
            max_graph_edges=512,
            hidden_size=16,
            hybrid_blocks=1,
            ffn_hidden_size=32,
        )
        model = GraphHybridNetwork(config).eval()
        inputs = graph_inputs_from_state(state, config)
        with torch.inference_mode():
            expected = tuple(model(*inputs))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "graph.pt"
            save_graph_checkpoint(path, model, metadata={"golden": True})
            restored, metadata, swa_state = load_graph_checkpoint(path)
        self.assertEqual(metadata["golden"], True)
        self.assertIsNone(swa_state)
        with torch.inference_mode():
            actual = tuple(restored(*inputs))
        for left, right in zip(expected, actual, strict=True):
            self.assertTrue(torch.allclose(left, right))


if __name__ == "__main__":
    unittest.main()
