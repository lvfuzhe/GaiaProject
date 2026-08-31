import json
import tempfile
import unittest
from pathlib import Path

from gaiazero.config import GaiaTrainingConfig, load_training_config
from gaiazero.contracts import ACTION_TUPLE_SCHEMA_VERSION, RULES_VERSION
from gaiazero.distributed import load_pipeline_config


class TrainingConfigLoaderTests(unittest.TestCase):
    def test_repository_training_config_controls_pipeline_and_schema(self) -> None:
        config = load_training_config()
        self.assertEqual(config.rules_version, RULES_VERSION)
        self.assertEqual(config.action_schema_version, ACTION_TUPLE_SCHEMA_VERSION)
        self.assertEqual(config.seed_stream_version, "setup-seed-stream-v1")
        self.assertEqual(config.network_capacity["hidden_size"], 256)
        self.assertEqual(config.profile(3)["network_id"], "graph-hybrid-3p")

        pipeline = config.pipeline_config(3)
        self.assertEqual(pipeline.players, 3)
        self.assertEqual(pipeline.seed, 0)
        self.assertEqual(pipeline.simulations, 128)
        self.assertEqual(pipeline.batch_size, 128)
        self.assertEqual(pipeline.seed_stream_version, "setup-seed-stream-v1")
        self.assertEqual(pipeline.network_config_id, "graph-hybrid-3p")
        self.assertEqual(pipeline.training_config_hash, config.config_hash)
        self.assertEqual(pipeline.poll_seconds, 10.0)

    def test_loader_rejects_schema_drift(self) -> None:
        source = Path(__file__).parents[1] / "configs" / "gaia-training.json"
        with tempfile.TemporaryDirectory() as temporary:
            payload = json.loads(source.read_text(encoding="utf-8"))
            payload["action_schema"]["version"] = "action-tuple-v0"
            path = Path(temporary) / "bad.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "action_schema.version"):
                GaiaTrainingConfig.load(path)

    def test_explicit_pipeline_override_is_limited_to_runtime_fields(self) -> None:
        config = load_training_config()
        pipeline = config.pipeline_config(
            2,
            root="runs/test-config",
            seed=42,
            overrides={"simulations": 7},
        )
        self.assertEqual(pipeline.root, Path("runs/test-config"))
        self.assertEqual(pipeline.seed, 42)
        self.assertEqual(pipeline.simulations, 7)
        with self.assertRaisesRegex(ValueError, "unknown PipelineConfig override"):
            config.pipeline_values(2, overrides={"unknown": 1})

    def test_pipeline_loader_accepts_training_document_directly(self) -> None:
        pipeline = load_pipeline_config(
            Path(__file__).parents[1] / "configs" / "gaia-training.json",
            players=4,
        )
        self.assertEqual(pipeline.players, 4)
        self.assertEqual(pipeline.seed_stream_version, "setup-seed-stream-v1")
        self.assertEqual(pipeline.network_config_id, "graph-hybrid-4p")


if __name__ == "__main__":
    unittest.main()
