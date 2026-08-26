import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from gaiazero.cli import build_parser, command_train_all


class CliTrainingSplitTests(unittest.TestCase):
    def test_multiplayer_pipeline_uses_native_npz_and_pytorch_settings(self) -> None:
        args = build_parser().parse_args(["pipeline", "--players", "3"])

        self.assertEqual(args.players, 3)
        self.assertEqual(args.root, "runs/multiplayer-pipeline")
        self.assertEqual(args.shuffle_pack_size, 4096)
        self.assertEqual(args.device, "auto")

    def test_train_all_defaults_to_three_player_counts(self) -> None:
        args = build_parser().parse_args(["train-all"])

        self.assertEqual(tuple(args.player_counts), (2, 3, 4))
        self.assertEqual(args.ruleset, "standard")
        self.assertEqual(args.output_dir, "runs/models")
        self.assertEqual(args.metrics_dir, "runs/metrics-by-players")

    def test_train_all_dispatches_independent_models(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "models"
            metrics_dir = Path(temporary) / "metrics"
            args = build_parser().parse_args([
                "train-all",
                "--output-dir",
                str(output_dir),
                "--metrics-dir",
                str(metrics_dir),
                "--seed",
                "11",
            ])

            with patch("gaiazero.cli.command_train") as train:
                command_train_all(args)

            self.assertEqual(train.call_count, 3)
            children = [call.args[0] for call in train.call_args_list]
            self.assertEqual([child.players for child in children], [2, 3, 4])
            self.assertEqual(
                [Path(child.output).name for child in children],
                [
                    "gaia-standard-2p-nnue.pt",
                    "gaia-standard-3p-katago.pt",
                    "gaia-standard-4p-katago.pt",
                ],
            )
            self.assertEqual(
                [Path(child.metrics).name for child in children],
                [
                    "metrics-standard-2p-nnue.jsonl",
                    "metrics-standard-3p-katago.jsonl",
                    "metrics-standard-4p-katago.jsonl",
                ],
            )
            self.assertEqual([child.seed for child in children], [11, 1_000_014, 2_000_017])
            self.assertTrue(output_dir.is_dir())
            self.assertTrue(metrics_dir.is_dir())


if __name__ == "__main__":
    unittest.main()
