import argparse
import unittest

from gaiazero.cli import build_parser


class CliTrainingSplitTests(unittest.TestCase):
    def test_multiplayer_pipeline_uses_native_npz_and_pytorch_settings(self) -> None:
        args = build_parser().parse_args(["pipeline", "--players", "3"])

        self.assertEqual(args.players, 3)
        self.assertEqual(args.root, "runs/multiplayer-pipeline")
        self.assertEqual(args.shuffle_pack_size, 4096)
        self.assertEqual(args.device, "auto")

    def test_synchronous_training_commands_are_removed(self) -> None:
        parser = build_parser()
        subparsers = next(
            action
            for action in parser._actions
            if isinstance(action, argparse._SubParsersAction)
        )

        self.assertNotIn("train", subparsers.choices)
        self.assertNotIn("train-all", subparsers.choices)
        self.assertIn("pipeline", subparsers.choices)

    def test_dashboard_defaults_to_async_pipeline_storage(self) -> None:
        args = build_parser().parse_args(["dashboard"])

        self.assertEqual(args.storage_dir, "runs")
        self.assertEqual(args.pipeline_root, "runs/multiplayer-pipeline")


if __name__ == "__main__":
    unittest.main()
