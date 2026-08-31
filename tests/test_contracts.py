import tempfile
import unittest
from pathlib import Path

import numpy as np

from gaiazero.contracts import (
    ACTION_TUPLE_SCHEMA_VERSION,
    NPZ_TRAJECTORY_SCHEMA_VERSION,
    RULES_VERSION,
    STATE_HASH_VERSION,
    ActionTuple,
    canonical_json,
    compose_parameterized_policy,
)
from gaiazero.distributed import read_npz_trajectory, write_npz_shard
from gaiazero.game import GaiaState
from gaiazero.npz_history import convert_npz_to_history
from gaiazero.replay import TrainingExample


class ContractTests(unittest.TestCase):
    def _example(self) -> TrainingExample:
        return TrainingExample(
            observation=np.zeros(4, dtype=np.float32),
            legal_mask=np.asarray([True, False], dtype=np.bool_),
            policy_target=np.asarray([1.0, 0.0], dtype=np.float32),
            value_target=np.asarray([0.0, 0.0], dtype=np.float32),
        )

    def test_action_tuple_round_trip_for_legal_actions(self) -> None:
        state = GaiaState.initial(num_players=2, seed=17)
        tuples = state.legal_action_tuples()
        self.assertEqual(len(tuples), len(state.legal_actions()))
        self.assertTrue(all(item.schema_version == ACTION_TUPLE_SCHEMA_VERSION for item in tuples))
        self.assertEqual(
            [state.action_from_tuple(item) for item in tuples],
            list(state.legal_actions()),
        )
        self.assertEqual(
            ActionTuple.from_json(tuples[0].to_json()),
            tuples[0],
        )
        # Every compatibility slot has a deterministic semantic fallback so
        # logs can be migrated even when a slot is not legal in this position.
        from gaiazero.game.gaia_state import ACTION_SIZE

        for action_id in range(ACTION_SIZE):
            self.assertEqual(
                state.action_from_tuple(state.action_tuple(action_id)), action_id
            )

    def test_state_hash_covers_state_and_is_in_snapshot(self) -> None:
        first = GaiaState.initial(num_players=2, seed=23)
        second = GaiaState.initial(num_players=2, seed=23)
        self.assertEqual(first.state_hash(), second.state_hash())
        self.assertEqual(first.snapshot()["state_hash_version"], STATE_HASH_VERSION)
        self.assertEqual(first.snapshot()["state_hash"], first.state_hash())
        changed = first.apply(first.legal_actions()[0])
        self.assertNotEqual(first.state_hash(), changed.state_hash())

    def test_parameterized_policy_composes_only_legal_tuples(self) -> None:
        actions = (ActionTuple("build_mine", (2,)), ActionTuple("pass_final"))
        type_logits = np.zeros(55, dtype=np.float32)
        type_logits[actions[0].action_type_id] = 1.0
        argument_logits = np.zeros((8, 8), dtype=np.float32)
        argument_logits[0, 2] = 2.0
        probabilities = compose_parameterized_policy(
            type_logits, argument_logits, actions
        )
        self.assertEqual(probabilities.shape, (2,))
        self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=6)
        self.assertGreater(float(probabilities[0]), float(probabilities[1]))

    def test_trajectory_npz_contains_terminal_row_and_can_be_materialized(self) -> None:
        state = GaiaState.initial(num_players=2, seed=31)
        action = state.legal_actions()[0]
        terminal = state.apply(action)
        trajectory = (
            {
                "position_index": 0,
                "semantic_turn_index": 0,
                "round": state.round_number,
                "player_to_move": state.current_player,
                "action_id": action,
                "action_tuple": state.action_tuple(action).to_dict(),
                "state_hash": state.state_hash(),
                "state_json": canonical_json(state),
                "state": state.snapshot(),
            },
            {
                "position_index": 1,
                "semantic_turn_index": 1,
                "round": terminal.round_number,
                "player_to_move": terminal.current_player,
                "action_id": None,
                "action_tuple": None,
                "state_hash": terminal.state_hash(),
                "state_json": canonical_json(terminal),
                "state": terminal.snapshot(),
            },
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "game.npz"
            write_npz_shard(
                source,
                [self._example()],
                {"game_id": "contract-game", "rules_version": RULES_VERSION},
                trajectory=trajectory,
            )
            decoded = read_npz_trajectory(source)
            self.assertEqual(
                decoded["metadata"]["schema_version"],
                NPZ_TRAJECTORY_SCHEMA_VERSION,
            )
            self.assertEqual(len(decoded["position_index"]), 2)
            self.assertEqual(int(decoded["action_ids"][-1]), -1)
            history = convert_npz_to_history(source, root / "history")
            self.assertTrue(history.is_file())


if __name__ == "__main__":
    unittest.main()
