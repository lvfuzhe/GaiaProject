# standard-v22 Contracts

This document records the Python-side contract implemented in the repository.
It is the compatibility boundary for the future C++/TensorRT self-play and
gatekeeper workers.

## Version identifiers

| Contract | Version |
| --- | --- |
| Rules | `standard-v22` |
| Semantic action tuple | `action-tuple-v1` |
| Raw one-game NPZ | `npz-trajectory-v1` |
| Shuffled training NPZ | `npz-shard-v1` |
| State digest | `state-hash-v1` |

## ActionTuple

`ActionTuple(action_type, args, schema_version)` is the semantic action.  The
argument list is canonicalized in fixed slot order and is limited to eight
integer entity IDs in the current rules (planet, track, tile, or board-space).
`ActionRegistry` supplies stable `action_type_id` values.  `GaiaState` exposes:

```python
state.legal_action_tuples()
state.action_tuple(legacy_id)
state.action_from_tuple(action_tuple)
state.apply_tuple(action_tuple)
```

The integer action array remains only as a compatibility/audit key for the
existing Python MCTS.  A parameterized network can consume the action type and
argument slots and combine logits only for tuples returned by the rules engine.
`compose_parameterized_policy()` implements the reference composition: type
logit plus populated conditional slot logits, followed by a softmax over the
legal tuple list.

## Observation envelope

`GaiaState.contract_metadata()` identifies the observation as `standard-v22`,
`float32`, absolute-seat ordered, and exact to the configured player count.
The current Python baseline exposes the legacy flat vector while the graph
encoder is being integrated; its shape is recorded in every NPZ metadata
envelope so a future GNN cannot silently consume an incompatible tensor.

## State hash

`GaiaState.state_hash()` is SHA-256 over a canonical JSON envelope containing
the rules/hash versions and every dataclass field of `GaiaState` and
`PlayerState`.  Enum values, tuples, NumPy scalars and mappings are normalized
before encoding.  Therefore pending decisions, player order, resources,
technology, map/setup data, satellites and all other rule-relevant fields are
covered automatically.  `snapshot()` includes both `state_hash` and
`state_hash_version` for UI/replay auditing.

## Raw NPZ trajectory

`distributed.write_npz_shard(..., trajectory=...)` writes one game per file.
There must be one pre-action row per training example plus one terminal row.
The terminal row has `action_ids=-1`, an empty action tuple, and a final state
hash.  In addition to the four legacy training arrays, the file contains:

- `state_trace_json` (canonical complete state) and `state_snapshot_json` (UI snapshot)
- `state_hashes`
- `action_tuples_json`, `action_type_ids`, `action_args[trace,8]`, `action_arg_mask[trace,8]`
- `legal_action_tuples_json` and `policy_visit_targets_by_tuple_json`
- `policy_type_targets_json`, `policy_argument_targets_json`,
  `root_visit_counts_by_tuple_json`, `root_policy_priors_by_tuple_json`
- `position_index`, `semantic_turn_index`, `round`, `player_to_move`
- scalar `terminal_valid`

`read_npz_trajectory()` validates schema/version, contiguous positions, terminal
sentinels, array lengths, tuple encodings, and recomputes every state hash.
`read_npz_shard(..., include_metadata=False)` ignores these replay fields, so
shuffle/train do not parse or consume history data.

`npz_history.convert_npz_to_history()` is the explicit materialization boundary:
it converts a validated trajectory into a deletable dashboard JSON copy.
