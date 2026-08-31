"""Explicit conversion from GaiaZero self-play NPZ to dashboard history.

This module is intentionally outside the five training workers.  Nothing in
``selfplay``, ``shuffle`` or ``train`` imports it; conversion happens only when
the user invokes the CLI/script explicitly.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from gaiazero.contracts import NPZ_TRAJECTORY_SCHEMA_VERSION
from gaiazero.distributed import read_npz_shard, read_npz_trajectory
from gaiazero.telemetry import delete_local_game, write_local_game


def _history_record(
    source: Path,
    metadata: dict[str, Any],
    *,
    run_id: str,
) -> dict[str, Any]:
    now = datetime.now(UTC).isoformat()
    history = metadata.get("history")
    steps = history.get("steps") if isinstance(history, dict) else None
    summary = history.get("summary") if isinstance(history, dict) else None
    if not isinstance(steps, list) or not steps:
        raise ValueError(
            "NPZ does not contain a complete replay trace; only raw self-play game NPZ files can be converted"
        )
    normalized_steps = []
    for index, raw_step in enumerate(steps):
        if not isinstance(raw_step, dict):
            continue
        step = dict(raw_step)
        step.setdefault("move", index)
        step.setdefault("player", None)
        step.setdefault("action", None)
        step.setdefault("action_label", "training sample")
        if not isinstance(step.get("state"), dict):
            raise ValueError("NPZ replay trace contains a step without a state snapshot")
        normalized_steps.append(step)
    normalized_steps.sort(key=lambda item: int(item.get("move", 0)))
    moves = max(0, len(normalized_steps) - 1)
    if not normalized_steps or int(normalized_steps[0].get("move", -1)) != 0:
        raise ValueError("NPZ replay trace must start with move 0")
    if [int(step.get("move", -1)) for step in normalized_steps] != list(range(moves + 1)):
        raise ValueError("NPZ replay trace contains missing or duplicate moves")
    if not isinstance(summary, dict):
        summary = {}
    summary = {
        "moves": int(summary.get("moves", moves) or moves),
        "positions": int(summary.get("positions", max(0, moves)) or 0),
        "scores": summary.get("scores"),
        "returns": summary.get("returns"),
        "duration_seconds": summary.get("duration_seconds"),
    }
    return {
        "format": "gaiazero-local-history-v1",
        "run_id": run_id,
        "source": "training_npz",
        "status": "complete",
        "started_at": now,
        "updated_at": now,
        "completed_at": now,
        "ruleset": metadata.get("ruleset", "unknown"),
        "engine": "GaiaZero NPZ self-play",
        "config": {
            key: value
            for key, value in metadata.items()
            if key != "history"
        },
        "trace": {
            "run_id": run_id,
            "iteration": 1,
            "game": 1,
            "summary": summary,
        "trace_complete": summary["moves"] == moves,
            "captured_moves": moves,
            "steps": normalized_steps,
        },
        "npz_source": str(source.resolve()),
    }


def convert_npz_to_history(
    source: str | Path,
    history_dir: str | Path,
    *,
    run_id: str | None = None,
) -> Path:
    """Convert one self-play NPZ into a dashboard-loadable local replay."""
    source_path = Path(source)
    if source_path.suffix.lower() != ".npz":
        raise ValueError("source must be an .npz file")
    examples, metadata = read_npz_shard(source_path)
    typed_trajectory = None
    if metadata.get("schema_version") == NPZ_TRAJECTORY_SCHEMA_VERSION:
        # Validate the authoritative typed trace even when a legacy dashboard
        # history object is also present in metadata.
        typed_trajectory = read_npz_trajectory(source_path)
    # New trajectory shards keep the typed state arrays outside metadata so
    # training never has to parse replay JSON.  Materialize the dashboard's
    # familiar step shape only at this explicit conversion boundary.
    if not isinstance(metadata.get("history"), dict):
        trajectory = typed_trajectory or read_npz_trajectory(source_path)
        rows = []
        for index in range(len(trajectory["position_index"])):
            state = json.loads(str(trajectory["state_snapshot_json"][index]))
            action_id = int(trajectory["action_ids"][index])
            action_tuple = trajectory["action_tuples_json"][index]
            legal_tuples = trajectory["legal_action_tuples_json"][index]
            policy_targets = trajectory["policy_visit_targets_by_tuple_json"][index]
            rows.append(
                {
                    "move": index,
                    "position_index": int(trajectory["position_index"][index]),
                    "semantic_turn_index": int(trajectory["semantic_turn_index"][index]),
                    "player": int(trajectory["player_to_move"][index]),
                    "action": None if action_id < 0 else action_id,
                    "action_tuple": json.loads(str(action_tuple)) if action_tuple else None,
                    "legal_action_tuples": json.loads(str(legal_tuples))
                    if legal_tuples
                    else [],
                    "policy_visit_targets_by_tuple": json.loads(str(policy_targets))
                    if policy_targets
                    else [],
                    "state_hash": str(trajectory["state_hashes"][index]),
                    "state": state,
                }
            )
        metadata = dict(metadata)
        metadata["history"] = {
            "summary": {
                "moves": len(examples),
                "positions": len(examples),
                "scores": None,
                "returns": None,
            },
            "trace_complete": bool(trajectory["terminal_valid"]),
            "captured_moves": len(examples),
            "steps": rows,
        }
    resolved_id = run_id or f"npz-{source_path.stem}"
    record = _history_record(source_path, metadata, run_id=resolved_id)
    return write_local_game(history_dir, record)


def convert_npz_directory(
    source_dir: str | Path,
    history_dir: str | Path,
) -> list[Path]:
    """Convert every NPZ in a directory, in filename order."""
    return [
        convert_npz_to_history(path, history_dir)
        for path in sorted(Path(source_dir).glob("*.npz"))
    ]


def delete_training_history(history_dir: str | Path, run_id: str) -> bool:
    """Delete only a replay produced by NPZ conversion."""
    path = Path(history_dir) / f"{run_id}.json"
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if payload.get("source") != "training_npz":
        raise ValueError("the selected history is not an NPZ training replay")
    return delete_local_game(history_dir, run_id=run_id)
