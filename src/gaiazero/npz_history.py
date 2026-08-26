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

from gaiazero.distributed import read_npz_shard
from gaiazero.telemetry import delete_local_game, write_local_game


def _minimal_state(metadata: dict[str, Any], move: int, player: Any) -> dict[str, Any]:
    players = max(1, int(metadata.get("players", 3) or 3))
    return {
        "ruleset": metadata.get("ruleset", "unknown"),
        "round": 0,
        "max_rounds": 6,
        "phase": "training_sample",
        "current_player": player,
        "terminal": False,
        "scores": [0.0] * players,
        "players": [
            {
                "id": index,
                "name": f"P{index}",
                "faction": "Unknown",
                "credits": 0,
                "ore": 0,
                "knowledge": 0,
                "qic": 0,
                "vp": 0,
                "tracks": [0, 0, 0, 0, 0, 0],
            }
            for index in range(players)
        ],
        "planets": [],
        "training_sample_move": move,
    }


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
        steps = []
        try:
            examples, _ = read_npz_shard(source)
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            examples = []
        for move, _example in enumerate(examples, start=1):
            steps.append(
                {
                    "move": move,
                    "player": None,
                    "action": None,
                    "action_label": "training sample (state snapshot unavailable)",
                    "state": _minimal_state(metadata, move, None),
                }
            )
        steps.insert(
            0,
            {
                "move": 0,
                "player": None,
                "action": None,
                "action_label": "training sample archive",
                "state": _minimal_state(metadata, 0, None),
            },
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
        step.setdefault("state", _minimal_state(metadata, int(step["move"]), step.get("player")))
        normalized_steps.append(step)
    normalized_steps.sort(key=lambda item: int(item.get("move", 0)))
    moves = max(0, len(normalized_steps) - 1)
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
            "trace_complete": bool(normalized_steps and summary["moves"] == moves),
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
    _examples, metadata = read_npz_shard(source_path)
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
