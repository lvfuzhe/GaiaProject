from __future__ import annotations

import json
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


class JsonlTelemetry:
    """Append-only event sink readable while another process is training."""

    def __init__(self, path: str | Path, run_id: str | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self._sequence = _last_sequence(self.path)
        self._lock = threading.Lock()

    def emit(self, event_type: str, **payload: Any) -> dict[str, Any]:
        with self._lock:
            self._sequence += 1
            event = {
                "sequence": self._sequence,
                "timestamp": datetime.now(UTC).isoformat(),
                "run_id": self.run_id,
                "type": event_type,
                "payload": payload,
            }
            encoded = json.dumps(event, ensure_ascii=False, separators=(",", ":"), default=_json_default)
            with self.path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(encoded + "\n")
                stream.flush()
            return event


def _last_sequence(path: Path) -> int:
    if not path.exists():
        return 0
    last = 0
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            try:
                last = max(last, int(json.loads(line).get("sequence", 0)))
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
    return last


def read_events(path: str | Path, *, after: int = 0, limit: int = 5_000) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return []
    events: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as stream:
        for line in stream:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if int(event.get("sequence", 0)) > after:
                events.append(event)
    return events[-max(1, limit) :]


def build_history_index(path: str | Path) -> dict[str, Any]:
    """Build a compact run/iteration/game index without retaining full snapshots."""

    runs: dict[str, dict[str, Any]] = {}
    latest_sequence = 0
    for event in _iter_events(path):
        sequence = int(event.get("sequence", 0))
        latest_sequence = max(latest_sequence, sequence)
        run_id = str(event.get("run_id") or "legacy")
        event_type = str(event.get("type") or "")
        payload = event.get("payload") or {}
        run = runs.setdefault(
            run_id,
            {
                "run_id": run_id,
                "started_at": event.get("timestamp"),
                "completed_at": None,
                "status": "running",
                "ruleset": None,
                "config": {},
                "iterations": {},
            },
        )
        snapshot = payload.get("state") or {}
        if snapshot.get("ruleset"):
            run["ruleset"] = snapshot["ruleset"]
        if event_type == "run_started":
            run["started_at"] = event.get("timestamp")
            run["config"] = payload.get("config") or {}
        elif event_type == "run_completed":
            run["status"] = "complete"
            run["completed_at"] = event.get("timestamp")
        elif event_type == "run_failed":
            run["status"] = "failed"
            run["completed_at"] = event.get("timestamp")

        iteration_number = payload.get("iteration")
        if iteration_number is None:
            continue
        iteration_number = int(iteration_number)
        iteration = run["iterations"].setdefault(
            iteration_number,
            {"iteration": iteration_number, "metrics": None, "games": {}},
        )
        if event_type == "iteration_completed":
            iteration["metrics"] = {
                key: payload.get(key)
                for key in (
                    "new_positions",
                    "replay_positions",
                    "loss",
                    "policy_loss",
                    "value_loss",
                    "policy_entropy",
                    "duration_seconds",
                    "checkpoint",
                )
            }

        game_number = payload.get("game_in_iteration")
        if game_number is None:
            continue
        game_number = int(game_number)
        game = iteration["games"].setdefault(
            game_number,
            {
                "game": game_number,
                "started_at": None,
                "completed_at": None,
                "moves": None,
                "captured_moves": 0,
                "positions": None,
                "scores": None,
                "returns": None,
                "duration_seconds": None,
                "complete": False,
                "trace_complete": False,
                "_moves": set(),
            },
        )
        if event_type == "self_play_started":
            game["started_at"] = event.get("timestamp")
        elif event_type == "self_play_step":
            move = payload.get("move")
            if move is not None:
                game["_moves"].add(int(move))
        elif event_type == "self_play_completed":
            game.update(
                completed_at=event.get("timestamp"),
                moves=int(payload.get("moves", 0)),
                positions=payload.get("positions"),
                scores=payload.get("scores"),
                returns=payload.get("returns"),
                duration_seconds=payload.get("duration_seconds"),
                complete=True,
            )

    normalized_runs: list[dict[str, Any]] = []
    for run in runs.values():
        normalized_iterations: list[dict[str, Any]] = []
        for iteration in run["iterations"].values():
            normalized_games: list[dict[str, Any]] = []
            for game in iteration["games"].values():
                observed = game.pop("_moves")
                game["captured_moves"] = len(observed)
                expected = game["moves"]
                game["trace_complete"] = (
                    expected is not None and observed == set(range(1, expected + 1))
                )
                normalized_games.append(game)
            iteration["games"] = normalized_games
            normalized_iterations.append(iteration)
        run["iterations"] = normalized_iterations
        normalized_runs.append(run)
    return {"runs": normalized_runs, "latest_sequence": latest_sequence}


def read_game_trace(
    path: str | Path,
    *,
    run_id: str,
    iteration: int,
    game: int,
) -> dict[str, Any] | None:
    """Read one self-play game, including initial, sampled/full steps, and result."""

    started: dict[str, Any] | None = None
    completed: dict[str, Any] | None = None
    steps: list[dict[str, Any]] = []
    for event in _iter_events(path):
        if str(event.get("run_id") or "legacy") != run_id:
            continue
        payload = event.get("payload") or {}
        if int(payload.get("iteration", -1)) != iteration:
            continue
        if int(payload.get("game_in_iteration", -1)) != game:
            continue
        event_type = event.get("type")
        if event_type == "self_play_started":
            started = event
            snapshot = payload.get("state")
            if snapshot:
                steps.append(
                    {
                        "sequence": event.get("sequence"),
                        "timestamp": event.get("timestamp"),
                        "move": 0,
                        "player": snapshot.get("current_player"),
                        "action": None,
                        "action_label": "initial state",
                        "legal_actions": None,
                        "search_sampled": False,
                        "root_value": None,
                        "candidates": [],
                        "state": snapshot,
                    }
                )
        elif event_type == "self_play_step":
            steps.append(
                {
                    "sequence": event.get("sequence"),
                    "timestamp": event.get("timestamp"),
                    "move": int(payload.get("move", 0)),
                    "player": payload.get("player"),
                    "action": payload.get("action"),
                    "action_label": payload.get("action_label"),
                    "legal_actions": payload.get("legal_actions"),
                    "search_sampled": bool(payload.get("search_sampled", payload.get("candidates"))),
                    "root_value": payload.get("root_value"),
                    "candidates": payload.get("candidates") or [],
                    "state": payload.get("state"),
                }
            )
        elif event_type == "self_play_completed":
            completed = event

    if started is None and completed is None and not steps:
        return None
    expected_moves = int((completed or {}).get("payload", {}).get("moves", 0)) if completed else None
    observed_moves = {step["move"] for step in steps if step["move"] > 0}
    if completed is not None and expected_moves not in observed_moves:
        payload = completed.get("payload") or {}
        snapshot = payload.get("state")
        if snapshot:
            steps.append(
                {
                    "sequence": completed.get("sequence"),
                    "timestamp": completed.get("timestamp"),
                    "move": expected_moves,
                    "player": None,
                    "action": None,
                    "action_label": "completed state",
                    "legal_actions": None,
                    "search_sampled": False,
                    "root_value": None,
                    "candidates": [],
                    "state": snapshot,
                }
            )
    steps.sort(key=lambda step: (step["move"], int(step.get("sequence") or 0)))
    completed_payload = (completed or {}).get("payload") or {}
    return {
        "run_id": run_id,
        "iteration": iteration,
        "game": game,
        "started_at": started.get("timestamp") if started else None,
        "completed_at": completed.get("timestamp") if completed else None,
        "summary": {
            key: completed_payload.get(key)
            for key in ("moves", "positions", "scores", "returns", "duration_seconds")
        },
        "trace_complete": (
            expected_moves is not None and observed_moves == set(range(1, expected_moves + 1))
        ),
        "captured_moves": len(observed_moves),
        "steps": steps,
    }


def _iter_events(path: str | Path):
    source = Path(path)
    if not source.exists():
        return
    with source.open("r", encoding="utf-8") as stream:
        for line in stream:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue
