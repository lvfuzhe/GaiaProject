from __future__ import annotations

import json
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from time import sleep
from typing import Any

import numpy as np


LOCAL_HISTORY_FORMAT = "gaiazero-local-history-v1"
_LOCAL_HISTORY_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def write_local_game(path: str | Path, record: dict[str, Any]) -> Path:
    """Atomically persist one interactive game in the local history archive."""

    directory = Path(path)
    run_id = str(record.get("run_id") or "")
    if not _LOCAL_HISTORY_ID.fullmatch(run_id):
        raise ValueError("local history run_id contains unsupported characters")
    payload = dict(record)
    payload["format"] = LOCAL_HISTORY_FORMAT
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{run_id}.json"
    temporary = directory / f".{run_id}.{uuid.uuid4().hex}.tmp"
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        default=_json_default,
    )
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded + "\n")
            stream.flush()
        for attempt in range(6):
            try:
                temporary.replace(target)
                break
            except PermissionError:
                if attempt == 5:
                    raise
                sleep(0.01 * (attempt + 1))
    finally:
        temporary.unlink(missing_ok=True)
    return target


def build_local_history_index(path: str | Path) -> dict[str, Any]:
    """Build a dashboard-compatible index for locally saved interactive games."""

    directory = Path(path)
    runs: list[dict[str, Any]] = []
    if not directory.is_dir():
        return {"runs": runs, "source": str(directory)}
    for source in sorted(directory.glob("*.json")):
        record = _read_local_game(source)
        if record is None:
            continue
        trace = record["trace"]
        summary = trace.get("summary") or {}
        steps = trace.get("steps") or []
        try:
            expected_moves = int(summary.get("moves", max(0, len(steps) - 1)))
            observed_moves = {
                int(step.get("move", 0))
                for step in steps
                if isinstance(step, dict) and int(step.get("move", 0)) > 0
            }
            trace_complete = (
                bool(steps)
                and isinstance(steps[0], dict)
                and int(steps[0].get("move", -1)) == 0
                and observed_moves == set(range(1, expected_moves + 1))
            )
        except (TypeError, ValueError):
            continue
        status = str(record.get("status") or "active")
        record_source = (
            "bga"
            if record.get("source") == "bga"
            else "training_npz"
            if record.get("source") == "training_npz"
            else "local"
        )
        completed_at = record.get("completed_at")
        run_id = str(record["run_id"])
        game = {
            "game": 1,
            "started_at": record.get("started_at"),
            "completed_at": completed_at,
            "moves": expected_moves,
            "captured_moves": len(observed_moves),
            "positions": summary.get("positions", expected_moves),
            "scores": summary.get("scores"),
            "returns": summary.get("returns"),
            "duration_seconds": summary.get("duration_seconds"),
            "complete": status == "complete",
            "trace_complete": trace_complete,
        }
        runs.append(
            {
                "run_id": run_id,
                "source": record_source,
                "started_at": record.get("started_at"),
                "updated_at": record.get("updated_at"),
                "completed_at": completed_at,
                "status": status,
                "ruleset": record.get("ruleset"),
                "config": record.get("config") or {},
                "roles": record.get("roles") or [],
                "engine": record.get("engine"),
                "iterations": [
                    {
                        "iteration": 1,
                        "source": record_source,
                        "metrics": None,
                        "games": [game],
                    }
                ],
            }
        )
    runs.sort(key=lambda run: str(run.get("started_at") or ""))
    return {"runs": runs, "source": str(directory)}


def read_local_game_trace(
    path: str | Path,
    *,
    run_id: str,
    iteration: int = 1,
    game: int = 1,
) -> dict[str, Any] | None:
    """Load one locally persisted interactive game as a replay trace."""

    if iteration != 1 or game != 1 or not _LOCAL_HISTORY_ID.fullmatch(run_id):
        return None
    record = _read_local_game(Path(path) / f"{run_id}.json")
    if record is None:
        return None
    record_source = (
        "bga"
        if record.get("source") == "bga"
        else "training_npz"
        if record.get("source") == "training_npz"
        else "local"
    )
    trace = (
        _rebuild_legacy_bga_trace(record)
        if record_source == "bga"
        else dict(record["trace"])
    )
    trace.update(
        source=record_source,
        status=record.get("status"),
        updated_at=record.get("updated_at"),
        config=record.get("config") or {},
        roles=record.get("roles") or [],
        engine=record.get("engine"),
    )
    if record_source == "bga":
        bga = record.get("bga") or {}
        trace["bga_audit"] = {
            "notification_catalog": bga.get("notification_catalog") or [],
            "notification_coverage": bga.get("notification_coverage") or {},
        }
    return trace


def _rebuild_legacy_bga_trace(record: dict[str, Any]) -> dict[str, Any]:
    trace = dict(record["trace"])
    steps = trace.get("steps") or []
    if not any(
        isinstance(player, dict)
        and (
            "federation_unused" not in player
            or "federation_used" not in player
        )
        for step in steps
        if isinstance(step, dict)
        for player in (step.get("state") or {}).get("players", [])
    ):
        return trace

    bga = record.get("bga") or {}
    packets = bga.get("log_packets")
    if not isinstance(packets, list) or not packets:
        return trace
    players = bga.get("players") or []
    try:
        table_id = int(bga.get("table_id"))
        game_data = {
            "gamename": bga.get("game") or "gaiaproject",
            "tableId": str(table_id),
            "players": [
                {
                    "id": int(player["bga_player_id"]),
                    "no": int(player.get("seat", index)) + 1,
                }
                for index, player in enumerate(players)
                if isinstance(player, dict) and player.get("bga_player_id") is not None
            ],
            "tableOptions": bga.get("table_options") or [],
        }
        review_players = {
            int(player["bga_player_id"]): {
                "name": player.get("name"),
                "score": player.get("score"),
            }
            for player in players
            if isinstance(player, dict) and player.get("bga_player_id") is not None
        }
        # Import lazily because the BGA importer persists records through this module.
        from gaiazero.bga import convert_bga_replay

        rebuilt = convert_bga_replay(
            table_id=table_id,
            source_url=str(bga.get("source_url") or ""),
            replay_url=str(bga.get("replay_url") or ""),
            game_data=game_data,
            packets=packets,
            review_players=review_players,
            initial_state=bga.get("initial_setup"),
        )
    except (KeyError, TypeError, ValueError):
        return trace
    rebuilt_trace = rebuilt.get("trace")
    return dict(rebuilt_trace) if isinstance(rebuilt_trace, dict) else trace


def delete_local_game(path: str | Path, *, run_id: str) -> bool:
    """Delete one validated local replay without accepting arbitrary paths."""

    if not _LOCAL_HISTORY_ID.fullmatch(run_id):
        raise ValueError("local history run_id contains unsupported characters")
    target = Path(path) / f"{run_id}.json"
    if _read_local_game(target) is None:
        return False
    target.unlink()
    return True


def _read_local_game(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as stream:
            record = json.load(stream)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(record, dict) or record.get("format") != LOCAL_HISTORY_FORMAT:
        return None
    run_id = str(record.get("run_id") or "")
    trace = record.get("trace")
    if (
        not _LOCAL_HISTORY_ID.fullmatch(run_id)
        or path.stem != run_id
        or not isinstance(trace, dict)
        or not isinstance(trace.get("steps"), list)
    ):
        return None
    return record
