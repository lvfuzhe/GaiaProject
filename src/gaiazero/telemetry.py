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

