from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, IO

from gaiazero.distributed import (
    PIPELINE_FORMAT,
    PipelineConfig,
    ensure_pipeline_dirs,
    pipeline_paths,
    save_pipeline_config,
)


WORKER_NAMES = ("selfplay", "shuffle", "train", "export", "gatekeeper")
WORKER_LABELS = {
    "selfplay": "自对弈",
    "shuffle": "样本洗牌",
    "train": "网络训练",
    "export": "模型导出",
    "gatekeeper": "守门测试",
}


@dataclass(slots=True)
class _WorkerProcess:
    process: subprocess.Popen[str]
    log: IO[str]


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}


def _read_jsonl(path: Path, limit: int = 40) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    records: list[dict[str, Any]] = []
    for line in lines[-limit:]:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def _tail_log(path: Path, lines: int = 80, max_bytes: int = 96 * 1024) -> list[str]:
    try:
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size - max_bytes))
            content = stream.read().decode("utf-8", errors="replace")
    except OSError:
        return []
    return content.splitlines()[-lines:]


def _path_info(path: Path) -> dict[str, Any] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return {
        "name": path.name,
        "path": str(path.resolve()),
        "size": stat.st_size,
        "modified_at": datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(),
    }


def _files_info(directory: Path, pattern: str, limit: int = 12) -> dict[str, Any]:
    try:
        paths = sorted(
            directory.glob(pattern),
            key=lambda item: item.stat().st_mtime_ns,
            reverse=True,
        )
    except OSError:
        paths = []
    details = [item for path in paths[:limit] if (item := _path_info(path)) is not None]
    total_bytes = 0
    for path in paths:
        try:
            total_bytes += path.stat().st_size
        except OSError:
            continue
    return {"count": len(paths), "bytes": total_bytes, "recent": details}


def _pipeline_config(payload: dict[str, Any], default_root: Path) -> PipelineConfig:
    values: dict[str, Any] = {}
    integer_fields = {
        "players",
        "seed",
        "simulations",
        "temperature_moves",
        "max_moves",
        "games_per_cycle",
        "shuffle_pack_size",
        "replay_capacity",
        "batch_size",
        "updates_per_cycle",
        "min_replay",
        "hidden_size",
        "residual_blocks",
        "gate_games",
    }
    float_fields = {
        "c_puct",
        "poll_seconds",
        "learning_rate",
        "weight_decay",
        "gate_threshold",
    }
    for name in PipelineConfig.__dataclass_fields__:
        if name not in payload:
            continue
        value = payload[name]
        if name == "root":
            values[name] = Path(str(value)).expanduser().resolve()
        elif name in integer_fields:
            values[name] = int(value)
        elif name in float_fields:
            values[name] = float(value)
        else:
            values[name] = str(value)
    values.setdefault("root", default_root)
    return PipelineConfig(**values)


class PipelineSupervisor:
    """Own and observe the five independent asynchronous training workers."""

    def __init__(self, default_root: str | Path) -> None:
        self.default_root = Path(default_root).resolve()
        self.root = self.default_root
        self._config: PipelineConfig | None = None
        self._workers: dict[str, _WorkerProcess] = {}
        self._lock = threading.RLock()
        self.started_at: str | None = None
        self.stop_requested_at: str | None = None

    def start(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            if any(handle.process.poll() is None for handle in self._workers.values()):
                raise RuntimeError("the asynchronous training pipeline is already running")
            self._close_finished_logs()
            config = _pipeline_config(payload, self.default_root)
            paths = ensure_pipeline_dirs(config.root)
            (paths["root"] / "STOP").unlink(missing_ok=True)
            config_path = save_pipeline_config(config)
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
            started: dict[str, _WorkerProcess] = {}
            try:
                for name in WORKER_NAMES:
                    log = (paths["logs"] / f"{name}.log").open(
                        "a",
                        encoding="utf-8",
                        newline="\n",
                    )
                    process = subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            "gaiazero.distributed",
                            name,
                            "--config",
                            str(config_path),
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        text=True,
                        creationflags=creationflags,
                    )
                    started[name] = _WorkerProcess(process=process, log=log)
            except Exception:
                for handle in started.values():
                    if handle.process.poll() is None:
                        handle.process.terminate()
                    handle.log.close()
                raise
            self.root = Path(config.root).resolve()
            self._config = config
            self._workers = started
            self.started_at = _utc_now()
            self.stop_requested_at = None
            return self.status()

    def stop(self) -> dict[str, Any]:
        with self._lock:
            paths = ensure_pipeline_dirs(self.root)
            (paths["root"] / "STOP").touch()
            self.stop_requested_at = _utc_now()
            for handle in self._workers.values():
                if handle.process.poll() is None:
                    handle.process.terminate()
            return self.status()

    def close(self) -> None:
        with self._lock:
            if any(handle.process.poll() is None for handle in self._workers.values()):
                self.stop()
            for handle in self._workers.values():
                if handle.process.poll() is None:
                    try:
                        handle.process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        handle.process.kill()
                        handle.process.wait(timeout=5)
                if not handle.log.closed:
                    handle.log.close()

    def status(self) -> dict[str, Any]:
        with self._lock:
            paths = pipeline_paths(self.root)
            workers = {
                name: self._worker_snapshot(name, paths)
                for name in WORKER_NAMES
            }
            process_states = [item["process_status"] for item in workers.values()]
            if any(item == "running" for item in process_states):
                overall = "stopping" if self.stop_requested_at else "running"
            elif any(item == "failed" for item in process_states):
                overall = "failed"
            elif self.stop_requested_at or (paths["root"] / "STOP").exists():
                overall = "stopped"
            elif any(item["snapshot"] for item in workers.values()):
                overall = "unmanaged"
            else:
                overall = "idle"

            config = self._config.json_dict() if self._config is not None else _read_json(paths["root"] / "pipeline.json")
            config.pop("format", None)
            summary = {
                "raw": _files_info(paths["raw"], "*.npz"),
                "shuffled": _files_info(paths["shuffled"], "*.npz"),
                "candidates": _files_info(paths["candidates"], "*.pt"),
                "exported": _files_info(paths["exported"], "model-*.bin"),
                "approved": _path_info(paths["approved"] / "current.pt"),
            }
            return {
                "format": PIPELINE_FORMAT,
                "status": overall,
                "root": str(paths["root"].resolve()),
                "started_at": self.started_at,
                "stop_requested_at": self.stop_requested_at,
                "config": config,
                "summary": summary,
                "workers": workers,
            }

    def _worker_snapshot(self, name: str, paths: dict[str, Path]) -> dict[str, Any]:
        handle = self._workers.get(name)
        exit_code: int | None = None
        pid: int | None = None
        if handle is None:
            process_status = "unmanaged"
        else:
            pid = handle.process.pid
            exit_code = handle.process.poll()
            process_status = (
                "running"
                if exit_code is None
                else "stopped"
                if exit_code == 0 or self.stop_requested_at
                else "failed"
            )
            if exit_code is not None and not handle.log.closed:
                handle.log.close()
        snapshot = _read_json(paths["status"] / f"{name}.json")
        if handle is None and not snapshot:
            process_status = "idle"
        result: dict[str, Any] = {
            "name": name,
            "label": WORKER_LABELS[name],
            "process_status": process_status,
            "pid": pid or snapshot.get("pid"),
            "exit_code": exit_code,
            "snapshot": snapshot,
            "log": _tail_log(paths["logs"] / f"{name}.log"),
        }
        if name == "selfplay":
            result["artifacts"] = _files_info(paths["raw"], "*.npz")
        elif name == "shuffle":
            result["state"] = _read_json(paths["root"] / "shuffle-state.json")
            result["artifacts"] = _files_info(paths["shuffled"], "*.npz")
        elif name == "train":
            result["state"] = _read_json(paths["root"] / "train-state.json")
            result["history"] = _read_jsonl(paths["logs"] / "train.jsonl", 120)
            result["artifacts"] = _files_info(paths["candidates"], "*.pt")
            result["latest"] = _path_info(paths["training"] / "latest.pt")
        elif name == "export":
            result["state"] = _read_json(paths["root"] / "export-state.json")
            result["artifacts"] = _files_info(paths["exported"], "model-*.bin")
            result["current"] = _path_info(paths["exported"] / "current.bin")
        elif name == "gatekeeper":
            result["state"] = _read_json(paths["root"] / "gatekeeper-state.json")
            result["history"] = _read_jsonl(paths["logs"] / "gatekeeper.jsonl", 80)
            result["approved"] = _read_json(paths["approved"] / "current.json")
        return result

    def _close_finished_logs(self) -> None:
        for handle in self._workers.values():
            if handle.process.poll() is not None and not handle.log.closed:
                handle.log.close()

