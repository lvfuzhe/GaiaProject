"""GaiaZero-native asynchronous multiplayer training pipeline.

The process layout follows the proven asynchronous pattern used by KataGo,
but every game rule, sample, network and exported weight belongs to GaiaZero.
There is no TensorFlow, TFRecord, KataGo engine protocol or Go compatibility.

Workers communicate through atomically published files:

    selfplay -> raw/*.npz -> shuffle -> shuffled/*.npz
    -> train -> candidates/*.pt -> gatekeeper -> approved/current.pt
    -> export -> exported/current.bin
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import shutil
import struct
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from gaiazero.arena import evaluate_against
from gaiazero.game import (
    GaiaHeuristicEvaluator,
    GaiaState,
)
from gaiazero.mcts import SearchConfig
from gaiazero.model import (
    NetworkConfig,
    NetworkEvaluator,
    PolicyValueNetwork,
    architecture_for_players,
    load_checkpoint,
    resolve_device,
    save_checkpoint,
)
from gaiazero.replay import ReplayBuffer, TrainingExample
from gaiazero.selfplay import SelfPlayConfig, play_self_game
from gaiazero.training import AlphaZeroTrainer, TrainerConfig


PIPELINE_FORMAT = 1
EXPORT_MAGIC = b"GAIAZERO-MULTIPLAYER-BIN-V1\0"


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    root: Path = Path("runs/multiplayer-pipeline")
    players: int = 4
    seed: int = 0
    simulations: int = 64
    c_puct: float = 1.5
    temperature_moves: int = 24
    max_moves: int = 512
    poll_seconds: float = 2.0
    games_per_cycle: int = 1
    shuffle_pack_size: int = 4096
    replay_capacity: int = 200_000
    batch_size: int = 256
    updates_per_cycle: int = 32
    min_replay: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    hidden_size: int = 256
    residual_blocks: int = 4
    device: str = "auto"
    gate_games: int = 20
    gate_threshold: float = 0.55

    def __post_init__(self) -> None:
        if self.players not in (2, 3, 4):
            raise ValueError("the multiplayer pipeline supports 2, 3 or 4 players")
        positive = (
            "simulations",
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
        )
        for name in positive:
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.poll_seconds <= 0:
            raise ValueError("poll_seconds must be positive")
        if not 0.0 < self.gate_threshold <= 1.0:
            raise ValueError("gate_threshold must be in (0, 1]")

    def json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["root"] = str(self.root)
        return payload


def pipeline_paths(root: str | Path) -> dict[str, Path]:
    base = Path(root)
    return {
        "root": base,
        "raw": base / "raw",
        "shuffled": base / "shuffled",
        "training": base / "training",
        "candidates": base / "candidates",
        "approved": base / "approved",
        "exported": base / "exported",
        "logs": base / "logs",
        "status": base / "status",
    }


def ensure_pipeline_dirs(root: str | Path) -> dict[str, Path]:
    paths = pipeline_paths(root)
    for name, path in paths.items():
        if name != "root":
            path.mkdir(parents=True, exist_ok=True)
    return paths


def _atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    content = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    _atomic_bytes(path, content)


def _read_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else dict(default)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return dict(default)


def _publish_worker_status(
    paths: dict[str, Path],
    worker: str,
    phase: str,
    **metrics: Any,
) -> None:
    previous = _read_json(paths["status"] / f"{worker}.json", {})
    previous_metrics = previous.get("metrics")
    if not metrics and isinstance(previous_metrics, dict):
        metrics = previous_metrics
    _atomic_json(
        paths["status"] / f"{worker}.json",
        {
            "format": PIPELINE_FORMAT,
            "worker": worker,
            "phase": phase,
            "pid": os.getpid(),
            "updated_at": datetime.now(UTC).isoformat(),
            "metrics": metrics,
        },
    )


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
        stream.flush()


def save_pipeline_config(config: PipelineConfig) -> Path:
    path = Path(config.root) / "pipeline.json"
    _atomic_json(path, {"format": PIPELINE_FORMAT, **config.json_dict()})
    return path


def load_pipeline_config(path: str | Path) -> PipelineConfig:
    payload = _read_json(Path(path), {})
    payload.pop("format", None)
    # Older pipeline files may contain the removed ruleset selector. The
    # current pipeline always runs the standard GaiaState environment.
    payload.pop("ruleset", None)
    if not payload:
        raise ValueError(f"pipeline configuration is missing or invalid: {path}")
    payload["root"] = Path(payload["root"])
    return PipelineConfig(**payload)


def _stop_requested(root: Path) -> bool:
    return (root / "STOP").exists()


def _game_components():
    return GaiaState, GaiaHeuristicEvaluator()


def _network_config(config: PipelineConfig) -> tuple[type, NetworkConfig]:
    state_type, _baseline = _game_components()
    template = state_type.initial(config.players, config.seed)
    network = NetworkConfig(
        observation_size=template.observation_size,
        action_size=template.action_size,
        num_players=config.players,
        hidden_size=config.hidden_size,
        residual_blocks=config.residual_blocks,
        architecture=architecture_for_players(config.players),
    )
    return state_type, network


def _search_config(config: PipelineConfig, seed: int) -> SearchConfig:
    return SearchConfig(
        simulations=config.simulations,
        c_puct=config.c_puct,
        seed=seed,
    )


def _approved_checkpoint(paths: dict[str, Path]) -> Path | None:
    path = paths["approved"] / "current.pt"
    return path if path.is_file() else None


def _save_checkpoint_atomic(
    path: Path,
    model: PolicyValueNetwork,
    **kwargs: Any,
) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    save_checkpoint(temporary, model, **kwargs)
    os.replace(temporary, path)


def write_npz_shard(
    path: Path,
    examples: Sequence[TrainingExample],
    metadata: dict[str, Any] | None = None,
) -> int:
    """Atomically write a native NumPy shard consumed directly by PyTorch.

    Self-play raw shards may additionally carry a complete replay trace in
    ``metadata['history']``. Training ignores that field; the explicit NPZ
    converter uses it to rebuild the dashboard replay.
    """
    if not examples:
        raise ValueError("cannot write an empty training shard")
    payload = io.BytesIO()
    np.savez_compressed(
        payload,
        observations=np.stack([item.observation for item in examples]).astype(np.float32),
        legal_masks=np.stack([item.legal_mask for item in examples]).astype(np.bool_),
        policy_targets=np.stack([item.policy_target for item in examples]).astype(np.float32),
        value_targets=np.stack([item.value_target for item in examples]).astype(np.float32),
        metadata=np.asarray(
            json.dumps(metadata or {}, ensure_ascii=False),
            dtype=np.str_,
        ),
    )
    _atomic_bytes(path, payload.getvalue())
    return len(examples)


def read_npz_shard(
    path: Path,
    *,
    include_metadata: bool = True,
) -> tuple[list[TrainingExample], dict[str, Any]]:
    """Read and validate one raw or shuffled NumPy shard.

    Training workers pass ``include_metadata=False`` so complete replay traces
    remain opaque and are not parsed during optimization. The explicit history
    converter keeps the default and reads the trace metadata.
    """
    with np.load(path, allow_pickle=False) as values:
        observations = np.asarray(values["observations"], dtype=np.float32)
        legal_masks = np.asarray(values["legal_masks"], dtype=np.bool_)
        policy_targets = np.asarray(values["policy_targets"], dtype=np.float32)
        value_targets = np.asarray(values["value_targets"], dtype=np.float32)
        raw_metadata = (
            str(values["metadata"].item())
            if include_metadata and "metadata" in values
            else "{}"
        )
    lengths = {len(observations), len(legal_masks), len(policy_targets), len(value_targets)}
    if len(lengths) != 1:
        raise ValueError(f"sample lengths do not match in {path}")
    if observations.ndim != 2 or legal_masks.ndim != 2 or policy_targets.ndim != 2:
        raise ValueError(f"invalid training array dimensions in {path}")
    try:
        metadata = json.loads(raw_metadata)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid shard metadata in {path}") from error
    examples = [
        TrainingExample(
            observation=observations[index],
            legal_mask=legal_masks[index],
            policy_target=policy_targets[index],
            value_target=value_targets[index],
        )
        for index in range(len(observations))
    ]
    return examples, metadata


def run_selfplay(config: PipelineConfig, *, once: bool = False) -> int:
    """Continuously run GaiaZero PIMCTS games and publish raw `.npz` files."""
    paths = ensure_pipeline_dirs(config.root)
    state_type, expected_network = _network_config(config)
    evaluator: NetworkEvaluator | None = None
    loaded_source: tuple[str, int] | None = None
    game_index = 0
    _publish_worker_status(paths, "selfplay", "starting", games=0)
    while not _stop_requested(paths["root"]):
        approved = _approved_checkpoint(paths)
        source = (
            (str(approved), approved.stat().st_mtime_ns)
            if approved is not None
            else ("random-bootstrap", 0)
        )
        if evaluator is None or source != loaded_source:
            if approved is None:
                model = PolicyValueNetwork(expected_network)
            else:
                model, _metadata = load_checkpoint(approved, config.device)
                if model.config != expected_network:
                    raise ValueError("approved checkpoint does not match this pipeline")
            evaluator = NetworkEvaluator(model, config.device)
            loaded_source = source
            print(f"[selfplay] loaded {source[0]}", flush=True)

        games = 1 if once else config.games_per_cycle
        for _ in range(games):
            seed = config.seed + game_index
            _publish_worker_status(
                paths,
                "selfplay",
                "playing",
                games=game_index,
                seed=seed,
                weight=source[0],
            )
            initial = state_type.initial(config.players, seed)
            trace_started = time.perf_counter()
            trace_steps: list[dict[str, Any]] = [{
                "move": 0,
                "player": initial.current_player,
                "action": None,
                "action_label": "initial state",
                "legal_actions": len(initial.legal_actions()),
                "search_sampled": False,
                "root_value": None,
                "candidates": [],
                "state": initial.snapshot(),
            }]

            def observe_move(
                move: int,
                before: Any,
                action: int,
                after: Any,
                search_result: Any,
            ) -> None:
                top_actions = np.argsort(search_result.policy)[-3:][::-1]
                candidates = [
                    {
                        "action": int(candidate),
                        "label": before.describe_action(int(candidate)),
                        "probability": float(search_result.policy[candidate]),
                        "visits": int(search_result.visits[candidate]),
                    }
                    for candidate in top_actions
                    if search_result.policy[candidate] > 0
                ]
                root_value = np.asarray(search_result.root_value)
                root_value_payload: Any = (
                    root_value.item()
                    if root_value.ndim == 0
                    else root_value.tolist()
                )
                trace_steps.append({
                    "move": move,
                    "player": before.current_player,
                    "action": action,
                    "action_label": before.describe_action(action),
                    "legal_actions": len(before.legal_actions()),
                    "search_sampled": True,
                    "root_value": root_value_payload,
                    "candidates": candidates,
                    "state": after.snapshot(),
                })

            result = play_self_game(
                initial,
                evaluator,
                _search_config(config, seed),
                SelfPlayConfig(
                    temperature_moves=config.temperature_moves,
                    max_moves=config.max_moves,
                    seed=seed,
                ),
                observer=observe_move,
            )
            destination = paths["raw"] / (
                f"game-{time.time_ns()}-{os.getpid()}-{game_index:08d}.npz"
            )
            write_npz_shard(
                destination,
                result.examples,
                {
                    "kind": "selfplay-game",
                    "seed": seed,
                    "players": config.players,
                    "ruleset": "standard-v22",
                    "architecture": expected_network.architecture,
                    "moves": len(result.actions),
                    "weight": source[0],
                    "history_format": "gaiazero-replay-trace-v1",
                    "history": {
                        "summary": {
                            "moves": len(result.actions),
                            "positions": len(result.examples),
                            "scores": list(result.final_state.final_scores()),
                            "returns": result.final_state.returns().tolist(),
                            "duration_seconds": time.perf_counter() - trace_started,
                        },
                        "trace_complete": len(trace_steps) == len(result.actions) + 1,
                        "captured_moves": len(result.actions),
                        "steps": trace_steps,
                    },
                },
            )
            _publish_worker_status(
                paths,
                "selfplay",
                "waiting",
                games=game_index + 1,
                seed=seed,
                weight=source[0],
                latest_shard=destination.name,
                positions=len(result.examples),
                moves=len(result.actions),
                duration_seconds=time.perf_counter() - trace_started,
            )
            print(
                f"[selfplay] wrote {destination.name} positions={len(result.examples)}",
                flush=True,
            )
            game_index += 1
        if once:
            break
        time.sleep(config.poll_seconds)
    return game_index


def run_shuffle(config: PipelineConfig, *, once: bool = False) -> int:
    """Merge unprocessed games, shuffle examples, and write `.npz` packs."""
    paths = ensure_pipeline_dirs(config.root)
    state_path = paths["root"] / "shuffle-state.json"
    state = _read_json(
        state_path,
        {"format": PIPELINE_FORMAT, "processed": [], "sequence": 0, "examples": 0},
    )
    processed = set(str(name) for name in state.get("processed", []))
    total_written = 0
    _publish_worker_status(
        paths,
        "shuffle",
        "starting",
        processed_games=len(processed),
        examples=int(state.get("examples", 0)),
    )
    while not _stop_requested(paths["root"]):
        pending = [
            path
            for path in sorted(paths["raw"].glob("*.npz"))
            if path.name not in processed
        ]
        if pending:
            _publish_worker_status(
                paths,
                "shuffle",
                "packing",
                pending_games=len(pending),
                processed_games=len(processed),
            )
            examples: list[TrainingExample] = []
            for path in pending:
                shard, _metadata = read_npz_shard(path, include_metadata=False)
                examples.extend(shard)
            random.Random(config.seed + int(state.get("sequence", 0))).shuffle(examples)
            sequence = int(state.get("sequence", 0))
            for start in range(0, len(examples), config.shuffle_pack_size):
                pack = examples[start : start + config.shuffle_pack_size]
                destination = paths["shuffled"] / f"shuffle-{sequence:08d}.npz"
                write_npz_shard(
                    destination,
                    pack,
                    {
                        "kind": "shuffled-training-pack",
                        "sequence": sequence,
                        "source_games": len(pending),
                    },
                )
                sequence += 1
                total_written += len(pack)
            processed.update(path.name for path in pending)
            state = {
                "format": PIPELINE_FORMAT,
                "processed": sorted(processed),
                "sequence": sequence,
                "examples": int(state.get("examples", 0)) + len(examples),
            }
            _atomic_json(state_path, state)
            print(
                f"[shuffle] games={len(pending)} positions={len(examples)} packs={sequence}",
                flush=True,
            )
        _publish_worker_status(
            paths,
            "shuffle",
            "waiting",
            processed_games=len(processed),
            examples=int(state.get("examples", 0)),
            packs=int(state.get("sequence", 0)),
            last_cycle_positions=total_written,
        )
        if once:
            break
        time.sleep(config.poll_seconds)
    return total_written


def _fill_replay(
    paths: dict[str, Path],
    replay: ReplayBuffer,
    loaded: set[str],
) -> int:
    added = 0
    for path in sorted(paths["shuffled"].glob("*.npz")):
        if path.name in loaded:
            continue
        examples, _metadata = read_npz_shard(path, include_metadata=False)
        replay.extend(examples)
        loaded.add(path.name)
        added += len(examples)
    return added


def run_train(config: PipelineConfig, *, once: bool = False) -> int:
    """Read shuffled `.npz` packs directly and train with native PyTorch."""
    paths = ensure_pipeline_dirs(config.root)
    _state_type, expected_network = _network_config(config)
    latest = paths["training"] / "latest.pt"
    approved = _approved_checkpoint(paths)
    starting = latest if latest.is_file() else approved
    if starting is None:
        model = PolicyValueNetwork(expected_network)
    else:
        model, _metadata = load_checkpoint(starting, config.device)
        if model.config != expected_network:
            raise ValueError("training checkpoint does not match this pipeline")
    trainer = AlphaZeroTrainer(
        model,
        TrainerConfig(
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            device=str(resolve_device(config.device)),
        ),
    )
    replay = ReplayBuffer(config.replay_capacity, config.seed)
    loaded: set[str] = set()
    state_path = paths["root"] / "train-state.json"
    state = _read_json(
        state_path,
        {"format": PIPELINE_FORMAT, "trained_shards": [], "generation": 0, "updates": 0},
    )
    trained_shards = set(str(name) for name in state.get("trained_shards", []))
    generation = int(state.get("generation", 0))
    updates = int(state.get("updates", 0))

    # Reconstruct the in-memory replay after a process restart. ReplayBuffer
    # enforces its own rolling capacity while shards remain immutable on disk.
    _fill_replay(paths, replay, loaded)
    _publish_worker_status(
        paths,
        "train",
        "starting",
        generation=generation,
        updates=updates,
        replay_positions=len(replay),
    )
    while not _stop_requested(paths["root"]):
        new_positions = _fill_replay(paths, replay, loaded)
        untrained = loaded - trained_shards
        should_train = len(replay) >= config.min_replay and (bool(untrained) or once)
        if should_train:
            _publish_worker_status(
                paths,
                "train",
                "optimizing",
                generation=generation,
                updates=updates,
                replay_positions=len(replay),
                new_positions=new_positions,
            )
            metrics = trainer.train_updates(replay, config.updates_per_cycle)
            generation += 1
            updates += config.updates_per_cycle
            metadata = {
                "pipeline_generation": generation,
                "players": config.players,
                "ruleset": "standard-v22",
                "architecture": model.architecture,
                "replay_positions": len(replay),
                "updates": updates,
                "loss": metrics.loss,
            }
            candidate = paths["candidates"] / f"candidate-{generation:08d}.pt"
            _save_checkpoint_atomic(
                candidate,
                model,
                optimizer=trainer.optimizer,
                metadata=metadata,
            )
            _save_checkpoint_atomic(
                latest,
                model,
                optimizer=trainer.optimizer,
                metadata=metadata,
            )
            trained_shards.update(loaded)
            training_record = {
                "timestamp": datetime.now(UTC).isoformat(),
                "generation": generation,
                "updates": updates,
                "replay_positions": len(replay),
                "new_positions": new_positions,
                "candidate": candidate.name,
                "loss": metrics.loss,
                "policy_loss": metrics.policy_loss,
                "value_loss": metrics.value_loss,
                "policy_entropy": metrics.policy_entropy,
            }
            _append_jsonl(paths["logs"] / "train.jsonl", training_record)
            _publish_worker_status(paths, "train", "waiting", **training_record)
            print(
                f"[train] candidate={candidate.name} new={new_positions} "
                f"replay={len(replay)} loss={metrics.loss:.5f}",
                flush=True,
            )
        elif not should_train:
            _publish_worker_status(
                paths,
                "train",
                "waiting_for_replay",
                generation=generation,
                updates=updates,
                replay_positions=len(replay),
                minimum_replay=config.min_replay,
                loaded_shards=len(loaded),
            )
        _atomic_json(
            state_path,
            {
                "format": PIPELINE_FORMAT,
                "trained_shards": sorted(trained_shards),
                "generation": generation,
                "updates": updates,
            },
        )
        if once:
            break
        time.sleep(config.poll_seconds)
    return generation


def _approve_candidate(paths: dict[str, Path], candidate: Path) -> None:
    destination = paths["approved"] / "current.pt"
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    shutil.copyfile(candidate, temporary)
    os.replace(temporary, destination)
    _atomic_json(
        paths["approved"] / "current.json",
        {"candidate": candidate.name, "approved_at": time.time()},
    )


def run_gatekeeper(config: PipelineConfig, *, once: bool = False) -> int:
    """Test candidate vs approved weights and atomically promote passing models."""
    paths = ensure_pipeline_dirs(config.root)
    state_path = paths["root"] / "gatekeeper-state.json"
    state = _read_json(
        state_path,
        {"format": PIPELINE_FORMAT, "evaluated": []},
    )
    evaluated = set(str(name) for name in state.get("evaluated", []))
    approved_count = 0
    _publish_worker_status(
        paths,
        "gatekeeper",
        "starting",
        evaluated=len(evaluated),
        approved=approved_count,
    )
    while not _stop_requested(paths["root"]):
        candidates = [
            path
            for path in sorted(paths["candidates"].glob("candidate-*.pt"))
            if path.name not in evaluated
        ]
        if candidates:
            candidate = candidates[0]
            _publish_worker_status(
                paths,
                "gatekeeper",
                "evaluating",
                candidate=candidate.name,
                evaluated=len(evaluated),
            )
            current = _approved_checkpoint(paths)
            if current is None:
                passed = True
                result: dict[str, Any] = {"bootstrap": True}
            else:
                state_type, expected_network = _network_config(config)
                challenger_model, _ = load_checkpoint(candidate, config.device)
                champion_model, _ = load_checkpoint(current, config.device)
                if challenger_model.config != expected_network or champion_model.config != expected_network:
                    raise ValueError("gatekeeper checkpoint does not match this pipeline")
                summary = evaluate_against(
                    lambda seed: state_type.initial(
                        config.players,
                        config.seed + 100_000 + seed,
                    ),
                    NetworkEvaluator(challenger_model, config.device),
                    NetworkEvaluator(champion_model, config.device),
                    num_players=config.players,
                    games=config.gate_games,
                    search_config=_search_config(config, config.seed + 700_000),
                )
                # `returns()` is the average pairwise result against every
                # opponent. Mapping [-1, 1] to [0, 1] gives a 0.5 baseline for
                # equally strong models in both three- and four-player games.
                match_score = (summary.mean_value + 1.0) / 2.0
                passed = match_score >= config.gate_threshold
                result = {
                    "games": summary.games,
                    "first_places": summary.first_places,
                    "draws": summary.draws,
                    "mean_value": summary.mean_value,
                    "mean_score": summary.mean_score,
                    "match_score": match_score,
                    "threshold": config.gate_threshold,
                }
            if passed:
                _approve_candidate(paths, candidate)
                approved_count += 1
            evaluated.add(candidate.name)
            _atomic_json(
                state_path,
                {"format": PIPELINE_FORMAT, "evaluated": sorted(evaluated)},
            )
            with (paths["logs"] / "gatekeeper.jsonl").open(
                "a",
                encoding="utf-8",
            ) as stream:
                stream.write(
                    json.dumps(
                        {"candidate": candidate.name, "passed": passed, **result},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            _publish_worker_status(
                paths,
                "gatekeeper",
                "waiting",
                candidate=candidate.name,
                result="approved" if passed else "rejected",
                evaluated=len(evaluated),
                approved=approved_count,
                **result,
            )
            print(
                f"[gatekeeper] {candidate.name} "
                f"{'approved' if passed else 'rejected'}",
                flush=True,
            )
        elif not candidates:
            _publish_worker_status(
                paths,
                "gatekeeper",
                "waiting_for_candidate",
                evaluated=len(evaluated),
                approved=approved_count,
            )
        if once:
            break
        time.sleep(config.poll_seconds)
    return approved_count


def _model_binary(model: PolicyValueNetwork, metadata: dict[str, Any]) -> bytes:
    tensors = [
        (name, tensor.detach().cpu().numpy().copy())
        for name, tensor in model.state_dict().items()
    ]
    manifest = {
        "format": "gaiazero-multiplayer-bin-v1",
        "architecture": model.architecture,
        "network_config": asdict(model.config),
        "tensor_count": len(tensors),
        "metadata": metadata,
    }
    header = json.dumps(
        manifest,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    output = io.BytesIO()
    output.write(EXPORT_MAGIC)
    output.write(struct.pack("<I", len(header)))
    output.write(header)
    for name, array in tensors:
        name_bytes = name.encode("utf-8")
        dtype_bytes = str(array.dtype).encode("ascii")
        output.write(struct.pack("<I", len(name_bytes)))
        output.write(name_bytes)
        output.write(struct.pack("<I", len(dtype_bytes)))
        output.write(dtype_bytes)
        output.write(struct.pack("<I", array.ndim))
        output.write(struct.pack("<" + "Q" * array.ndim, *array.shape))
        raw = array.tobytes(order="C")
        output.write(struct.pack("<Q", len(raw)))
        output.write(raw)
    return output.getvalue()


def export_checkpoint(checkpoint: Path, destination: Path) -> None:
    """Export a GaiaZero-native binary snapshot, not a KataGo checkpoint."""
    model, metadata = load_checkpoint(checkpoint, "cpu")
    _atomic_bytes(
        destination,
        _model_binary(model, {"source": str(checkpoint), **metadata}),
    )


def read_export_manifest(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        if stream.read(len(EXPORT_MAGIC)) != EXPORT_MAGIC:
            raise ValueError("not a GaiaZero multiplayer binary")
        size_bytes = stream.read(4)
        if len(size_bytes) != 4:
            raise ValueError("truncated GaiaZero multiplayer binary")
        size = struct.unpack("<I", size_bytes)[0]
        return json.loads(stream.read(size).decode("utf-8"))


def load_exported_model(
    path: Path,
    device: str = "auto",
) -> tuple[PolicyValueNetwork, dict[str, Any]]:
    """Load a GaiaZero `.bin` snapshot for inference."""
    with path.open("rb") as stream:
        if stream.read(len(EXPORT_MAGIC)) != EXPORT_MAGIC:
            raise ValueError("not a GaiaZero multiplayer binary")
        size_bytes = stream.read(4)
        if len(size_bytes) != 4:
            raise ValueError("truncated GaiaZero multiplayer binary")
        header_size = struct.unpack("<I", size_bytes)[0]
        header_bytes = stream.read(header_size)
        if len(header_bytes) != header_size:
            raise ValueError("truncated GaiaZero multiplayer manifest")
        manifest = json.loads(header_bytes.decode("utf-8"))
        model = PolicyValueNetwork(NetworkConfig(**manifest["network_config"]))
        expected = model.state_dict()
        tensors: dict[str, Any] = {}
        for _ in range(int(manifest["tensor_count"])):
            name_size_bytes = stream.read(4)
            if len(name_size_bytes) != 4:
                raise ValueError("truncated GaiaZero tensor name")
            name_size = struct.unpack("<I", name_size_bytes)[0]
            name = stream.read(name_size).decode("utf-8")
            dtype_size_bytes = stream.read(4)
            if len(dtype_size_bytes) != 4:
                raise ValueError("truncated GaiaZero tensor dtype")
            dtype_size = struct.unpack("<I", dtype_size_bytes)[0]
            dtype = np.dtype(stream.read(dtype_size).decode("ascii"))
            ndim_bytes = stream.read(4)
            if len(ndim_bytes) != 4:
                raise ValueError("truncated GaiaZero tensor rank")
            ndim = struct.unpack("<I", ndim_bytes)[0]
            shape_bytes = stream.read(8 * ndim)
            if len(shape_bytes) != 8 * ndim:
                raise ValueError("truncated GaiaZero tensor shape")
            shape = struct.unpack("<" + "Q" * ndim, shape_bytes)
            raw_size_bytes = stream.read(8)
            if len(raw_size_bytes) != 8:
                raise ValueError("truncated GaiaZero tensor length")
            raw_size = struct.unpack("<Q", raw_size_bytes)[0]
            raw = stream.read(raw_size)
            if len(raw) != raw_size:
                raise ValueError("truncated GaiaZero tensor data")
            if name not in expected:
                raise ValueError(f"unexpected tensor in GaiaZero binary: {name}")
            array = np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
            tensors[name] = expected[name].new_tensor(array)
        if set(tensors) != set(expected):
            missing = sorted(set(expected) - set(tensors))
            raise ValueError(f"GaiaZero binary is missing tensors: {missing}")
        model.load_state_dict(tensors)
        model.to(resolve_device(device))
        return model, dict(manifest.get("metadata", {}))


def run_export(config: PipelineConfig, *, once: bool = False) -> int:
    """Export each newly approved PyTorch model to GaiaZero `.bin`."""
    paths = ensure_pipeline_dirs(config.root)
    state_path = paths["root"] / "export-state.json"
    state = _read_json(
        state_path,
        {"format": PIPELINE_FORMAT, "sha256": "", "exports": 0},
    )
    exports = int(state.get("exports", 0))
    _publish_worker_status(paths, "export", "starting", exports=exports)
    while not _stop_requested(paths["root"]):
        source = _approved_checkpoint(paths)
        if source is not None:
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            if digest != state.get("sha256"):
                exports += 1
                destination = paths["exported"] / f"model-{exports:08d}.bin"
                export_checkpoint(source, destination)
                _atomic_bytes(
                    paths["exported"] / "current.bin",
                    destination.read_bytes(),
                )
                state = {
                    "format": PIPELINE_FORMAT,
                    "source": str(source),
                    "sha256": digest,
                    "exports": exports,
                }
                _atomic_json(state_path, state)
                _publish_worker_status(
                    paths,
                    "export",
                    "waiting",
                    exports=exports,
                    latest_model=destination.name,
                    source=str(source),
                    sha256=digest,
                )
                print(
                    f"[export] {source.name} -> {destination.name}",
                    flush=True,
                )
        if source is None:
            _publish_worker_status(
                paths,
                "export",
                "waiting_for_approved",
                exports=exports,
            )
        if once:
            break
        time.sleep(config.poll_seconds)
    return exports


def _worker_command(config_path: Path, worker: str) -> list[str]:
    return [
        sys.executable,
        "-m",
        "gaiazero.distributed",
        worker,
        "--config",
        str(config_path),
    ]


def run_pipeline(config: PipelineConfig) -> None:
    """Run the complete five-Python-process asynchronous loop."""
    paths = ensure_pipeline_dirs(config.root)
    (paths["root"] / "STOP").unlink(missing_ok=True)
    config_path = save_pipeline_config(config)
    names = ("selfplay", "shuffle", "train", "export", "gatekeeper")
    processes: list[tuple[str, subprocess.Popen[str], Any]] = []
    try:
        for name in names:
            log = (paths["logs"] / f"{name}.log").open("a", encoding="utf-8")
            process = subprocess.Popen(
                _worker_command(config_path, name),
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            processes.append((name, process, log))
        print("[pipeline] started " + ", ".join(names), flush=True)
        while not _stop_requested(paths["root"]):
            exited = [
                (name, process.returncode)
                for name, process, _log in processes
                if process.poll() is not None
            ]
            if exited:
                raise RuntimeError(f"pipeline worker exited unexpectedly: {exited}")
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        (paths["root"] / "STOP").touch()
        for _name, process, _log in processes:
            if process.poll() is None:
                process.terminate()
        for _name, process, log in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            log.close()


def _add_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", type=Path, default=Path("runs/multiplayer-pipeline"))
    parser.add_argument("--players", type=int, choices=(2, 3, 4), default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--simulations", type=int, default=64)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--temperature-moves", type=int, default=24)
    parser.add_argument("--max-moves", type=int, default=512)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--games-per-cycle", type=int, default=1)
    parser.add_argument("--shuffle-pack-size", type=int, default=4096)
    parser.add_argument("--replay-capacity", type=int, default=200_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--updates-per-cycle", type=int, default=32)
    parser.add_argument("--min-replay", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--residual-blocks", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--gate-games", type=int, default=20)
    parser.add_argument("--gate-threshold", type=float, default=0.55)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GaiaZero asynchronous multiplayer training workers"
    )
    parser.add_argument(
        "worker",
        choices=("selfplay", "shuffle", "train", "export", "gatekeeper"),
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--once", action="store_true")
    _add_config_arguments(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.config is not None:
        config = load_pipeline_config(args.config)
    else:
        values = {
            field: getattr(args, field)
            for field in PipelineConfig.__dataclass_fields__
        }
        config = PipelineConfig(**values)
    workers = {
        "selfplay": run_selfplay,
        "shuffle": run_shuffle,
        "train": run_train,
        "export": run_export,
        "gatekeeper": run_gatekeeper,
    }
    paths = ensure_pipeline_dirs(config.root)
    try:
        workers[args.worker](config, once=args.once)
    except Exception as error:
        _publish_worker_status(
            paths,
            args.worker,
            "failed",
            error_type=type(error).__name__,
            message=str(error),
        )
        raise
    _publish_worker_status(paths, args.worker, "stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
