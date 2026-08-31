"""Loader and typed accessors for ``configs/gaia-training.json``.

The JSON file is the source of truth for training-line settings.  The legacy
``PipelineConfig`` remains a small runtime snapshot written into each run
directory; it is produced from this loader and carries the normalized config
hash for auditability.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from gaiazero.contracts import ACTION_TUPLE_SCHEMA_VERSION, RULES_VERSION
from gaiazero.game.gaia_setup import SETUP_SEED_STREAM_VERSION


CONFIG_VERSION = 2
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "gaia-training.json"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _required_mapping(payload: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"gaia-training config section {key!r} must be an object")
    return dict(value)


@dataclass(frozen=True, slots=True)
class GaiaTrainingConfig:
    """Validated, immutable view over the training configuration document."""

    path: Path
    data: dict[str, Any]
    config_hash: str

    @classmethod
    def load(cls, path: str | Path | None = None) -> "GaiaTrainingConfig":
        selected = Path(
            path
            or os.environ.get("GAIA_TRAINING_CONFIG", str(DEFAULT_CONFIG_PATH))
        ).expanduser()
        if not selected.is_file():
            raise FileNotFoundError(f"training configuration not found: {selected}")
        try:
            payload = json.loads(selected.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON in training configuration: {selected}") from error
        if not isinstance(payload, Mapping):
            raise ValueError("gaia-training config root must be an object")
        data = json.loads(_canonical_json(payload))
        if int(data.get("config_version", -1)) != CONFIG_VERSION:
            raise ValueError(
                f"unsupported gaia-training config_version={data.get('config_version')!r}; "
                f"expected {CONFIG_VERSION}"
            )
        observation = _required_mapping(data, "observation_schema")
        action = _required_mapping(data, "action_schema")
        setup = _required_mapping(data, "setup_distribution")
        network = _required_mapping(data, "network")
        runtime = _required_mapping(data, "training_runtime")
        if "pipeline" in data and not isinstance(data["pipeline"], Mapping):
            raise ValueError("gaia-training config section 'pipeline' must be an object")
        if observation.get("version") != RULES_VERSION:
            raise ValueError("observation_schema.version must be standard-v22")
        if action.get("version") != ACTION_TUPLE_SCHEMA_VERSION:
            raise ValueError("action_schema.version must be action-tuple-v1")
        if setup.get("version") != SETUP_SEED_STREAM_VERSION:
            raise ValueError(
                "setup_distribution.version must match setup-seed-stream-v1"
            )
        if not str(network.get("network_config_id", "")):
            raise ValueError("network.network_config_id must not be empty")
        heads = network.get("heads")
        if heads != ["parameterized_policy", "pairwise_wdl", "vp_belief"]:
            raise ValueError(
                "network.heads must contain parameterized_policy, pairwise_wdl and vp_belief"
            )
        if runtime.get("mode") != "single_gpu":
            raise ValueError("only single_gpu training_runtime is supported")
        stream = _required_mapping(setup, "seed_stream")
        stream_names = stream.get("independent_streams")
        if not isinstance(stream_names, list) or not stream_names or len(set(stream_names)) != len(stream_names):
            raise ValueError("setup_distribution.seed_stream.independent_streams must be unique")
        if "map" not in stream_names:
            raise ValueError("setup_distribution.seed_stream must include the map stream")
        digest = hashlib.sha256(_canonical_json(data).encode("utf-8")).hexdigest()
        return cls(path=selected.resolve(), data=data, config_hash=digest)

    @property
    def rules_version(self) -> str:
        return str(self.data["observation_schema"]["version"])

    @property
    def seed_stream_version(self) -> str:
        return str(self.data["setup_distribution"]["seed_stream"]["version"])

    @property
    def action_schema_version(self) -> str:
        return str(self.data["action_schema"]["version"])

    @property
    def network_id(self) -> str:
        return str(self.data["network"]["network_config_id"])

    @property
    def network_capacity(self) -> dict[str, Any]:
        return dict(self.data["network"]["capacity"])

    @property
    def setup_distribution(self) -> dict[str, Any]:
        return dict(self.data["setup_distribution"])

    @property
    def seed_stream_names(self) -> tuple[str, ...]:
        streams = self.setup_distribution.get("seed_stream", {}).get(
            "independent_streams", []
        )
        return tuple(str(item) for item in streams)

    def setup_kwargs(self) -> dict[str, Any]:
        """Keyword arguments shared by every ``GaiaState.initial`` call."""

        return {"seed_stream_version": self.seed_stream_version}

    @property
    def search_settings(self) -> dict[str, Any]:
        return dict(self.data["search"])

    @property
    def training_settings(self) -> dict[str, Any]:
        return dict(self.data["training"])

    @property
    def runtime_settings(self) -> dict[str, Any]:
        return dict(self.data["training_runtime"])

    def network_settings(self, players: int) -> dict[str, Any]:
        profile = self.profile(players)
        return {
            "network_config_id": str(profile.get("network_id", self.network_id)),
            "architecture_family": str(self.data.get("architecture_family", "")),
            "capacity": self.network_capacity,
            "heads": list(self.data["network"].get("heads", [])),
            "player_count": int(players),
            "observation_schema_version": self.rules_version,
            "action_schema_version": self.action_schema_version,
        }

    def network_config(
        self,
        players: int,
        *,
        observation_size: int | None = None,
        action_size: int | None = None,
    ) -> Any:
        """Build the current Python network config from the JSON capacity."""

        from gaiazero.game import GaiaState
        from gaiazero.model import NetworkConfig, architecture_for_players

        if observation_size is None or action_size is None:
            template = GaiaState.initial(
                int(players),
                seed=0,
                seed_stream_version=self.seed_stream_version,
            )
            observation_size = template.observation_size
            action_size = template.action_size
        return NetworkConfig(
            observation_size=int(observation_size),
            action_size=int(action_size),
            num_players=int(players),
            hidden_size=int(self.network_capacity.get("hidden_size", 256)),
            residual_blocks=int(self.network_capacity.get("hybrid_blocks", 4)),
            architecture=architecture_for_players(int(players)),
        )

    def search_config(self, seed: int = 0) -> Any:
        from gaiazero.mcts import SearchConfig

        search = self.search_settings
        mcts = dict(search.get("mcts", {}))
        budgets = dict(search.get("budgets", {}))
        return SearchConfig(
            simulations=int(budgets.get("full_simulations", 128)),
            c_puct=float(mcts.get("c_puct", 1.5)),
            seed=int(seed),
        )

    def selfplay_config(self, seed: int = 0) -> Any:
        from gaiazero.selfplay import SelfPlayConfig

        pipeline = dict(self.data.get("pipeline", {}))
        return SelfPlayConfig(
            temperature_moves=int(pipeline.get("temperature_moves", 24)),
            max_moves=int(pipeline.get("max_moves", 512)),
            add_root_noise=bool(
                dict(self.search_settings.get("mcts", {})).get(
                    "root_noise_enabled", True
                )
            ),
            seed=int(seed),
        )

    def trainer_config(self) -> Any:
        from gaiazero.training import TrainerConfig

        training = self.training_settings
        return TrainerConfig(
            batch_size=int(training.get("batch_size", 256)),
            learning_rate=float(training.get("learning_rate", 1e-3)),
            weight_decay=float(training.get("weight_decay", 1e-4)),
            gradient_clip=float(training.get("gradient_clip_norm", 5.0)),
            device=str(self.runtime_settings.get("device", "auto")),
        )

    def manifest(self, players: int | None = None) -> dict[str, Any]:
        payload = {
            "config_version": CONFIG_VERSION,
            "config_hash": self.config_hash,
            "config_path": str(self.path),
            "rules_version": self.rules_version,
            "action_schema_version": self.action_schema_version,
            "seed_stream_version": self.seed_stream_version,
            "seed_stream_names": list(self.seed_stream_names),
            "architecture_family": self.data.get("architecture_family"),
        }
        if players is not None:
            payload["network"] = self.network_settings(players)
        return payload

    def profile(self, players: int) -> dict[str, Any]:
        profiles = self.data.get("player_profiles")
        if not isinstance(profiles, Mapping):
            raise ValueError("player_profiles section is required")
        value = profiles.get(str(int(players)))
        if not isinstance(value, Mapping):
            raise ValueError(f"no player profile configured for {players} players")
        if int(value.get("player_count", -1)) != int(players):
            raise ValueError(f"player profile {players} has an inconsistent player_count")
        return dict(value)

    def pipeline_values(
        self,
        players: int,
        *,
        root: str | Path | None = None,
        seed: int | None = None,
        overrides: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return normalized ``PipelineConfig`` keyword arguments.

        A small optional ``pipeline`` section can override operational values;
        otherwise values are derived from the documented search/training
        sections.  Explicit method arguments take precedence over JSON.
        """

        profile = self.profile(players)
        runtime = self.runtime_settings
        training = self.training_settings
        search = self.search_settings
        budgets = dict(search.get("budgets", {}))
        mcts = dict(search.get("mcts", {}))
        pipeline = dict(self.data.get("pipeline", {}))
        values: dict[str, Any] = {
            "root": Path(root or profile.get("pipeline_root", "runs/multiplayer-pipeline")),
            "players": int(players),
            "seed": int(pipeline.get("seed", self.data.get("seed", 0)) if seed is None else seed),
            "simulations": int(pipeline.get("simulations", budgets.get("full_simulations", 128))),
            "c_puct": float(pipeline.get("c_puct", mcts.get("c_puct", 1.5))),
            "dirichlet_alpha": float(pipeline.get("dirichlet_alpha", 0.3)),
            "root_noise_fraction": float(pipeline.get("root_noise_fraction", 0.25)),
            "add_root_noise": bool(
                pipeline.get(
                    "add_root_noise",
                    mcts.get("root_noise_enabled", True),
                )
            ),
            "temperature_moves": int(pipeline.get("temperature_moves", 24)),
            "max_moves": int(pipeline.get("max_moves", 512)),
            "poll_seconds": float(
                pipeline.get(
                    "poll_seconds",
                    self.data.get("data", {}).get("shuffle", {}).get("scan_interval_seconds", 10),
                )
            ),
            "games_per_cycle": int(pipeline.get("games_per_cycle", 1)),
            "shuffle_pack_size": int(
                pipeline.get(
                    "shuffle_pack_size",
                    self.data.get("data", {}).get("shuffle", {}).get("positions_per_shard", 4096),
                )
            ),
            "replay_capacity": int(pipeline.get("replay_capacity", 200_000)),
            "batch_size": int(pipeline.get("batch_size", training.get("batch_size", 256))),
            "updates_per_cycle": int(pipeline.get("updates_per_cycle", 32)),
            "min_replay": int(
                pipeline.get(
                    "min_replay",
                    self.data.get("data", {}).get("window", {}).get("min_positions", 200_000),
                )
            ),
            "learning_rate": float(pipeline.get("learning_rate", training.get("learning_rate", 1e-3))),
            "weight_decay": float(pipeline.get("weight_decay", training.get("weight_decay", 1e-4))),
            "hidden_size": int(self.network_capacity.get("hidden_size", 256)),
            "residual_blocks": int(
                pipeline.get(
                    "residual_blocks",
                    self.network_capacity.get("hybrid_blocks", 4),
                )
            ),
            "device": str(pipeline.get("device", runtime.get("device", "auto"))),
            "gate_games": int(pipeline.get("gate_games", 20)),
            "gate_threshold": float(pipeline.get("gate_threshold", 0.55)),
            "seed_stream_version": self.seed_stream_version,
            "training_config_path": str(self.path),
            "training_config_hash": self.config_hash,
            "network_config_id": str(profile.get("network_id", self.network_id)),
        }
        if overrides:
            for key, value in overrides.items():
                if key not in values:
                    raise ValueError(f"unknown PipelineConfig override: {key}")
                values[key] = value
        return values

    def pipeline_config(self, players: int, **kwargs: Any) -> Any:
        """Construct the runtime ``PipelineConfig`` without an import cycle."""

        from gaiazero.distributed import PipelineConfig

        return PipelineConfig(**self.pipeline_values(players, **kwargs))


def load_training_config(path: str | Path | None = None) -> GaiaTrainingConfig:
    return GaiaTrainingConfig.load(path)


load_gaia_training_config = load_training_config
TrainingConfig = GaiaTrainingConfig
