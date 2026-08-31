"""Versioned data contracts shared by rules, self-play and training.

The Python implementation still exposes the legacy integer action vector to
the existing MCTS.  ``ActionTuple`` is the stable semantic representation;
integer IDs are only a compatibility/audit key and must never be used as a
network-facing contract.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Any, Mapping

import numpy as np


RULES_VERSION = "standard-v22"
ACTION_TUPLE_SCHEMA_VERSION = "action-tuple-v1"
NPZ_TRAJECTORY_SCHEMA_VERSION = "npz-trajectory-v1"
NPZ_SHARD_SCHEMA_VERSION = "npz-shard-v1"
STATE_HASH_VERSION = "state-hash-v1"
MAX_ACTION_ARGUMENTS = 8

# Short aliases used by integrations that refer to a generic schema version.
STANDARD_RULES_VERSION = RULES_VERSION
ACTION_SCHEMA_VERSION = ACTION_TUPLE_SCHEMA_VERSION
NPZ_SCHEMA_VERSION = NPZ_TRAJECTORY_SCHEMA_VERSION
STATE_HASH_SCHEMA_VERSION = STATE_HASH_VERSION

# The IDs are stable across games and independent of the legacy action array.
# Keeping the table explicit makes manifests and cross-language implementations
# deterministic.  Unknown/legacy actions use the final reserved entry.
ACTION_TYPE_IDS: dict[str, int] = {
    name: index
    for index, name in enumerate(
        (
            "build_mine",
            "place_starting_structure",
            "gaia_project",
            "upgrade_trading",
            "upgrade_lab",
            "upgrade_planetary_institute",
            "upgrade_academy",
            "upgrade_qic_academy",
            "research",
            "power_action",
            "tech_take",
            "federation",
            "qic_academy",
            "standard_tech",
            "advanced_tech",
            "qic_tech",
            "qic_federation",
            "qic_planet_types",
            "booster_terraform",
            "booster_range",
            "pass_booster",
            "pass_final",
            "skip_tech_research",
            "brainstone",
            "terrans_gaia_credit",
            "terrans_gaia_ore",
            "terrans_gaia_knowledge",
            "terrans_gaia_qic",
            "terrans_gaia_finish",
            "taklons_passive_before",
            "taklons_passive_after",
            "ivits_space_station",
            "bal_taks_gaiaformer_qic",
            "bescods_research",
            "itars_burn_power",
            "itars_gaia_technology",
            "itars_gaia_finish",
            "nevlas_power_to_gaia",
            "nevlas_credits",
            "nevlas_credit_ore",
            "nevlas_ore",
            "nevlas_qic",
            "nevlas_knowledge",
            "lost_planet",
            "passive_charge_accept",
            "passive_charge_decline",
            "power_to_credit",
            "power_to_ore",
            "power_to_knowledge",
            "power_to_qic",
            "qic_to_ore",
            "ore_to_credit",
            "knowledge_to_credit",
            "legacy_action",
        )
    )
}


@dataclass(frozen=True, slots=True)
class ActionTuple:
    """Canonical semantic action independent of a fixed action array index."""

    action_type: str
    args: tuple[int, ...] = ()
    schema_version: str = ACTION_TUPLE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ACTION_TUPLE_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported action tuple schema: {self.schema_version}"
            )
        if not self.action_type:
            raise ValueError("action_type must not be empty")
        normalized = tuple(int(value) for value in self.args)
        if len(normalized) > MAX_ACTION_ARGUMENTS:
            raise ValueError(
                f"ActionTuple supports at most {MAX_ACTION_ARGUMENTS} arguments"
            )
        object.__setattr__(self, "args", normalized)

    @property
    def action_type_id(self) -> int:
        return ACTION_TYPE_IDS.get(
            self.action_type,
            ACTION_TYPE_IDS["legacy_action"],
        )

    @property
    def canonical_key(self) -> tuple[str, tuple[int, ...]]:
        return self.action_type, self.args

    @property
    def parameter_types(self) -> tuple[str, ...]:
        """Parameter type tags in canonical slot order.

        The current rule engine uses integer entity IDs (planet, track, tile,
        or board-space indices).  Keeping the type tags explicit leaves room
        for a future string/enum slot without changing the tuple envelope.
        """

        return ("int",) * len(self.args)

    @property
    def parameter_mask(self) -> tuple[bool, ...]:
        return (True,) * len(self.args)

    def padded_arguments(self, width: int = MAX_ACTION_ARGUMENTS) -> tuple[int, ...]:
        if width < len(self.args):
            raise ValueError("width cannot be smaller than the argument count")
        return self.args + (-1,) * (width - len(self.args))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "action_type": self.action_type,
            "action_type_id": self.action_type_id,
            "args": list(self.args),
            "parameter_types": list(self.parameter_types),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ActionTuple:
        if not isinstance(value, Mapping):
            raise TypeError("ActionTuple must be decoded from a mapping")
        item = cls(
            action_type=str(value.get("action_type", "")),
            args=tuple(int(item) for item in value.get("args", ())),
            schema_version=str(
                value.get("schema_version", ACTION_TUPLE_SCHEMA_VERSION)
            ),
        )
        declared_types = value.get("parameter_types")
        if declared_types is not None and tuple(str(item) for item in declared_types) != item.parameter_types:
            raise ValueError("ActionTuple parameter_types do not match args")
        return item

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    @classmethod
    def from_json(cls, value: str) -> ActionTuple:
        decoded = json.loads(value)
        return cls.from_dict(decoded)


class ActionRegistry:
    """Small versioned registry shared by Python/C++ action encoders."""

    schema_version = ACTION_TUPLE_SCHEMA_VERSION
    max_parameter_slots = MAX_ACTION_ARGUMENTS

    @staticmethod
    def type_id(action_type: str) -> int:
        try:
            return ACTION_TYPE_IDS[action_type]
        except KeyError as error:
            raise ValueError(f"unknown action type: {action_type}") from error

    @staticmethod
    def canonical(action: ActionTuple) -> ActionTuple:
        if not isinstance(action, ActionTuple):
            raise TypeError("action must be an ActionTuple")
        return ActionTuple(action.action_type, tuple(action.args))

    @staticmethod
    def key(action: ActionTuple) -> tuple[str, tuple[int, ...]]:
        return ActionRegistry.canonical(action).canonical_key


def canonicalize(value: Any) -> Any:
    """Convert state/dataclass values to deterministic JSON-compatible data."""

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: canonicalize(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, (Enum, IntEnum)):
        return canonicalize(value.value)
    if isinstance(value, np.ndarray):
        return canonicalize(value.tolist())
    if isinstance(value, Mapping):
        return {
            str(key): canonicalize(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [canonicalize(item) for item in value]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(
        canonicalize(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def state_hash(state: Any) -> str:
    """Return the stable digest of every dataclass field affecting the state."""

    payload = {
        "state_hash_version": STATE_HASH_VERSION,
        "rules_version": RULES_VERSION,
        "state": canonicalize(state),
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def trajectory_metadata(
    *,
    trace_length: int,
    terminal_valid: bool,
    game_id: str | None = None,
) -> dict[str, Any]:
    """Build the metadata fragment required by ``npz-trajectory-v1``."""

    payload: dict[str, Any] = {
        "schema_version": NPZ_TRAJECTORY_SCHEMA_VERSION,
        "rules_version": RULES_VERSION,
        "action_schema_version": ACTION_TUPLE_SCHEMA_VERSION,
        "state_hash_version": STATE_HASH_VERSION,
        "trace_alignment": "pre_action_positions_plus_terminal",
        "trace_length": int(trace_length),
        "terminal_valid": bool(terminal_valid),
    }
    if game_id is not None:
        payload["game_id"] = str(game_id)
    return payload


def compose_parameterized_policy(
    action_type_logits: np.ndarray,
    argument_logits: np.ndarray,
    legal_actions: list[ActionTuple] | tuple[ActionTuple, ...],
) -> np.ndarray:
    """Compose conditional type/slot logits for the legal tuple list.

    ``action_type_logits`` has shape ``[C]`` and ``argument_logits`` has shape
    ``[R,V]``.  Each tuple receives the sum of its type logit and the logits of
    its populated argument slots, then a single softmax is taken over legal
    tuples.  This is the exact aggregation used by the action contract and is
    independent of legacy action IDs.
    """

    type_values = np.asarray(action_type_logits, dtype=np.float64)
    argument_values = np.asarray(argument_logits, dtype=np.float64)
    if type_values.ndim != 1:
        raise ValueError("action_type_logits must have shape [C]")
    if argument_values.ndim != 2:
        raise ValueError("argument_logits must have shape [R, V]")
    if len(legal_actions) == 0:
        return np.zeros(0, dtype=np.float32)
    scores = np.full(len(legal_actions), -np.inf, dtype=np.float64)
    for index, action in enumerate(legal_actions):
        type_id = action.action_type_id
        if type_id >= len(type_values):
            continue
        score = float(type_values[type_id])
        valid = True
        for slot, argument in enumerate(action.args):
            if slot >= argument_values.shape[0] or argument < 0 or argument >= argument_values.shape[1]:
                valid = False
                break
            score += float(argument_values[slot, argument])
        if valid:
            scores[index] = score
    finite = np.isfinite(scores)
    if not np.any(finite):
        raise ValueError("no legal ActionTuple can be represented by the logits")
    maximum = float(np.max(scores[finite]))
    probabilities = np.zeros_like(scores)
    probabilities[finite] = np.exp(scores[finite] - maximum)
    probabilities[finite] /= float(np.sum(probabilities[finite]))
    return probabilities.astype(np.float32)
