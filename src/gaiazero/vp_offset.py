"""Offline VP-offset contract shared by setup, self-play and replay export.

Offsets are integer, seat-ordered compensation values.  They are deliberately
kept separate from the auction flow: a training game starts immediately with
``10 + starting_vp_offset`` VP and no auction actions.
"""

from __future__ import annotations

from itertools import product
import hashlib
from typing import Sequence

import numpy as np


COMPENSATION_MODE = "offline-vp-offset"
COMPENSATION_VERSION = "offsets-v0"
BASE_INITIAL_VP = 10
OFFSET_LIMITS = {2: 30, 3: 50, 4: 50}
OBSERVATION_SCALES = {2: 30.0, 3: 50.0, 4: 50.0}


def offset_limit(player_count: int) -> int:
    try:
        return OFFSET_LIMITS[int(player_count)]
    except KeyError as error:
        raise ValueError("player_count must be two, three or four") from error


def observation_scale(player_count: int) -> float:
    try:
        return OBSERVATION_SCALES[int(player_count)]
    except KeyError as error:
        raise ValueError("player_count must be two, three or four") from error


def validate_offsets(
    player_count: int,
    offsets: Sequence[int],
    *,
    name: str = "offsets",
) -> tuple[int, ...]:
    """Normalize and validate a seat-ordered integer zero-sum offset vector."""

    count = int(player_count)
    limit = offset_limit(count)
    values = tuple(offsets)
    if len(values) != count:
        raise ValueError(f"{name} must contain exactly {count} values")
    normalized: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must contain integers")
        item = int(value)
        if abs(item) > limit:
            raise ValueError(f"{name} values must be within +/-{limit} for {count} players")
        normalized.append(item)
    if sum(normalized) != 0:
        raise ValueError(f"{name} must sum to zero")
    return tuple(normalized)


def zero_offsets(player_count: int) -> tuple[int, ...]:
    return validate_offsets(player_count, (0,) * int(player_count))


def sample_zero_sum_perturbation(
    player_count: int,
    rng: np.random.Generator,
    *,
    max_abs_per_player: int = 4,
    base_offsets: Sequence[int] | None = None,
) -> tuple[int, ...]:
    """Sample uniformly from bounded integer zero-sum vectors.

    Enumerating the small bounded space avoids rejection-sampling bias and is
    deterministic for a versioned NumPy seed stream.
    """

    count = int(player_count)
    limit = min(offset_limit(count), int(max_abs_per_player))
    if limit < 0:
        raise ValueError("max_abs_per_player must be non-negative")
    base = validate_offsets(count, base_offsets, name="published_vp_offsets") if base_offsets is not None else (0,) * count
    candidates = [
        values
        for values in product(range(-limit, limit + 1), repeat=count)
        if sum(values) == 0
        and all(abs(base[index] + values[index]) <= offset_limit(count) for index in range(count))
    ]
    if not candidates:
        raise ValueError("no bounded zero-sum perturbation exists")
    return tuple(int(value) for value in candidates[int(rng.integers(len(candidates)))])


def perturbation_rng(seed: int, compensation_version: str = COMPENSATION_VERSION) -> np.random.Generator:
    """Derive a reproducible offset-only RNG without consuming map streams."""

    material = f"vp-offset|{compensation_version}|{int(seed)}".encode("utf-8")
    derived = int.from_bytes(hashlib.sha256(material).digest()[:8], "little")
    return np.random.default_rng(derived)


def resolve_offsets(
    player_count: int,
    *,
    starting: Sequence[int] | None = None,
    published: Sequence[int] | None = None,
    perturbation: Sequence[int] | None = None,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Resolve ``published + perturbation`` and return all three vectors."""

    count = int(player_count)
    if starting is not None:
        actual = validate_offsets(count, starting, name="starting_vp_offsets")
        if published is None:
            published_values = actual
        else:
            published_values = validate_offsets(count, published, name="published_vp_offsets")
        if perturbation is None:
            perturbation_values = tuple(actual[i] - published_values[i] for i in range(count))
        else:
            perturbation_values = validate_offsets(count, perturbation, name="vp_offset_perturbations")
        if tuple(published_values[i] + perturbation_values[i] for i in range(count)) != actual:
            raise ValueError("starting_vp_offsets must equal published_vp_offsets + vp_offset_perturbations")
        return published_values, perturbation_values, actual
    published_values = validate_offsets(
        count, published if published is not None else zero_offsets(count), name="published_vp_offsets"
    )
    perturbation_values = validate_offsets(
        count, perturbation if perturbation is not None else zero_offsets(count), name="vp_offset_perturbations"
    )
    actual = validate_offsets(
        count,
        tuple(published_values[i] + perturbation_values[i] for i in range(count)),
        name="starting_vp_offsets",
    )
    return published_values, perturbation_values, actual
