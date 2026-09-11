import numpy as np
import pytest

from gaiazero.game import GaiaState
from gaiazero.vp_offset import resolve_offsets, sample_zero_sum_perturbation, validate_offsets


def test_offsets_validate_bounds_and_zero_sum():
    assert validate_offsets(2, (30, -30)) == (30, -30)
    assert validate_offsets(4, (50, -20, -10, -20)) == (50, -20, -10, -20)
    with pytest.raises(ValueError):
        validate_offsets(2, (1, 0))
    with pytest.raises(ValueError):
        validate_offsets(3, (51, -25, -26))
    with pytest.raises(ValueError):
        validate_offsets(2, (True, -1))


def test_perturbation_is_deterministic_and_bounded():
    first = sample_zero_sum_perturbation(3, np.random.default_rng(17), max_abs_per_player=4)
    second = sample_zero_sum_perturbation(3, np.random.default_rng(17), max_abs_per_player=4)
    assert first == second
    assert sum(first) == 0
    assert max(abs(value) for value in first) <= 4


def test_state_applies_offset_and_exposes_it_in_observation_and_snapshot():
    plain = GaiaState.initial(2, seed=7)
    shifted = GaiaState.initial(
        2,
        seed=7,
        published_vp_offsets=(7, -7),
        vp_offset_perturbations=(2, -2),
    )
    assert [player.vp for player in shifted.players] == [19, 1]
    assert shifted.starting_vp_offsets == (9, -9)
    assert shifted.observation().shape == plain.observation().shape
    assert not np.array_equal(plain.observation(), shifted.observation())
    assert plain.setup_hash == shifted.setup_hash
    snapshot = shifted.snapshot()
    assert snapshot["published_vp_offsets"] == [7, -7]
    assert snapshot["vp_offset_perturbations"] == [2, -2]
    assert snapshot["starting_vp_offsets"] == [9, -9]
    assert snapshot["players"][0]["starting_vp_offset"] == 9


def test_resolve_rejects_mismatched_components():
    with pytest.raises(ValueError):
        resolve_offsets(2, starting=(3, -3), published=(2, -2), perturbation=(0, 0))
