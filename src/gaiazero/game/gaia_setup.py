from __future__ import annotations

from dataclasses import dataclass

import numpy as np


MAX_SECTORS = 10
PLANETS_PER_SECTOR = 4
MAX_PLANETS = MAX_SECTORS * PLANETS_PER_SECTOR
BOOSTER_COUNT = 10

# Terrain integers mirror gaia_state.Terrain without introducing a circular import.
SECTOR_TERRAINS: tuple[tuple[int, ...], ...] = (
    (0, 1, 2, 7),
    (3, 4, 5, 8),
    (6, 0, 1, 7),
    (2, 3, 4, 8),
    (5, 6, 0, 7),
    (1, 2, 3, 8),
    (4, 5, 6, 7),
    (0, 2, 5, 8),
    (1, 3, 6, 7),
    (4, 2, 0, 8),
)

SECTOR_LOCAL_POSITIONS: tuple[tuple[tuple[int, int], ...], ...] = (
    ((-1, 0), (1, -1), (0, 1), (2, 0)),
    ((0, -2), (1, 0), (-1, 1), (2, -1)),
    ((-2, 1), (0, -1), (1, 1), (2, -2)),
    ((-1, -1), (1, 0), (-1, 2), (2, -1)),
    ((0, -2), (-1, 0), (1, 1), (2, -1)),
    ((-2, 0), (0, 1), (1, -1), (2, 0)),
    ((-1, 1), (0, -1), (2, 0), (1, -2)),
    ((-2, 1), (0, -2), (1, 0), (1, 1)),
    ((-1, -1), (-1, 1), (1, 0), (2, -2)),
    ((-2, 0), (0, -1), (1, 1), (2, -1)),
)

# (5, -2) and its rotations align the three-cell edges of two radius-2 sectors.
SECTOR_CENTERS_2P: tuple[tuple[int, int], ...] = (
    (0, 0),
    (3, -5),
    (5, -2),
    (2, 3),
    (-3, 5),
    (-5, 2),
    (-2, -3),
)

SECTOR_CENTERS_34P: tuple[tuple[int, int], ...] = (
    (-4, -2),
    (1, -4),
    (6, -6),
    (-7, 3),
    (-2, 1),
    (3, -1),
    (8, -3),
    (-5, 6),
    (0, 4),
    (5, 2),
)


@dataclass(frozen=True, slots=True)
class GaiaSetup:
    seed: int
    first_player: int
    faction_indices: tuple[int, ...]
    starting_planets: tuple[tuple[int, ...], ...]
    active_planets: tuple[bool, ...]
    planet_q: tuple[int, ...]
    planet_r: tuple[int, ...]
    terrains: tuple[int, ...]
    planet_sectors: tuple[int, ...]
    sector_tiles: tuple[int, ...]
    sector_rotations: tuple[int, ...]
    sector_centers: tuple[tuple[int, int], ...]
    booster_owner: tuple[int, ...]
    round_scoring_tiles: tuple[int, ...]
    final_scoring_tiles: tuple[int, ...]
    standard_tech_tiles: tuple[int, ...]
    advanced_tech_tiles: tuple[int, ...]
    terraforming_federation_tile: int


def generate_setup(
    num_players: int,
    seed: int,
    *,
    faction_boards: tuple[tuple[int, int], ...],
    faction_homes: tuple[int, ...],
    faction_starting_structures: tuple[int, ...],
    faction_indices: tuple[int, ...] | None = None,
    first_player: int | None = None,
) -> GaiaSetup:
    if not 2 <= num_players <= 4:
        raise ValueError("num_players must be between two and four")
    rng = np.random.default_rng(seed)
    centers = SECTOR_CENTERS_2P if num_players == 2 else SECTOR_CENTERS_34P
    tile_pool = np.arange(7 if num_players == 2 else MAX_SECTORS)
    map_data = _random_map(rng, tile_pool, centers)

    if faction_indices is None:
        board_choices = rng.choice(len(faction_boards), size=num_players, replace=False)
        selected_factions = tuple(
            int(faction_boards[int(board)][int(rng.integers(0, 2))])
            for board in board_choices
        )
    else:
        selected_factions = tuple(int(faction) for faction in faction_indices)
        if len(selected_factions) != num_players:
            raise ValueError("one faction is required for each player")
        faction_to_board = {
            faction: board
            for board, factions in enumerate(faction_boards)
            for faction in factions
        }
        if any(faction not in faction_to_board for faction in selected_factions):
            raise ValueError("faction index is out of range")
        selected_boards = [faction_to_board[faction] for faction in selected_factions]
        if len(set(selected_boards)) != len(selected_boards):
            raise ValueError("selected factions must use different double-sided boards")
    starting_planets = tuple(
        _choose_starting_planets(
            rng,
            map_data[0],
            map_data[1],
            map_data[2],
            map_data[3],
            faction_homes[faction],
            faction_starting_structures[faction],
        )
        for faction in selected_factions
    )

    selected_first_player = seed % num_players if first_player is None else int(first_player)
    if not 0 <= selected_first_player < num_players:
        raise ValueError("first_player is out of range")
    selected_boosters = [
        int(value)
        for value in rng.choice(BOOSTER_COUNT, size=num_players + 3, replace=False)
    ]
    booster_owner = [-2] * BOOSTER_COUNT
    for booster in selected_boosters:
        booster_owner[booster] = -1
    turn_order = [
        (selected_first_player + offset) % num_players
        for offset in range(num_players)
    ]
    for player, booster in zip(
        reversed(turn_order),
        selected_boosters[:num_players],
        strict=True,
    ):
        booster_owner[booster] = player

    return GaiaSetup(
        seed=seed,
        first_player=selected_first_player,
        faction_indices=selected_factions,
        starting_planets=starting_planets,
        active_planets=map_data[0],
        planet_q=map_data[1],
        planet_r=map_data[2],
        terrains=map_data[3],
        planet_sectors=map_data[4],
        sector_tiles=map_data[5],
        sector_rotations=map_data[6],
        sector_centers=tuple(centers),
        booster_owner=tuple(booster_owner),
        round_scoring_tiles=tuple(
            int(value) for value in rng.choice(10, size=6, replace=False)
        ),
        final_scoring_tiles=tuple(
            int(value) for value in rng.choice(6, size=2, replace=False)
        ),
        standard_tech_tiles=tuple(int(value) for value in rng.permutation(9)),
        advanced_tech_tiles=tuple(
            int(value) for value in rng.choice(15, size=6, replace=False)
        ),
        terraforming_federation_tile=int(rng.integers(0, 6)),
    )


def hex_distance(aq: int, ar: int, bq: int, br: int) -> int:
    dq = aq - bq
    dr = ar - br
    return (abs(dq) + abs(dr) + abs(dq + dr)) // 2


def _random_map(
    rng: np.random.Generator,
    tile_pool: np.ndarray,
    centers: tuple[tuple[int, int], ...],
) -> tuple[
    tuple[bool, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
]:
    last: tuple[
        tuple[bool, ...],
        tuple[int, ...],
        tuple[int, ...],
        tuple[int, ...],
        tuple[int, ...],
        tuple[int, ...],
        tuple[int, ...],
    ] | None = None
    for _ in range(2_000):
        sector_tiles = tuple(int(value) for value in rng.permutation(tile_pool))
        rotations = tuple(int(value) for value in rng.integers(0, 6, size=len(centers)))
        active = [False] * MAX_PLANETS
        planet_q = [0] * MAX_PLANETS
        planet_r = [0] * MAX_PLANETS
        terrains = [0] * MAX_PLANETS
        sectors = [-1] * MAX_PLANETS
        for position, ((center_q, center_r), tile, rotation) in enumerate(
            zip(centers, sector_tiles, rotations, strict=True)
        ):
            for local, ((q, r), terrain) in enumerate(
                zip(SECTOR_LOCAL_POSITIONS[tile], SECTOR_TERRAINS[tile], strict=True)
            ):
                slot = position * PLANETS_PER_SECTOR + local
                rotated_q, rotated_r = _rotate(q, r, rotation)
                active[slot] = True
                planet_q[slot] = center_q + rotated_q
                planet_r[slot] = center_r + rotated_r
                terrains[slot] = terrain
                sectors[slot] = tile + 1
        last = (
            tuple(active),
            tuple(planet_q),
            tuple(planet_r),
            tuple(terrains),
            tuple(sectors),
            sector_tiles,
            rotations,
        )
        if _map_is_valid(*last[:4]):
            return last
    raise RuntimeError("unable to assemble a valid random sector map")


def _map_is_valid(
    active: tuple[bool, ...],
    planet_q: tuple[int, ...],
    planet_r: tuple[int, ...],
    terrains: tuple[int, ...],
) -> bool:
    positions = [
        (planet_q[index], planet_r[index])
        for index, is_active in enumerate(active)
        if is_active
    ]
    if len(set(positions)) != len(positions):
        return False
    active_indices = [index for index, is_active in enumerate(active) if is_active]
    return not any(
        terrains[left] == terrains[right]
        and terrains[left] < 7
        and hex_distance(
            planet_q[left],
            planet_r[left],
            planet_q[right],
            planet_r[right],
        ) == 1
        for offset, left in enumerate(active_indices)
        for right in active_indices[offset + 1 :]
    )


def _choose_starting_planets(
    rng: np.random.Generator,
    active: tuple[bool, ...],
    planet_q: tuple[int, ...],
    planet_r: tuple[int, ...],
    terrains: tuple[int, ...],
    home: int,
    count: int,
) -> tuple[int, ...]:
    candidates = [
        index
        for index, is_active in enumerate(active)
        if is_active and terrains[index] == home
    ]
    if len(candidates) < count:
        raise RuntimeError(f"random map has only {len(candidates)} home planets for terrain {home}")
    rng.shuffle(candidates)
    chosen = [candidates.pop()]
    while len(chosen) < count:
        best_distance = max(
            min(
                hex_distance(
                    planet_q[candidate],
                    planet_r[candidate],
                    planet_q[selected],
                    planet_r[selected],
                )
                for selected in chosen
            )
            for candidate in candidates
        )
        choices = [
            candidate
            for candidate in candidates
            if min(
                hex_distance(
                    planet_q[candidate],
                    planet_r[candidate],
                    planet_q[selected],
                    planet_r[selected],
                )
                for selected in chosen
            ) == best_distance
        ]
        selected = choices[int(rng.integers(0, len(choices)))]
        candidates.remove(selected)
        chosen.append(selected)
    return tuple(chosen)


def _rotate(q: int, r: int, steps: int) -> tuple[int, int]:
    for _ in range(steps % 6):
        q, r = -r, q + r
    return q, r
