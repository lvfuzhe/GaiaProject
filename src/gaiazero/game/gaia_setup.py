from __future__ import annotations

from dataclasses import dataclass

import numpy as np


MAX_SECTORS = 10
PLANET_SLOTS_PER_SECTOR = 7
MAX_PLANETS = MAX_SECTORS * PLANET_SLOTS_PER_SECTOR
BOOSTER_COUNT = 10

# Terrain integers mirror gaia_state.Terrain without introducing a circular import.
# Each entry is (q, r, terrain) in the printed sector's zero-degree orientation.
# The ten solid sides contain the standard 42 home, 12 Transdim and 7 Gaia
# planets. Sector 02 has seven planets; the remaining sectors have six.
SECTOR_PLANETS_SOLID: tuple[tuple[tuple[int, int, int], ...], ...] = (
    ((1, -2, 7), (0, -1, 0), (2, -1, 4), (2, 0, 3), (-1, 1, 2), (-1, 2, 1)),
    ((1, -2, 1), (0, -1, 6), (2, -1, 7), (-2, 0, 5), (-2, 1, 4), (0, 1, 2), (1, 1, 3)),
    ((1, -2, 5), (1, -1, 6), (-2, 0, 7), (2, 0, 1), (-1, 1, 8), (1, 1, 0)),
    ((2, -2, 0), (1, -1, 2), (-2, 0, 5), (-1, 0, 3), (0, 1, 4), (-1, 2, 6)),
    ((0, -2, 7), (1, -2, 3), (-2, 0, 6), (2, 0, 1), (-1, 1, 8), (1, 1, 4)),
    ((2, -2, 1), (-1, -1, 7), (0, -1, 0), (2, -1, 7), (1, 0, 8), (-1, 1, 2)),
    ((-1, -1, 2), (1, -1, 8), (-1, 0, 3), (2, 0, 5), (0, 1, 8), (-2, 2, 7)),
    ((0, -2, 7), (1, -1, 5), (-2, 0, 0), (-1, 0, 6), (0, 1, 4), (1, 1, 7)),
    ((0, -2, 6), (-1, -1, 7), (1, -1, 8), (-2, 1, 4), (0, 1, 5), (0, 2, 2)),
    ((0, -2, 7), (-1, -1, 7), (1, -1, 8), (-1, 1, 1), (1, 1, 3), (0, 2, 0)),
)

# In one- and two-player games sectors 05, 06 and 07 use their outlined side.
SECTOR_PLANETS_OUTLINED: dict[int, tuple[tuple[int, int, int], ...]] = {
    4: ((0, -2, 7), (1, -2, 3), (-2, 0, 6), (-1, 1, 8), (1, 1, 4)),
    5: ((2, -2, 1), (-1, -1, 7), (0, -1, 0), (2, -1, 7), (1, 0, 8)),
    6: ((1, -1, 2), (-1, 0, 8), (2, 0, 5), (0, 1, 8), (-2, 2, 7)),
}

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
    placement_order: tuple[int, ...]
    active_planets: tuple[bool, ...]
    planet_q: tuple[int, ...]
    planet_r: tuple[int, ...]
    planet_source_q: tuple[int, ...]
    planet_source_r: tuple[int, ...]
    planet_source_ids: tuple[int, ...]
    planet_source_catalog: tuple[tuple[int, int, int, int, int], ...]
    terrains: tuple[int, ...]
    planet_sectors: tuple[int, ...]
    sector_tiles: tuple[int, ...]
    sector_rotations: tuple[int, ...]
    sector_centers: tuple[tuple[int, int], ...]
    map_mode: str
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
    sector_tiles: tuple[int, ...] | None = None,
    sector_rotations: tuple[int, ...] | None = None,
    planet_positions: tuple[tuple[int, int, int], ...] | None = None,
    planet_layout: tuple[tuple[int, int, int, int], ...] | None = None,
    booster_tiles: tuple[int, ...] | None = None,
    round_scoring_tiles: tuple[int, ...] | None = None,
    final_scoring_tiles: tuple[int, ...] | None = None,
    standard_tech_tiles: tuple[int, ...] | None = None,
    advanced_tech_tiles: tuple[int, ...] | None = None,
    terraforming_federation_tile: int | None = None,
    map_mode: str = "bga-random",
) -> GaiaSetup:
    if not 2 <= num_players <= 4:
        raise ValueError("num_players must be between two and four")
    rng = np.random.default_rng(seed)
    if map_mode not in ("bga-random", "manual"):
        raise ValueError("map_mode must be 'bga-random' or 'manual'")
    centers = SECTOR_CENTERS_2P if num_players == 2 else SECTOR_CENTERS_34P
    tile_pool = np.arange(7 if num_players == 2 else MAX_SECTORS)
    outlined = num_players == 2
    random_map = _random_map(rng, tile_pool, centers, outlined=outlined)
    if (sector_tiles is None) != (sector_rotations is None):
        raise ValueError("sector_tiles and sector_rotations must be provided together")
    if map_mode == "manual" and sector_tiles is None:
        raise ValueError("manual map_mode requires sector tiles and rotations")
    if sector_tiles is None:
        map_data = random_map
    else:
        selected_sector_tiles = _validate_selection(
            sector_tiles,
            expected_count=len(centers),
            available_count=len(tile_pool),
            label="sector tiles",
            require_all=True,
        )
        selected_rotations = tuple(int(value) for value in sector_rotations)
        if len(selected_rotations) != len(centers):
            raise ValueError(f"sector rotations must contain {len(centers)} values")
        if any(rotation < 0 or rotation > 5 for rotation in selected_rotations):
            raise ValueError("sector rotations must be between zero and five")
        map_data = _assemble_map(
            centers,
            selected_sector_tiles,
            selected_rotations,
            outlined=outlined,
        )
        if not _map_is_valid(*map_data[:4]):
            raise ValueError(
                "sector arrangement is illegal: equal home planet types may not be adjacent"
            )

    planet_source_q = map_data[1]
    planet_source_r = map_data[2]
    planet_source_ids = tuple(range(MAX_PLANETS))
    planet_source_catalog = tuple(
        (
            planet,
            map_data[1][planet],
            map_data[2][planet],
            map_data[3][planet],
            map_data[4][planet],
        )
        for planet, active in enumerate(map_data[0])
        if active
    )
    if planet_positions is not None and planet_layout is not None:
        raise ValueError("planet positions and planet layout cannot both be provided")
    if planet_layout is not None:
        if map_mode != "manual":
            raise ValueError("planet layout requires manual map_mode")
        (
            active,
            planet_q,
            planet_r,
            terrains,
            sectors,
            planet_source_q,
            planet_source_r,
            planet_source_ids,
        ) = _apply_planet_layout(*map_data[:5], centers, planet_layout)
        map_data = (active, planet_q, planet_r, terrains, sectors, *map_data[5:])
    if planet_positions is not None:
        if map_mode != "manual":
            raise ValueError("planet positions require manual map_mode")
        planet_q, planet_r = _apply_planet_positions(
            map_data[0],
            map_data[1],
            map_data[2],
            map_data[3],
            centers,
            planet_positions,
        )
        map_data = (map_data[0], planet_q, planet_r, *map_data[3:])

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
    selected_first_player = seed % num_players if first_player is None else int(first_player)
    if not 0 <= selected_first_player < num_players:
        raise ValueError("first_player is out of range")
    for faction in selected_factions:
        home = faction_homes[faction]
        required = faction_starting_structures[faction]
        available = sum(
            active and terrain == home
            for active, terrain in zip(map_data[0], map_data[3], strict=True)
        )
        if available < required:
            raise RuntimeError(
                f"map has only {available} home planets for terrain {home}; "
                f"{required} are required"
            )
    placement_order = _starting_placement_order(
        selected_first_player,
        selected_factions,
        faction_starting_structures,
        num_players,
    )
    random_boosters = tuple(
        int(value)
        for value in rng.choice(BOOSTER_COUNT, size=num_players + 3, replace=False)
    )
    selected_boosters = random_boosters if booster_tiles is None else _validate_selection(
        booster_tiles,
        expected_count=num_players + 3,
        available_count=BOOSTER_COUNT,
        label="booster tiles",
    )
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

    random_round_scoring = tuple(
        int(value) for value in rng.choice(10, size=6, replace=False)
    )
    random_final_scoring = tuple(
        int(value) for value in rng.choice(6, size=2, replace=False)
    )
    random_standard_tech = tuple(int(value) for value in rng.permutation(9))
    random_advanced_tech = tuple(
        int(value) for value in rng.choice(15, size=6, replace=False)
    )
    random_federation = int(rng.integers(0, 6))
    selected_round_scoring = (
        random_round_scoring
        if round_scoring_tiles is None
        else _validate_selection(
            round_scoring_tiles,
            expected_count=6,
            available_count=10,
            label="round scoring tiles",
        )
    )
    selected_final_scoring = (
        random_final_scoring
        if final_scoring_tiles is None
        else _validate_selection(
            final_scoring_tiles,
            expected_count=2,
            available_count=6,
            label="final scoring tiles",
        )
    )
    selected_standard_tech = (
        random_standard_tech
        if standard_tech_tiles is None
        else _validate_selection(
            standard_tech_tiles,
            expected_count=9,
            available_count=9,
            label="standard technology tiles",
            require_all=True,
        )
    )
    selected_advanced_tech = (
        random_advanced_tech
        if advanced_tech_tiles is None
        else _validate_selection(
            advanced_tech_tiles,
            expected_count=6,
            available_count=15,
            label="advanced technology tiles",
        )
    )
    selected_federation = (
        random_federation
        if terraforming_federation_tile is None
        else int(terraforming_federation_tile)
    )
    if not 0 <= selected_federation < 6:
        raise ValueError("terraforming federation tile is out of range")

    return GaiaSetup(
        seed=seed,
        first_player=selected_first_player,
        faction_indices=selected_factions,
        starting_planets=tuple(() for _ in range(num_players)),
        placement_order=placement_order,
        active_planets=map_data[0],
        planet_q=map_data[1],
        planet_r=map_data[2],
        planet_source_q=planet_source_q,
        planet_source_r=planet_source_r,
        planet_source_ids=planet_source_ids,
        planet_source_catalog=planet_source_catalog,
        terrains=map_data[3],
        planet_sectors=map_data[4],
        sector_tiles=map_data[5],
        sector_rotations=map_data[6],
        sector_centers=tuple(centers),
        map_mode=map_mode,
        booster_owner=tuple(booster_owner),
        round_scoring_tiles=selected_round_scoring,
        final_scoring_tiles=selected_final_scoring,
        standard_tech_tiles=selected_standard_tech,
        advanced_tech_tiles=selected_advanced_tech,
        terraforming_federation_tile=selected_federation,
    )


def _validate_selection(
    values: tuple[int, ...],
    *,
    expected_count: int,
    available_count: int,
    label: str,
    require_all: bool = False,
) -> tuple[int, ...]:
    selected = tuple(int(value) for value in values)
    if len(selected) != expected_count:
        raise ValueError(f"{label} must contain {expected_count} values")
    if any(value < 0 or value >= available_count for value in selected):
        raise ValueError(f"{label} contain an out-of-range value")
    if len(set(selected)) != len(selected):
        raise ValueError(f"{label} must not contain duplicates")
    if require_all and set(selected) != set(range(available_count)):
        raise ValueError(f"{label} must contain every available tile")
    return selected


def hex_distance(aq: int, ar: int, bq: int, br: int) -> int:
    dq = aq - bq
    dr = ar - br
    return (abs(dq) + abs(dr) + abs(dq + dr)) // 2


def _random_map(
    rng: np.random.Generator,
    tile_pool: np.ndarray,
    centers: tuple[tuple[int, int], ...],
    *,
    outlined: bool,
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
        last = _assemble_map(
            centers,
            sector_tiles,
            rotations,
            outlined=outlined,
        )
        if _map_is_valid(*last[:4]):
            return last
    raise RuntimeError("unable to assemble a valid random sector map")


def _assemble_map(
    centers: tuple[tuple[int, int], ...],
    sector_tiles: tuple[int, ...],
    rotations: tuple[int, ...],
    *,
    outlined: bool,
) -> tuple[
    tuple[bool, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
]:
    active = [False] * MAX_PLANETS
    planet_q = [0] * MAX_PLANETS
    planet_r = [0] * MAX_PLANETS
    terrains = [0] * MAX_PLANETS
    sectors = [-1] * MAX_PLANETS
    for position, ((center_q, center_r), tile, rotation) in enumerate(
        zip(centers, sector_tiles, rotations, strict=True)
    ):
        planets = (
            SECTOR_PLANETS_OUTLINED[tile]
            if outlined and tile in SECTOR_PLANETS_OUTLINED
            else SECTOR_PLANETS_SOLID[tile]
        )
        for local, (q, r, terrain) in enumerate(planets):
            slot = position * PLANET_SLOTS_PER_SECTOR + local
            rotated_q, rotated_r = _rotate(q, r, rotation)
            active[slot] = True
            planet_q[slot] = center_q + rotated_q
            planet_r[slot] = center_r + rotated_r
            terrains[slot] = terrain
            sectors[slot] = tile + 1
    return (
        tuple(active),
        tuple(planet_q),
        tuple(planet_r),
        tuple(terrains),
        tuple(sectors),
        sector_tiles,
        rotations,
    )


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


def _apply_planet_positions(
    active: tuple[bool, ...],
    planet_q: tuple[int, ...],
    planet_r: tuple[int, ...],
    terrains: tuple[int, ...],
    centers: tuple[tuple[int, int], ...],
    positions: tuple[tuple[int, int, int], ...],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    active_ids = {index for index, is_active in enumerate(active) if is_active}
    try:
        normalized = tuple(
            (int(position[0]), int(position[1]), int(position[2]))
            for position in positions
        )
    except (IndexError, TypeError, ValueError) as error:
        raise ValueError("each planet position must contain id, q and r") from error
    ids = [planet for planet, _q, _r in normalized]
    if len(ids) != len(active_ids) or set(ids) != active_ids:
        raise ValueError("planet positions must contain every active planet exactly once")
    if len(set(ids)) != len(ids):
        raise ValueError("planet positions must not contain duplicate ids")

    board_spaces = {
        (center_q + local_q, center_r + local_r)
        for center_q, center_r in centers
        for local_q in range(-2, 3)
        for local_r in range(-2, 3)
        if max(abs(local_q), abs(local_r), abs(local_q + local_r)) <= 2
    }
    destinations = [(q, r) for _planet, q, r in normalized]
    if any(destination not in board_spaces for destination in destinations):
        raise ValueError("planet position is outside the assembled map")
    if len(set(destinations)) != len(destinations):
        raise ValueError("planet positions must not overlap")

    updated_q = list(planet_q)
    updated_r = list(planet_r)
    for planet, q, r in normalized:
        updated_q[planet] = q
        updated_r[planet] = r
    result_q = tuple(updated_q)
    result_r = tuple(updated_r)
    if not _map_is_valid(active, result_q, result_r, terrains):
        raise ValueError(
            "planet arrangement is illegal: equal home planet types may not be adjacent"
        )
    return result_q, result_r


def _apply_planet_layout(
    base_active: tuple[bool, ...],
    base_q: tuple[int, ...],
    base_r: tuple[int, ...],
    base_terrains: tuple[int, ...],
    base_sectors: tuple[int, ...],
    centers: tuple[tuple[int, int], ...],
    layout: tuple[tuple[int, int, int, int], ...],
) -> tuple[
    tuple[bool, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
]:
    try:
        normalized = tuple(
            (int(item[0]), int(item[1]), int(item[2]), int(item[3]))
            for item in layout
        )
    except (IndexError, TypeError, ValueError) as error:
        raise ValueError("each planet layout item must contain id, q, r and source_id") from error
    if not normalized:
        raise ValueError("planet layout must contain at least one planet")
    if len(normalized) > MAX_PLANETS:
        raise ValueError(f"planet layout supports at most {MAX_PLANETS} planets")

    ids = [planet for planet, _q, _r, _source in normalized]
    if any(planet < 0 or planet >= MAX_PLANETS for planet in ids):
        raise ValueError(f"planet id must be between 0 and {MAX_PLANETS - 1}")
    if len(set(ids)) != len(ids):
        raise ValueError("planet layout must not contain duplicate ids")
    if any(
        source < 0 or source >= MAX_PLANETS or not base_active[source]
        for _planet, _q, _r, source in normalized
    ):
        raise ValueError("planet layout source_id must reference a printed sector planet")

    board_spaces = {
        (center_q + local_q, center_r + local_r)
        for center_q, center_r in centers
        for local_q in range(-2, 3)
        for local_r in range(-2, 3)
        if max(abs(local_q), abs(local_r), abs(local_q + local_r)) <= 2
    }
    destinations = [(q, r) for _planet, q, r, _source in normalized]
    if any(destination not in board_spaces for destination in destinations):
        raise ValueError("planet position is outside the assembled map")
    if len(set(destinations)) != len(destinations):
        raise ValueError("planet layout must not overlap")

    active = [False] * MAX_PLANETS
    planet_q = [0] * MAX_PLANETS
    planet_r = [0] * MAX_PLANETS
    terrains = [0] * MAX_PLANETS
    sectors = [-1] * MAX_PLANETS
    source_q = [0] * MAX_PLANETS
    source_r = [0] * MAX_PLANETS
    source_ids = [-1] * MAX_PLANETS
    for planet, q, r, source in normalized:
        active[planet] = True
        planet_q[planet] = q
        planet_r[planet] = r
        terrains[planet] = base_terrains[source]
        sectors[planet] = base_sectors[source]
        source_q[planet] = base_q[source]
        source_r[planet] = base_r[source]
        source_ids[planet] = source
    result = (
        tuple(active),
        tuple(planet_q),
        tuple(planet_r),
        tuple(terrains),
        tuple(sectors),
        tuple(source_q),
        tuple(source_r),
        tuple(source_ids),
    )
    if not _map_is_valid(*result[:4]):
        raise ValueError(
            "planet arrangement is illegal: equal home planet types may not be adjacent"
        )
    return result


def _starting_placement_order(
    first_player: int,
    faction_indices: tuple[int, ...],
    faction_starting_structures: tuple[int, ...],
    num_players: int,
) -> tuple[int, ...]:
    """Return the player sequence for the snake-shaped starting placement."""
    forward = tuple((first_player + offset) % num_players for offset in range(num_players))
    max_structures = max(
        faction_starting_structures[faction] for faction in faction_indices
    )
    order: list[int] = []
    for layer in range(max_structures):
        sequence = forward if layer % 2 == 0 else tuple(reversed(forward))
        order.extend(
            player
            for player in sequence
            if faction_starting_structures[faction_indices[player]] > layer
        )
    return tuple(order)


def _rotate(q: int, r: int, steps: int) -> tuple[int, int]:
    for _ in range(steps % 6):
        q, r = -r, q + r
    return q, r
