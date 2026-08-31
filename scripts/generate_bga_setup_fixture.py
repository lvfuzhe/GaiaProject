"""Generate the versioned map-only BGA setup golden fixture.

The fixture intentionally records only the star map.  Factions, boosters,
scoring tiles and technology tiles remain independently randomized by the
normal setup seed streams.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from gaiazero.contracts import RULES_VERSION, canonical_json
from gaiazero.game import GaiaState
from gaiazero.game.gaia_setup import SETUP_SEED_STREAM_VERSION


VARIANTS: tuple[tuple[str, int, str], ...] = (
    ("2p-reduced", 2, "reduced"),
    ("3p-reduced", 3, "reduced"),
    ("3p-normal", 3, "normal"),
    ("4p-normal", 4, "normal"),
)


def map_payload(state: GaiaState) -> dict[str, object]:
    active_ids = [
        index for index, active in enumerate(state.active_planets) if active
    ]
    return {
        "map_size": state.map_size,
        "sector_tiles": list(state.sector_tiles),
        "sector_rotations": list(state.sector_rotations),
        "sector_centers": [list(center) for center in state.sector_centers],
        "outlined_sector_tiles": [4, 5, 6] if state.num_players == 2 else [],
        "active_planets": list(state.active_planets),
        "planet_q": [state.planet_q[index] for index in active_ids],
        "planet_r": [state.planet_r[index] for index in active_ids],
        "planet_terrain": [state.terrains[index] for index in active_ids],
        "planet_sectors": [state.planet_sectors[index] for index in active_ids],
        "planet_source_catalog": [
            list(item) for item in state.planet_source_catalog
        ],
    }


def map_hash(payload: dict[str, object]) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def build_fixture(sample_count: int = 256) -> dict[str, object]:
    variants: dict[str, object] = {}
    for variant, players, map_size in VARIANTS:
        samples: list[dict[str, object]] = []
        for seed in range(sample_count):
            state = GaiaState.initial(
                num_players=players,
                seed=seed,
                map_size=map_size,
            )
            payload = map_payload(state)
            samples.append(
                {
                    "seed": seed,
                    "map_hash": map_hash(payload),
                    **payload,
                }
            )
        variants[variant] = {
            "player_count": players,
            "map_size": map_size,
            "sector_count": len(samples[0]["sector_tiles"]),
            "active_planet_count": sum(samples[0]["active_planets"]),
            "seed_manifest": list(range(sample_count)),
            "samples": samples,
        }
    return {
        "schema_version": "bga-map-golden-v1",
        "rules_version": RULES_VERSION,
        "setup_seed_stream_version": SETUP_SEED_STREAM_VERSION,
        "fixture_scope": "map_only_bga_contract",
        "source": "current-gaiazero-bga-map-randomizer",
        "generator": "scripts/generate_bga_setup_fixture.py",
        "sample_count_per_variant": sample_count,
        "variants": variants,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/fixtures/bga_setup_golden.json"),
    )
    parser.add_argument("--samples", type=int, default=256)
    args = parser.parse_args()
    if args.samples < 1:
        raise ValueError("--samples must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            build_fixture(args.samples),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output} ({args.samples} samples x {len(VARIANTS)} variants)")


if __name__ == "__main__":
    main()
