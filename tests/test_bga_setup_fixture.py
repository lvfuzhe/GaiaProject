import hashlib
import json
import unittest
from pathlib import Path

from gaiazero.contracts import RULES_VERSION, canonical_json
from gaiazero.game import GaiaState
from gaiazero.game.gaia_setup import SETUP_SEED_STREAM_VERSION


class BgaSetupGoldenFixtureTests(unittest.TestCase):
    def test_four_map_variants_are_complete_and_reproducible(self) -> None:
        fixture_path = Path(__file__).parent / "fixtures" / "bga_setup_golden.json"
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        self.assertEqual(fixture["schema_version"], "bga-map-golden-v1")
        self.assertEqual(fixture["rules_version"], RULES_VERSION)
        self.assertEqual(
            fixture["setup_seed_stream_version"], SETUP_SEED_STREAM_VERSION
        )
        expected = {
            "2p-reduced": (2, "reduced", 7, 40),
            "3p-reduced": (3, "reduced", 8, 49),
            "3p-normal": (3, "normal", 10, 61),
            "4p-normal": (4, "normal", 10, 61),
        }
        self.assertEqual(set(fixture["variants"]), set(expected))
        for name, (players, map_size, sector_count, planet_count) in expected.items():
            variant = fixture["variants"][name]
            self.assertEqual(variant["player_count"], players)
            self.assertEqual(variant["map_size"], map_size)
            self.assertEqual(variant["sector_count"], sector_count)
            self.assertEqual(variant["active_planet_count"], planet_count)
            self.assertEqual(len(variant["samples"]), 256)
            self.assertEqual(variant["seed_manifest"], list(range(256)))
            centers = variant["samples"][0]["sector_centers"]
            for sample in variant["samples"]:
                self.assertEqual(len(sample["sector_tiles"]), sector_count)
                self.assertEqual(len(sample["sector_rotations"]), sector_count)
                self.assertEqual(len(sample["planet_q"]), planet_count)
                self.assertEqual(len(sample["planet_r"]), planet_count)
                self.assertEqual(len(sample["planet_terrain"]), planet_count)
                self.assertEqual(len(sample["planet_sectors"]), planet_count)
                self.assertEqual(sample["sector_centers"], centers)
                state = GaiaState.initial(
                    num_players=players,
                    seed=sample["seed"],
                    map_size=map_size,
                )
                self.assertEqual(sample["sector_tiles"], list(state.sector_tiles))
                self.assertEqual(
                    sample["sector_rotations"], list(state.sector_rotations)
                )
                active_ids = [
                    index
                    for index, active in enumerate(state.active_planets)
                    if active
                ]
                self.assertEqual(
                    sample["active_planets"], list(state.active_planets)
                )
                self.assertEqual(
                    sample["planet_q"], [state.planet_q[index] for index in active_ids]
                )
                self.assertEqual(
                    sample["planet_r"], [state.planet_r[index] for index in active_ids]
                )
                self.assertEqual(
                    sample["planet_terrain"],
                    [state.terrains[index] for index in active_ids],
                )
                self.assertEqual(
                    sample["planet_sectors"],
                    [state.planet_sectors[index] for index in active_ids],
                )
                map_payload = {
                    key: sample[key]
                    for key in (
                        "map_size",
                        "sector_tiles",
                        "sector_rotations",
                        "sector_centers",
                        "outlined_sector_tiles",
                        "active_planets",
                        "planet_q",
                        "planet_r",
                        "planet_terrain",
                        "planet_sectors",
                        "planet_source_catalog",
                    )
                }
                expected_hash = hashlib.sha256(
                    canonical_json(map_payload).encode("utf-8")
                ).hexdigest()
                self.assertEqual(sample["map_hash"], expected_hash)


if __name__ == "__main__":
    unittest.main()
