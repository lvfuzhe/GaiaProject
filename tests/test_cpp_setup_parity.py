from __future__ import annotations

import subprocess
import unittest
from pathlib import Path

from gaiazero.game import GaiaState


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "build" / "cpp-msvc" / "gaiazero_setup_probe.exe"


def _csv(values: tuple[int, ...]) -> str:
    return ",".join(str(value) for value in values)


@unittest.skipUnless(PROBE.exists(), "C++ setup probe has not been built")
class CppSetupParityTests(unittest.TestCase):
    def test_setup_matches_python_for_every_map_variant(self) -> None:
        variants = ((2, "reduced"), (3, "reduced"), (3, "normal"), (4, "normal"))
        for players, map_size in variants:
            for seed in (0, 1, 2, 17, 255, 20260828):
                with self.subTest(players=players, map_size=map_size, seed=seed):
                    state = GaiaState.initial(players, seed, map_size=map_size)
                    completed = subprocess.run(
                        [str(PROBE), str(players), str(seed), map_size],
                        cwd=ROOT,
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    result = dict(line.split("=", 1) for line in completed.stdout.splitlines())
                    self.assertEqual(result["setup_hash"], state.setup_hash)
                    self.assertEqual(result["first_player"], str(state.first_player))
                    self.assertEqual(result["factions"], _csv(tuple(p.faction for p in state.players)))
                    self.assertEqual(result["sector_tiles"], _csv(state.sector_tiles))
                    self.assertEqual(result["sector_rotations"], _csv(state.sector_rotations))
                    self.assertEqual(result["placement_order"], _csv(state.placement_order))
                    self.assertEqual(result["round_scoring"], _csv(state.round_scoring_tiles))
                    self.assertEqual(result["final_scoring"], _csv(state.final_scoring_tiles))
                    self.assertEqual(result["standard_tech"], _csv(state.standard_tech_tiles))
                    self.assertEqual(result["advanced_tech"], _csv(state.advanced_tech_tiles))
                    self.assertEqual(
                        result["terraforming_federation"],
                        str(state.terraforming_federation_tile),
                    )
                    self.assertEqual(
                        result["active"],
                        "".join("1" if value else "0" for value in state.active_planets[:-1]),
                    )
                    for key, values in (
                        ("planet_q", state.planet_q[:-1]),
                        ("planet_r", state.planet_r[:-1]),
                        ("terrains", state.terrains[:-1]),
                        ("planet_sectors", state.planet_sectors[:-1]),
                    ):
                        self.assertEqual(result[key], _csv(values))


if __name__ == "__main__":
    unittest.main()
