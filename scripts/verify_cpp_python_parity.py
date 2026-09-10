"""Run the complete Python/C++ standard-v22 parity acceptance.

The setup phase checks every seed stored in the BGA golden fixture. The trace
phase follows the same deterministic random choice function as the C++ probe
and compares every pre-action state, legal ActionTuple list, transition and
terminal canonical state.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from gaiazero.contracts import canonical_json
from gaiazero.game import GaiaState


ROOT = Path(__file__).resolve().parents[1]
SETUP_PROBE = ROOT / "build" / "cpp-msvc" / "gaiazero_setup_probe.exe"
STATE_PROBE = ROOT / "build" / "cpp-msvc" / "gaiazero_state_probe.exe"
FIXTURE = ROOT / "tests" / "fixtures" / "bga_setup_golden.json"


def _csv(values: tuple[int, ...] | list[int]) -> str:
    return ",".join(str(value) for value in values)


def _encoded_actions(state: GaiaState) -> str:
    return "|".join(
        f"{action.action_type}[{'.'.join(str(value) for value in action.args)}]"
        for action in state.legal_action_tuples()
    )


def _probe_choice(seed: int, step: int, count: int) -> int:
    mask = (1 << 64) - 1
    value = (seed & mask) + step * 1442695040888963407
    value &= mask
    value ^= value >> 30
    value = (value * 0xBF58476D1CE4E5B9) & mask
    value ^= value >> 27
    value = (value * 0x94D049BB133111EB) & mask
    value ^= value >> 31
    return value % count if count else 0


def _lines_as_dict(output: str) -> dict[str, str]:
    return dict(line.split("=", 1) for line in output.splitlines() if "=" in line)


def verify_setup() -> int:
    if not SETUP_PROBE.exists():
        raise RuntimeError(f"missing C++ setup probe: {SETUP_PROBE}")
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    checked = 0
    for key, variant in fixture["variants"].items():
        players = int(variant["player_count"])
        map_size = str(variant["map_size"])
        for sample in variant["samples"]:
            seed = int(sample["seed"])
            state = GaiaState.initial(players, seed, map_size=map_size)
            completed = subprocess.run(
                [str(SETUP_PROBE), str(players), str(seed), map_size],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            actual = _lines_as_dict(completed.stdout)
            expected = {
                "setup_hash": state.setup_hash,
                "first_player": str(state.first_player),
                "factions": _csv(tuple(player.faction for player in state.players)),
                "sector_tiles": _csv(state.sector_tiles),
                "sector_rotations": _csv(state.sector_rotations),
                "placement_order": _csv(state.placement_order),
                "round_scoring": _csv(state.round_scoring_tiles),
                "final_scoring": _csv(state.final_scoring_tiles),
                "standard_tech": _csv(state.standard_tech_tiles),
                "advanced_tech": _csv(state.advanced_tech_tiles),
                "terraforming_federation": str(state.terraforming_federation_tile),
                "active": "".join("1" if value else "0" for value in state.active_planets[:-1]),
                "planet_q": _csv(state.planet_q[:-1]),
                "planet_r": _csv(state.planet_r[:-1]),
                "terrains": _csv(state.terrains[:-1]),
                "planet_sectors": _csv(state.planet_sectors[:-1]),
            }
            for name, value in expected.items():
                if actual.get(name) != value:
                    raise AssertionError(
                        f"setup mismatch {key} seed={seed} field={name}: "
                        f"C++={actual.get(name)!r} Python={value!r}"
                    )
            checked += 1
    return checked


def verify_trace(players: int, seed: int, limit: int) -> int:
    if not STATE_PROBE.exists():
        raise RuntimeError(f"missing C++ state probe: {STATE_PROBE}")
    completed = subprocess.run(
        [str(STATE_PROBE), str(players), str(seed), f"trace-random={limit}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    state = GaiaState.initial(players, seed)
    steps = 0
    for line in completed.stdout.splitlines():
        if not line.startswith("step="):
            continue
        fields = line.split(",", 4)
        if len(fields) != 5:
            raise AssertionError(f"malformed trace line: {line}")
        step = int(fields[0].split("=", 1)[1])
        state_hash = fields[1].split("=", 1)[1]
        action_count = int(fields[2].split("=", 1)[1])
        chosen = int(fields[3].split("=", 1)[1])
        all_actions = fields[4].split("=", 1)[1]
        if step != steps:
            raise AssertionError(f"non-contiguous trace step {step}, expected {steps}")
        actions = state.legal_action_tuples()
        if state_hash != state.state_hash():
            raise AssertionError(f"hash mismatch at {players}p seed={seed} step={step}")
        if action_count != len(actions) or all_actions != _encoded_actions(state):
            raise AssertionError(f"legal actions mismatch at {players}p seed={seed} step={step}")
        if not actions:
            break
        expected_choice = _probe_choice(seed, step, len(actions))
        if chosen != expected_choice:
            raise AssertionError(f"choice mismatch at {players}p seed={seed} step={step}")
        state = state.apply_tuple(actions[chosen])
        steps += 1
    final = _lines_as_dict(completed.stdout)
    if final.get("state_hash") != state.state_hash():
        raise AssertionError(f"terminal hash mismatch at {players}p seed={seed}")
    if json.loads(final["canonical_json"]) != json.loads(canonical_json(state)):
        raise AssertionError(f"terminal canonical JSON mismatch at {players}p seed={seed}")
    return steps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-limit", type=int, default=2000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    setup_count = verify_setup()
    traces = []
    for players, seed in ((2, 0), (2, 17), (3, 0), (3, 4), (4, 17), (4, 8)):
        traces.append({"players": players, "seed": seed, "steps": verify_trace(players, seed, args.trace_limit)})
    payload = {
        "schema_version": "cpp-python-parity-acceptance-v1",
        "rules_version": "standard-v22",
        "setup_fixtures": setup_count,
        "trace_limit": args.trace_limit,
        "traces": traces,
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
