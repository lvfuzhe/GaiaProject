from __future__ import annotations

import json
import subprocess
import unittest
from dataclasses import replace
from pathlib import Path

from gaiazero.contracts import canonical_json
from gaiazero.game import GaiaState
from gaiazero.game.gaia_state import Building, PlayerState, Terrain


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "build" / "cpp-msvc" / "gaiazero_state_probe.exe"


def _action_text(state: GaiaState) -> str:
    return ";".join(
        f"{action.action_type}({','.join(str(value) for value in action.args)})"
        for action in state.legal_action_tuples()
    )


def _encoded_action_text(state: GaiaState) -> str:
    return "|".join(
        f"{action.action_type}[{'.'.join(str(value) for value in action.args)}]"
        for action in state.legal_action_tuples()
    )


def _probe_choice(seed: int, step: int, count: int) -> int:
    value = (seed & ((1 << 64) - 1)) + step * 1442695040888963407
    value &= (1 << 64) - 1
    value ^= value >> 30
    value = (value * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    value ^= value >> 27
    value = (value * 0x94D049BB133111EB) & ((1 << 64) - 1)
    value ^= value >> 31
    return value % count if count else 0


@unittest.skipUnless(PROBE.exists(), "C++ state probe has not been built")
class CppStateParityTests(unittest.TestCase):
    def _assert_scenario(
        self,
        mode: str,
        states: list[GaiaState],
        *,
        seed: int = 17,
    ) -> None:
        completed = subprocess.run(
            [str(PROBE), "2", str(seed), mode],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        lines = [
            line for line in completed.stdout.splitlines()
            if line.startswith("scenario_step=")
        ]
        self.assertEqual(len(lines), len(states))
        for expected_step, (line, state) in enumerate(zip(lines, states, strict=True)):
            values = dict(item.split("=", 1) for item in line.split(","))
            self.assertEqual(int(values["scenario_step"]), expected_step)
            self.assertEqual(values["hash"], state.state_hash())
            self.assertEqual(values["actions"], _encoded_action_text(state))

    def test_initial_state_and_legal_actions_match_python(self) -> None:
        for players in (2, 3, 4):
            for seed in (0, 1, 17, 255, 20260828):
                with self.subTest(players=players, seed=seed):
                    state = GaiaState.initial(players, seed)
                    completed = subprocess.run(
                        [str(PROBE), str(players), str(seed)],
                        cwd=ROOT,
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    result = dict(line.split("=", 1) for line in completed.stdout.splitlines())
                    self.assertEqual(result["setup_hash"], state.setup_hash)
                    self.assertEqual(
                        json.loads(result["canonical_json"]),
                        json.loads(canonical_json(state)),
                    )
                    self.assertEqual(result["state_hash"], state.state_hash())
                    self.assertEqual(result["legal_actions"], _action_text(state))

    def test_first_action_trace_matches_python_state_by_state(self) -> None:
        """The probe and Python engine must agree after every deterministic step.

        This deliberately follows the first legal ActionTuple (including
        pending decisions such as passive charge and final-round passing), so
        it exercises transition semantics rather than only initial encoding.
        """
        for players, seed in ((2, 0), (2, 17), (3, 0), (4, 17)):
            with self.subTest(players=players, seed=seed):
                state = GaiaState.initial(players, seed)
                completed = subprocess.run(
                    [str(PROBE), str(players), str(seed), "trace"],
                    cwd=ROOT,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                trace_lines = [line for line in completed.stdout.splitlines() if line.startswith("step=")]
                for line in trace_lines:
                    values = dict(item.split("=", 1) for item in line.split(","))
                    self.assertEqual(values["hash"], state.state_hash())
                    actions = state.legal_action_tuples()
                    self.assertEqual(int(values["action_count"]), len(actions))
                    self.assertEqual(values["all_actions"], _encoded_action_text(state))
                    if actions:
                        action = actions[0]
                        expected = f"{action.action_type}({','.join(str(value) for value in action.args)})"
                        self.assertEqual(values["first"], expected)
                        state = state.apply_tuple(action)
                    else:
                        break

    def test_alternative_action_trace_matches_python_state_by_state(self) -> None:
        """Exercise non-first legal choices, including pending special rules."""
        for players, seed in ((2, 1), (3, 4), (4, 8)):
            with self.subTest(players=players, seed=seed):
                state = GaiaState.initial(players, seed)
                completed = subprocess.run(
                    [str(PROBE), str(players), str(seed), "trace-random=200"],
                    cwd=ROOT,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                trace_lines = [line for line in completed.stdout.splitlines() if line.startswith("step=")]
                for line in trace_lines:
                    values = dict(item.split("=", 1) for item in line.split(","))
                    self.assertEqual(values["hash"], state.state_hash())
                    actions = state.legal_action_tuples()
                    self.assertEqual(int(values["action_count"]), len(actions))
                    self.assertEqual(values["all_actions"], _encoded_action_text(state))
                    if not actions:
                        break
                    chosen = _probe_choice(seed, int(values["step"]), len(actions))
                    self.assertEqual(int(values["chosen"]), chosen)
                    state = state.apply_tuple(actions[chosen])
                final_lines = [line for line in completed.stdout.splitlines() if line.startswith("state_hash=")]
                self.assertTrue(final_lines)

    def test_itars_gaia_technology_scenario_matches_python(self) -> None:
        state = GaiaState.initial(2, 17)
        planet = next(
            planet for planet, active in enumerate(state.active_planets) if active
        )
        owners = list(state.owners)
        buildings = list(state.buildings)
        starting_planets = list(state.starting_planets)
        players = list(state.players)
        owners[planet] = 0
        buildings[planet] = Building.PLANETARY_INSTITUTE
        starting_planets[0] = (planet,)
        players[0] = PlayerState(
            faction=13,
            ore=5,
            bowl_one=4,
            bowl_two=4,
            gaia_power=8,
            colonized_types=1 << int(Terrain.ICE),
        )
        state = replace(
            state,
            players=tuple(players),
            owners=tuple(owners),
            buildings=tuple(int(value) for value in buildings),
            starting_planets=tuple(starting_planets),
            round_number=1,
            placement_step=len(state.placement_order),
            booster_selection_step=state.player_count,
            player_to_move=0,
            pending_itars_gaia_player=0,
        )
        states = [state]
        state = state.apply_tuple(next(
            action for action in state.legal_action_tuples()
            if action.action_type == "itars_gaia_technology"
        ))
        states.append(state)
        state = state.apply_tuple(next(
            action for action in state.legal_action_tuples()
            if action.action_type == "tech_take" and action.args == (8,)
        ))
        states.append(state)
        state = state.apply_tuple(next(
            action for action in state.legal_action_tuples()
            if action.action_type == "skip_tech_research"
        ))
        states.append(state)
        state = state.apply_tuple(next(
            action for action in state.legal_action_tuples()
            if action.action_type == "itars_gaia_finish"
        ))
        states.append(state)
        self._assert_scenario("scenario-itars-gaia", states)

    def test_gleens_qic_federation_scenario_matches_python(self) -> None:
        state = GaiaState.initial(2, 17)
        planet = next(
            planet for planet, active in enumerate(state.active_planets) if active
        )
        owners = list(state.owners)
        buildings = list(state.buildings)
        starting_planets = list(state.starting_planets)
        players = list(state.players)
        owners[planet] = 0
        buildings[planet] = Building.PLANETARY_INSTITUTE
        starting_planets[0] = (planet,)
        players[0] = PlayerState(
            faction=3,
            qic=3,
            tracks=(0, 1, 0, 0, 0, 0),
            federation_tokens=1,
            federation_keys=1,
            gleens_federation_tokens=1,
            colonized_types=1 << int(Terrain.DESERT),
        )
        state = replace(
            state,
            players=tuple(players),
            owners=tuple(owners),
            buildings=tuple(int(value) for value in buildings),
            starting_planets=tuple(starting_planets),
            round_number=1,
            placement_step=len(state.placement_order),
            booster_selection_step=state.player_count,
            player_to_move=0,
        )
        states = [state]
        state = state.apply_tuple(next(
            action for action in state.legal_action_tuples()
            if action.action_type == "qic_federation" and action.args == (6,)
        ))
        states.append(state)
        self._assert_scenario("scenario-gleens-qic", states)


if __name__ == "__main__":
    unittest.main()
