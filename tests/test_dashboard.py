import json
import threading
import unittest
from pathlib import Path
from urllib.request import urlopen

from gaiazero.dashboard import create_dashboard_server
from gaiazero.game import MiniGaiaState
from gaiazero.telemetry import JsonlTelemetry, build_history_index, read_events, read_game_trace


class DashboardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.metrics = Path(__file__).parent / ".artifacts" / "dashboard.jsonl"
        self.metrics.unlink(missing_ok=True)

    def tearDown(self) -> None:
        self.metrics.unlink(missing_ok=True)

    def test_telemetry_round_trip_with_board_snapshot(self) -> None:
        telemetry = JsonlTelemetry(self.metrics, run_id="test-run")
        state = MiniGaiaState.initial(2, seed=3)
        first = telemetry.emit("run_started", state=state.snapshot(), config={"iterations": 2})
        second = telemetry.emit("training_update", loss=1.25, update=1)

        events = read_events(self.metrics)
        self.assertEqual([event["sequence"] for event in events], [first["sequence"], second["sequence"]])
        self.assertEqual(events[0]["run_id"], "test-run")
        self.assertEqual(len(events[0]["payload"]["state"]["planets"]), 19)
        self.assertEqual(read_events(self.metrics, after=first["sequence"]), [second])

    def test_http_api_and_static_dashboard(self) -> None:
        telemetry = JsonlTelemetry(self.metrics, run_id="http-test")
        state = MiniGaiaState.initial(2)
        telemetry.emit("run_started", config={}, state=state.snapshot())
        telemetry.emit(
            "self_play_started",
            iteration=1,
            game_in_iteration=1,
            state=state.snapshot(),
        )
        server = create_dashboard_server(self.metrics, port=0, quiet=True)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base = f"http://127.0.0.1:{server.server_port}"
            with urlopen(f"{base}/api/events", timeout=5) as response:
                data = json.loads(response.read())
            with urlopen(base, timeout=5) as response:
                page = response.read().decode("utf-8")
            with urlopen(f"{base}/assets/sectors/sector-01-solid.gif", timeout=5) as response:
                sector_image = response.read()
                sector_content_type = response.headers.get_content_type()
            with urlopen(f"{base}/assets/sectors/sector-05-outlined.gif", timeout=5) as response:
                outlined_sector_image = response.read()
            with urlopen(f"{base}/api/history", timeout=5) as response:
                history = json.loads(response.read())
            with urlopen(
                f"{base}/api/game?run_id=http-test&iteration=1&game=1",
                timeout=5,
            ) as response:
                game = json.loads(response.read())
            self.assertEqual(data["events"][0]["type"], "run_started")
            self.assertEqual(history["runs"][0]["iterations"][0]["iteration"], 1)
            self.assertEqual(game["steps"][0]["move"], 0)
            self.assertIn("GaiaZero", page)
            self.assertIn("loss-chart", page)
            self.assertEqual(sector_content_type, "image/gif")
            self.assertTrue(sector_image.startswith(b"GIF"))
            self.assertTrue(outlined_sector_image.startswith(b"GIF"))
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_history_index_and_complete_game_trace(self) -> None:
        telemetry = JsonlTelemetry(self.metrics, run_id="history-test")
        state = MiniGaiaState.initial(2)
        telemetry.emit("run_started", config={"iterations": 1}, state=state.snapshot())
        telemetry.emit(
            "self_play_started",
            iteration=1,
            game_in_iteration=1,
            state=state.snapshot(),
        )
        actions = []
        for move in range(1, 3):
            action = state.legal_actions()[0]
            next_state = state.apply(action)
            actions.append(action)
            telemetry.emit(
                "self_play_step",
                iteration=1,
                game_in_iteration=1,
                move=move,
                player=state.current_player,
                action=action,
                action_label=state.describe_action(action),
                legal_actions=len(state.legal_actions()),
                search_sampled=move == 1,
                state=next_state.snapshot(),
            )
            state = next_state
        telemetry.emit(
            "self_play_completed",
            iteration=1,
            game_in_iteration=1,
            moves=2,
            positions=2,
            scores=state.final_scores(),
            returns=[0.0, 0.0],
            duration_seconds=0.5,
            state=state.snapshot(),
        )
        telemetry.emit("iteration_completed", iteration=1, loss=1.0)

        index = build_history_index(self.metrics)
        game = index["runs"][0]["iterations"][0]["games"][0]
        self.assertTrue(game["trace_complete"])
        self.assertEqual(game["captured_moves"], 2)

        trace = read_game_trace(
            self.metrics,
            run_id="history-test",
            iteration=1,
            game=1,
        )
        self.assertIsNotNone(trace)
        self.assertTrue(trace["trace_complete"])
        self.assertEqual([step["move"] for step in trace["steps"]], [0, 1, 2])
        self.assertEqual([step["action"] for step in trace["steps"][1:]], actions)


if __name__ == "__main__":
    unittest.main()
