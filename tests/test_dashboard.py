import json
import threading
import unittest
from pathlib import Path
from urllib.request import urlopen

from gaiazero.dashboard import create_dashboard_server
from gaiazero.game import MiniGaiaState
from gaiazero.telemetry import JsonlTelemetry, read_events


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
        JsonlTelemetry(self.metrics, run_id="http-test").emit("run_started", config={})
        server = create_dashboard_server(self.metrics, port=0, quiet=True)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base = f"http://127.0.0.1:{server.server_port}"
            with urlopen(f"{base}/api/events", timeout=5) as response:
                data = json.loads(response.read())
            with urlopen(base, timeout=5) as response:
                page = response.read().decode("utf-8")
            self.assertEqual(data["events"][0]["type"], "run_started")
            self.assertIn("GaiaZero", page)
            self.assertIn("loss-chart", page)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
