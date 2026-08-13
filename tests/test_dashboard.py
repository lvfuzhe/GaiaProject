import json
import threading
import time
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from gaiazero.dashboard import create_dashboard_server
from gaiazero.game import MiniGaiaState
from gaiazero.telemetry import JsonlTelemetry, build_history_index, read_events, read_game_trace


class DashboardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.metrics = Path(__file__).parent / ".artifacts" / "dashboard.jsonl"
        self.metrics.unlink(missing_ok=True)

    def tearDown(self) -> None:
        self.metrics.unlink(missing_ok=True)

    @staticmethod
    def post_json(url: str, payload: dict[str, object]) -> tuple[int, dict[str, object]]:
        request = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read())

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
            with urlopen(f"{base}/app.js", timeout=5) as response:
                app_script = response.read().decode("utf-8")
            with urlopen(f"{base}/setup/random", timeout=5) as response:
                random_setup_page = response.read().decode("utf-8")
            with urlopen(f"{base}/setup/manual", timeout=5) as response:
                manual_setup_page = response.read().decode("utf-8")
            with urlopen(f"{base}/play", timeout=5) as response:
                play_page = response.read().decode("utf-8")
            with urlopen(f"{base}/assets/sectors/sector-01-solid.gif", timeout=5) as response:
                sector_image = response.read()
                sector_content_type = response.headers.get_content_type()
            with urlopen(f"{base}/assets/sectors/sector-05-outlined.gif", timeout=5) as response:
                outlined_sector_image = response.read()
            with urlopen(f"{base}/assets/boards/research-board.png", timeout=5) as response:
                research_board_image = response.read()
                research_board_content_type = response.headers.get_content_type()
            faction_assets = []
            player_board_assets = []
            for number in range(1, 15):
                with urlopen(f"{base}/assets/factions/faction-{number:02d}.jpg", timeout=5) as response:
                    faction_assets.append((response.headers.get_content_type(), response.read()))
                with urlopen(f"{base}/assets/factions/player-board-{number:02d}.jpg", timeout=5) as response:
                    player_board_assets.append((response.headers.get_content_type(), response.read()))
            tile_assets = {}
            for path in (
                "tech-standard-01.jpg",
                "tech-advanced-01.jpg",
                "round-scoring-01.gif",
                "final-scoring-01.jpg",
                "booster-01.jpg",
            ):
                with urlopen(f"{base}/assets/tiles/{path}", timeout=5) as response:
                    tile_assets[path] = (response.headers.get_content_type(), response.read())
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
            self.assertIn("setup-faction-catalog", page)
            self.assertIn("setup-research-tech", page)
            self.assertIn("setup-editor-form", page)
            self.assertIn("setup-editor-run", page)
            self.assertIn("setup-random-editor", page)
            self.assertIn("setup-editor-sectors", page)
            self.assertIn("setup-planet-editor-canvas", page)
            self.assertIn("setup-planet-editor-add", page)
            self.assertIn("setup-planet-editor-delete", page)
            self.assertIn("setup-planet-editor-reset", page)
            self.assertIn("setup-editor-standard-tech", page)
            self.assertIn("setup-editor-boosters", page)
            self.assertIn("setup-editor-map-mode", page)
            self.assertIn("play-config-form", page)
            self.assertIn("play-board-canvas", page)
            self.assertIn("play-live-roles", page)
            self.assertIn("play-auto-ai", page)
            self.assertIn("play-setup-workspace", page)
            self.assertIn("play-match-workspace", page)
            self.assertIn('data-play-workspace="setup"', page)
            self.assertNotIn('data-view="setup"', page)
            self.assertIn("function drawStarfield", app_script)
            self.assertIn("function drawPlanetArtwork", app_script)
            self.assertIn("function drawStarMapBoard", app_script)
            self.assertIn(
                'drawStarMapBoard(byId("history-board-canvas"), snapshot, true)',
                app_script,
            )
            self.assertIn("function handlePlanetEditorClick", app_script)
            self.assertIn("function resetPlanetLayout", app_script)
            self.assertIn("function addPlanetAt", app_script)
            self.assertIn("function deleteSelectedPlanet", app_script)
            self.assertIn("function snapshotRoundLabel", app_script)
            self.assertIn("function startInteractiveGame", app_script)
            self.assertIn("function prepareInteractiveMatch", app_script)
            self.assertIn("function submitHumanAction", app_script)
            self.assertIn("function runInteractiveAiTurn", app_script)
            self.assertIn("function updateLivePlayRole", app_script)
            self.assertIn("function planetAtPlayEvent", app_script)
            self.assertNotIn("function runManualSimulation", app_script)
            self.assertIn("开局基地按蛇形顺位放置", app_script)
            self.assertIn("planetArtwork: true", app_script)
            self.assertEqual(random_setup_page, page)
            self.assertEqual(manual_setup_page, page)
            self.assertEqual(play_page, page)
            self.assertIn("player-board-grid", page)
            self.assertIn("history-player-board-grid", page)
            self.assertIn("history-star-map-frame", page)
            self.assertEqual(sector_content_type, "image/gif")
            self.assertTrue(sector_image.startswith(b"GIF"))
            self.assertTrue(outlined_sector_image.startswith(b"GIF"))
            self.assertEqual(research_board_content_type, "image/png")
            self.assertTrue(research_board_image.startswith(b"\x89PNG\r\n\x1a\n"))
            self.assertTrue(all(
                content_type == "image/jpeg" and content.startswith(b"\xff\xd8")
                for content_type, content in faction_assets
            ))
            self.assertTrue(all(
                content_type == "image/jpeg"
                and content.startswith(b"\xff\xd8")
                and len(content) > 100_000
                for content_type, content in player_board_assets
            ))
            self.assertEqual(len({content for _content_type, content in player_board_assets}), 14)
            self.assertIn("function factionPlayerBoardAsset", app_script)
            self.assertIn("完整个人主板组合图", app_script)
            for name, (content_type, content) in tile_assets.items():
                expected_type = "image/gif" if name.endswith(".gif") else "image/jpeg"
                self.assertEqual(content_type, expected_type)
                self.assertTrue(content.startswith(b"GIF" if expected_type == "image/gif" else b"\xff\xd8"))
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_manual_setup_preview_and_validation(self) -> None:
        server = create_dashboard_server(self.metrics, port=0, quiet=True)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base = f"http://127.0.0.1:{server.server_port}"
            payload = {
                "players": 2,
                "seed": 23,
                "first_player": 1,
                "factions": [0, 2],
                "simulations": 1,
            }
            status, preview = self.post_json(f"{base}/api/setup/preview", payload)
            self.assertEqual(status, 200)
            for key, value in payload.items():
                self.assertEqual(preview["config"][key], value)
            random_setup = preview["config"]["random_setup"]
            self.assertEqual(random_setup["map_mode"], "bga-random")
            self.assertEqual(len(random_setup["sector_tiles"]), 7)
            self.assertEqual(len(random_setup["sector_rotations"]), 7)
            self.assertEqual(len(random_setup["booster_tiles"]), 5)
            self.assertEqual(len(random_setup["round_scoring_tiles"]), 6)
            self.assertEqual(len(random_setup["final_scoring_tiles"]), 2)
            self.assertEqual(len(random_setup["standard_tech_tiles"]), 9)
            self.assertEqual(len(random_setup["advanced_tech_tiles"]), 6)
            self.assertEqual(preview["state"]["first_player"], 1)
            self.assertEqual(len(preview["state"]["planets"]), 40)
            self.assertEqual(
                [player["faction_id"] for player in preview["state"]["players"]],
                [0, 2],
            )
            self.assertFalse(preview["state"]["setup"]["player_choices_resolved"])
            self.assertTrue(all(
                planet["owner"] == -1 and planet["building"] == "empty"
                for planet in preview["state"]["planets"]
            ))
            self.assertTrue(all(
                "starting_planets" not in faction
                for faction in preview["state"]["setup"]["factions"]
            ))
            self.assertTrue(all(
                booster["owner"] == -1
                for booster in preview["state"]["setup"]["boosters"]
            ))
            self.assertTrue(all(
                player["booster"] is None
                and all(item["built"] == 0 for item in player["structures"].values())
                for player in preview["state"]["players"]
            ))

            customized = {
                **payload,
                "random_setup": {
                    **random_setup,
                    "map_mode": "manual",
                    "planet_positions": [
                        {
                            "id": planet["id"],
                            "q": -planet["r"],
                            "r": planet["q"] + planet["r"],
                        }
                        for planet in preview["state"]["planets"]
                    ],
                    "standard_tech_tiles": list(reversed(random_setup["standard_tech_tiles"])),
                    "advanced_tech_tiles": list(reversed(random_setup["advanced_tech_tiles"])),
                    "terraforming_federation_tile": (
                        random_setup["terraforming_federation_tile"] + 1
                    ) % 6,
                },
            }
            _, custom_preview = self.post_json(f"{base}/api/setup/preview", customized)
            self.assertEqual(
                custom_preview["config"]["random_setup"],
                customized["random_setup"],
            )
            custom_setup = custom_preview["state"]["setup"]
            self.assertEqual(
                [tile["id"] for tile in custom_setup["standard_tech"]],
                customized["random_setup"]["standard_tech_tiles"],
            )
            self.assertEqual(
                [tile["id"] for tile in custom_setup["advanced_tech"]],
                customized["random_setup"]["advanced_tech_tiles"],
            )
            self.assertEqual(
                custom_setup["terraforming_federation"]["id"],
                customized["random_setup"]["terraforming_federation_tile"],
            )
            self.assertEqual(custom_setup["map"]["method"], "manual")
            expected_positions = {
                position["id"]: (position["q"], position["r"])
                for position in customized["random_setup"]["planet_positions"]
            }
            self.assertEqual(
                {
                    planet["id"]: (planet["q"], planet["r"])
                    for planet in custom_preview["state"]["planets"]
                },
                expected_positions,
            )
            self.assertTrue(all(
                (planet["source_q"], planet["source_r"])
                != (planet["q"], planet["r"])
                for planet in custom_preview["state"]["planets"]
                if (planet["q"], planet["r"]) != (0, 0)
            ))

            removable = next(
                planet for planet in custom_preview["state"]["planets"]
                if planet["terrain"] == 8
            )
            reduced_layout = [
                {
                    "id": planet["id"],
                    "q": planet["q"],
                    "r": planet["r"],
                    "source_id": planet["source_id"],
                }
                for planet in custom_preview["state"]["planets"]
                if planet["id"] != removable["id"]
            ]
            layout_setup = {
                key: value
                for key, value in customized["random_setup"].items()
                if key != "planet_positions"
            }
            layout_setup["planet_layout"] = reduced_layout
            _, reduced_preview = self.post_json(f"{base}/api/setup/preview", {
                **payload,
                "random_setup": layout_setup,
            })
            self.assertEqual(len(reduced_preview["state"]["planets"]), 39)
            self.assertNotIn(
                removable["id"],
                {planet["id"] for planet in reduced_preview["state"]["planets"]},
            )
            self.assertEqual(
                reduced_preview["config"]["random_setup"]["planet_layout"],
                reduced_layout,
            )

            missing_manual_map = {
                **payload,
                "random_setup": {"map_mode": "manual"},
            }
            with self.assertRaises(HTTPError) as raised:
                self.post_json(f"{base}/api/setup/preview", missing_manual_map)
            self.assertEqual(raised.exception.code, 400)

            invalid = {**payload, "factions": [0, 1]}
            with self.assertRaises(HTTPError) as raised:
                self.post_json(f"{base}/api/setup/preview", invalid)
            self.assertEqual(raised.exception.code, 400)
            error = json.loads(raised.exception.read())
            self.assertIn("different double-sided boards", error["error"])

            with urlopen(f"{base}/api/simulation", timeout=5) as response:
                simulation = json.loads(response.read())
            self.assertEqual(simulation["status"], "idle")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_single_simulation_writes_complete_replay(self) -> None:
        server = create_dashboard_server(self.metrics, port=0, quiet=True)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base = f"http://127.0.0.1:{server.server_port}"
            status, started = self.post_json(
                f"{base}/api/simulation",
                {
                    "players": 2,
                    "seed": 3,
                    "first_player": 1,
                    "factions": [0, 2],
                    "simulations": 1,
                },
            )
            self.assertEqual(status, 202)
            self.assertEqual(started["status"], "running")
            run_id = started["run_id"]

            deadline = time.monotonic() + 30
            simulation = started
            while simulation["status"] == "running" and time.monotonic() < deadline:
                time.sleep(0.1)
                with urlopen(f"{base}/api/simulation", timeout=5) as response:
                    simulation = json.loads(response.read())

            self.assertEqual(simulation["status"], "complete", simulation)
            self.assertGreater(simulation["move"], 0)
            self.assertEqual(len(simulation["scores"]), 2)

            trace = read_game_trace(self.metrics, run_id=run_id, iteration=1, game=1)
            self.assertIsNotNone(trace)
            self.assertTrue(trace["trace_complete"])
            self.assertEqual(trace["captured_moves"], simulation["move"])
            self.assertEqual(len(trace["steps"]), simulation["move"] + 1)
            self.assertEqual(trace["steps"][0]["state"]["phase"], "starting_placement")
            self.assertEqual(trace["steps"][0]["state"]["round"], 0)
            self.assertTrue(all(
                planet["owner"] == -1
                for planet in trace["steps"][0]["state"]["planets"]
            ))
            self.assertEqual(trace["steps"][1]["player"], 1)
            self.assertTrue(
                trace["steps"][1]["action_label"].startswith("place starting")
            )
            self.assertEqual(trace["steps"][1]["state"]["placement"]["step"], 1)
            self.assertTrue(trace["steps"][-1]["state"]["terminal"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_interactive_game_supports_human_ai_and_role_switching(self) -> None:
        server = create_dashboard_server(self.metrics, port=0, quiet=True)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base = f"http://127.0.0.1:{server.server_port}"
            setup_payload = {
                "players": 2,
                "seed": 13,
                "first_player": 0,
                "factions": [0, 2],
                "simulations": 1,
            }
            _, preview = self.post_json(
                f"{base}/api/setup/preview",
                setup_payload,
            )
            random_setup = preview["config"]["random_setup"]
            random_setup["round_scoring_tiles"] = list(reversed(
                random_setup["round_scoring_tiles"]
            ))
            status, game = self.post_json(
                f"{base}/api/play/start",
                {
                    **setup_payload,
                    "random_setup": random_setup,
                    "roles": ["human", "ai"],
                },
            )

            self.assertEqual(status, 201)
            self.assertEqual(game["status"], "active")
            self.assertEqual(game["roles"], ["human", "ai"])
            self.assertEqual(game["current_role"], "human")
            self.assertEqual(game["config"]["random_setup"], random_setup)
            self.assertEqual(
                [tile["id"] for tile in game["state"]["setup"]["round_scoring"]],
                random_setup["round_scoring_tiles"],
            )
            self.assertEqual(game["state"]["phase"], "starting_placement")
            self.assertTrue(game["legal_actions"])
            self.assertTrue(all(
                action["kind"] == "starting_placement"
                and isinstance(action["target"], int)
                for action in game["legal_actions"]
            ))

            human_action = game["legal_actions"][0]["id"]
            _, game = self.post_json(
                f"{base}/api/play/action",
                {"action": human_action},
            )
            self.assertEqual(game["move"], 1)
            self.assertEqual(game["history"][0]["role"], "human")
            self.assertEqual(game["current_role"], "ai")

            _, game = self.post_json(
                f"{base}/api/play/roles",
                {"roles": ["human", "human"]},
            )
            self.assertEqual(game["current_role"], "human")
            with self.assertRaises(HTTPError) as raised:
                self.post_json(f"{base}/api/play/ai", {})
            self.assertEqual(raised.exception.code, 409)

            _, game = self.post_json(
                f"{base}/api/play/roles",
                {"roles": ["human", "ai"]},
            )
            _, game = self.post_json(f"{base}/api/play/ai", {})
            self.assertEqual(game["move"], 2)
            self.assertEqual(game["history"][-1]["role"], "ai")
            self.assertEqual(game["last_action"]["player"], 1)
            self.assertTrue(game["last_search"]["candidates"])

            with urlopen(f"{base}/api/play", timeout=5) as response:
                restored = json.loads(response.read())
            self.assertEqual(restored["session_id"], game["session_id"])
            self.assertEqual(restored["move"], 2)
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
