import hashlib
import json
import threading
import time
import unittest
from dataclasses import replace
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from gaiazero.dashboard import (
    _interactive_action_record,
    _interactive_action_snapshot,
    _interactive_board_changes,
    create_dashboard_server,
)
from gaiazero.game import GaiaState, MiniGaiaState
from gaiazero.game.gaia_state import (
    BAL_TAKS_GAIAFORMER_QIC_ACTION,
    BRAINSTONE_ACTION,
    Building,
    IVITS_SPACE_STATION_OFFSET,
    QIC_ACADEMY_ACTION,
    PowerAction,
    TERRANS_GAIA_FINISH_ACTION,
    TERRANS_GAIA_KNOWLEDGE_ACTION,
    TERRANS_GAIA_ORE_ACTION,
    TERRANS_GAIA_QIC_ACTION,
    Terrain,
    Track,
)
from gaiazero.telemetry import JsonlTelemetry, build_history_index, read_events, read_game_trace


PLAYER_BOARD_SHA256 = (
    "B8C804C4CD83E4CA52183B771EB221D9761592169E62B6FF15D906522F1E1CE6",
    "EFFE0E9DF5A5611CD325381D2332B10E6C719D4E0A91D0DBF70050EDD83C7691",
    "9615BC9FD6CDD9D882CFFAFF969F42807B358E9027272F008F011FB676FB7D90",
    "5925510A5C0D64EF1FA070DFF4B01F7D75789971C3C41ACDB2A57FB2A5842C52",
    "C2A729135041A8D9E6D57A44995917856299B8333279AC19EA27F7B6FFEAAF43",
    "FE9FFE039C2DFB1A11488A5E42451560894D90391879FF2119AA6B76DD096869",
    "AA267C6020E1CF80C3ACC1F451EE30FB4DC600B8AAF3D79013C37603CB0F6676",
    "141EBCEFE4A26D9FA0F96DE7A1080B1E994145FCC04EA435B661D860784A0385",
    "AC61AA9C4F7A5E7076A2ABD02811A5E13F9E675CCE704C9F79E8C7BE921A8B52",
    "FC106493FADEFC60F19B6583DDA59BB6F6017E6F7D27AE18D86D5695B6D4197D",
    "4D96FF535EE50B529ABF662AB847F3F9BB6938CFDC8AEBFFFB4FD855757E5C36",
    "ED328DFD860A9D0428100D299E8A9BB66FD88E789A0D6D883216CFF457C50CC1",
    "A8AD6A1F676C5712F2650979260A7333507EC4367AC8E2C070CE6B9AC42F58A1",
    "38A05D18A67CDBD9FD11E5BA8F13398AF0B4DA294449359B5760DC7D782B33CF",
)
RESEARCH_BOARD_SHA256 = "6A9CB95AFD5410927303E56F671206821188FD309FC55E2B268116A67DE44418"


class DashboardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.metrics = Path(__file__).parent / ".artifacts" / "dashboard.jsonl"
        self.metrics.unlink(missing_ok=True)

    def tearDown(self) -> None:
        self.metrics.unlink(missing_ok=True)

    def test_lantids_coexisting_mine_is_visible_in_board_history(self) -> None:
        before = GaiaState.initial(2, faction_indices=(1, 2), first_player=0)
        planet = next(
            index for index, active in enumerate(before.active_planets) if active
        )
        owners = list(before.coexisting_mine_owner)
        federated = list(before.coexisting_mine_federated)
        owners[planet] = 0
        federated[planet] = True
        after = replace(
            before,
            coexisting_mine_owner=tuple(owners),
            coexisting_mine_federated=tuple(federated),
        )

        changes = _interactive_board_changes(before, after)

        self.assertIn(
            {
                "kind": "coexisting_mine",
                "planet": planet,
                "owner_before": -1,
                "owner_after": 0,
            },
            changes,
        )
        self.assertIn(
            {
                "kind": "coexisting_federated",
                "planet": planet,
                "owner": 0,
                "after": True,
            },
            changes,
        )

    def test_gleens_pi_history_identifies_special_federation_tile(self) -> None:
        before = GaiaState.initial(
            2,
            faction_indices=(3, 0),
            first_player=0,
            seed=43,
        )
        planet = next(
            index for index, active in enumerate(before.active_planets) if active
        )
        owners = [-1] * len(before.owners)
        buildings = [Building.EMPTY] * len(before.buildings)
        owners[planet] = 0
        buildings[planet] = Building.TRADING_STATION
        players = list(before.players)
        players[0] = replace(
            players[0],
            credits=10,
            ore=10,
            knowledge=0,
            vp=10,
        )
        before = replace(
            before,
            round_number=1,
            placement_step=len(before.placement_order),
            booster_selection_step=len(before.booster_selection_order),
            player_to_move=0,
            owners=tuple(owners),
            buildings=tuple(int(building) for building in buildings),
            players=tuple(players),
            round_scoring_tiles=(3, 0, 1, 2, 4, 5),
        )
        action = before.upgrade_pi_action(planet)
        after = before.apply(action)

        record = _interactive_action_record(
            before,
            after,
            action,
            move=1,
            player=0,
            role="human",
        )

        self.assertIn("GLE-FED", [item["code"] for item in record["components"]])
        self.assertIn("RND-04", [item["code"] for item in record["components"]])
        changes = record["effects"][0]["changes"]
        self.assertIn(
            {
                "kind": "counter",
                "counter": "gleens_federation_tokens",
                "before": 0,
                "after": 1,
            },
            changes,
        )
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

    def test_bal_taks_credit_academy_is_labeled_in_action_history(self) -> None:
        state = GaiaState.initial(
            2,
            faction_indices=(9, 0),
            first_player=0,
        )
        players = list(state.players)
        players[0] = replace(
            players[0],
            credits=0,
            qic=0,
            qic_academies=1,
        )
        state = replace(
            state,
            round_number=1,
            player_to_move=0,
            players=tuple(players),
        )

        after = state.apply(QIC_ACADEMY_ACTION)
        record = _interactive_action_record(
            state,
            after,
            QIC_ACADEMY_ACTION,
            move=1,
            player=0,
            role="human",
        )

        self.assertEqual(record["kind"], "credits_academy_action")
        self.assertIn("4 credits", record["label"])
        self.assertEqual(record["components"][0]["label"], "Gain 4 credits")
        self.assertEqual(record["effects"][0]["gains"], [{"resource": "credits", "amount": 4}])
        upgrade = _interactive_action_snapshot(
            state,
            state.upgrade_qic_academy_action(0),
        )
        self.assertEqual(upgrade["kind"], "upgrade_credits_academy")
        self.assertIn("credit academy", upgrade["label"])

    def test_bal_taks_gaiaformer_conversion_is_detailed_in_history(self) -> None:
        state = GaiaState.initial(
            2,
            faction_indices=(9, 0),
            first_player=0,
        )
        state = replace(
            state,
            round_number=1,
            placement_step=len(state.placement_order),
            booster_selection_step=len(state.booster_selection_order),
            player_to_move=0,
        )

        after = state.apply(BAL_TAKS_GAIAFORMER_QIC_ACTION)
        record = _interactive_action_record(
            state,
            after,
            BAL_TAKS_GAIAFORMER_QIC_ACTION,
            move=1,
            player=0,
            role="human",
        )

        self.assertEqual(record["kind"], "bal_taks_gaiaformer_qic")
        self.assertIn("BAL-GF-QIC", [item["code"] for item in record["components"]])
        self.assertEqual(
            record["effects"][0]["costs"],
            [{"resource": "gaiaformers", "amount": 1}],
        )
        self.assertEqual(
            record["effects"][0]["gains"],
            [{"resource": "qic", "amount": 1}],
        )
        changes = record["effects"][0]["changes"]
        self.assertIn(
            {
                "kind": "counter",
                "counter": "gaiaformers_in_gaia",
                "before": 0,
                "after": 1,
            },
            changes,
        )
        self.assertEqual(after.player_to_move, 0)

    def test_hadsch_hallas_credit_conversion_is_detailed_in_history(self) -> None:
        state = GaiaState.initial(2, faction_indices=(6, 0), first_player=0)
        planet = next(index for index, active in enumerate(state.active_planets) if active)
        owners = list(state.owners)
        buildings = list(state.buildings)
        owners[planet] = 0
        buildings[planet] = Building.PLANETARY_INSTITUTE
        players = list(state.players)
        players[0] = replace(players[0], credits=8, knowledge=0)
        state = replace(
            state,
            round_number=1,
            placement_step=len(state.placement_order),
            booster_selection_step=len(state.booster_selection_order),
            player_to_move=0,
            owners=tuple(owners),
            buildings=tuple(int(building) for building in buildings),
            players=tuple(players),
        )

        after = state.apply(TERRANS_GAIA_KNOWLEDGE_ACTION)
        record = _interactive_action_record(
            state,
            after,
            TERRANS_GAIA_KNOWLEDGE_ACTION,
            move=1,
            player=0,
            role="human",
        )

        self.assertEqual(record["kind"], "hadsch_credit_knowledge")
        self.assertIn("HAD-PI-CREDIT", [item["code"] for item in record["components"]])
        self.assertEqual(record["effects"][0]["costs"], [{"resource": "credits", "amount": 4}])
        self.assertEqual(record["effects"][0]["gains"], [{"resource": "knowledge", "amount": 1}])
        self.assertEqual(after.player_to_move, 0)
        self.assertIn(TERRANS_GAIA_ORE_ACTION, after.legal_actions())
        self.assertIn(TERRANS_GAIA_QIC_ACTION, after.legal_actions())

    def test_ivits_space_station_is_identified_in_actions_and_history(self) -> None:
        state = GaiaState.initial(
            2,
            seed=23,
            faction_indices=(7, 0),
            first_player=0,
        )
        while state.is_starting_placement or state.is_booster_selection:
            state = state.apply(state.legal_actions()[0])
        state = replace(state, player_to_move=0)
        action = next(
            action
            for action in state.legal_actions()
            if action >= IVITS_SPACE_STATION_OFFSET
        )

        summary = _interactive_action_snapshot(state, action)
        self.assertEqual(summary["kind"], "ivits_space_station")
        self.assertIsInstance(summary["space_station_slot"], int)
        self.assertIsInstance(summary["space_q"], int)
        self.assertIsInstance(summary["space_r"], int)

        after = state.apply(action)
        record = _interactive_action_record(
            state,
            after,
            action,
            move=1,
            player=0,
            role="human",
        )
        component = next(
            component
            for component in record["components"]
            if component["kind"] == "space_station"
        )
        self.assertEqual(record["kind"], "ivits_space_station")
        self.assertTrue(component["code"].startswith("IVI-SS-"))
        self.assertEqual(len(after.snapshot()["space_stations"]), 1)

    def test_geodens_pi_knowledge_reward_is_identified_in_history(self) -> None:
        state = GaiaState.initial(
            2,
            seed=37,
            faction_indices=(8, 0),
            first_player=0,
        )
        pi_planet = next(
            planet
            for planet, active in enumerate(state.active_planets)
            if active and Terrain(state.terrains[planet]) == Terrain.VOLCANIC
        )
        target = next(
            planet
            for planet, active in enumerate(state.active_planets)
            if active
            and Terrain(state.terrains[planet])
            not in (Terrain.VOLCANIC, Terrain.TRANSDIM)
            and state._distance(pi_planet, planet) <= 4
        )
        owners = [-1] * len(state.owners)
        buildings = [Building.EMPTY] * len(state.buildings)
        owners[pi_planet] = 0
        buildings[pi_planet] = Building.PLANETARY_INSTITUTE
        tracks = list(state.players[0].tracks)
        tracks[Track.TERRAFORMING] = 5
        tracks[Track.NAVIGATION] = 5
        players = list(state.players)
        players[0] = replace(
            players[0],
            credits=30,
            ore=15,
            knowledge=0,
            qic=10,
            tracks=tuple(tracks),
            colonized_types=1 << int(Terrain.VOLCANIC),
        )
        state = replace(
            state,
            round_number=1,
            placement_step=len(state.placement_order),
            booster_selection_step=len(state.booster_selection_order),
            player_to_move=0,
            owners=tuple(owners),
            buildings=tuple(int(building) for building in buildings),
            players=tuple(players),
        )
        action = state.build_action(target)
        self.assertIn(action, state.legal_actions())

        after = state.apply(action)
        record = _interactive_action_record(
            state,
            after,
            action,
            move=1,
            player=0,
            role="human",
        )
        self.assertIn("GEO-PI", [item["code"] for item in record["components"]])
        self.assertIn(
            {"resource": "knowledge", "amount": 3},
            record["effects"][0]["gains"],
        )

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
            with urlopen(f"{base}/styles.css", timeout=5) as response:
                styles = response.read().decode("utf-8")
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
            player_board_assets = []
            for number in range(1, 15):
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
            self.assertIn("play-research-stage", page)
            self.assertIn("play-research-markers", page)
            self.assertNotIn("play-research-players", page)
            self.assertNotIn("玩家科研等级", page)
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
            self.assertIn("function renderLiveResearchBoard", app_script)
            self.assertIn("function undoInteractiveTurn", app_script)
            self.assertIn("function updateLivePlayRole", app_script)
            self.assertIn("function planetAtPlayEvent", app_script)
            self.assertIn("function renderPlayActionEntry", app_script)
            self.assertIn("function renderPlayAutomaticStep", app_script)
            self.assertIn("PLAY_LOG_RESOURCE_LABELS", app_script)
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
            self.assertEqual(hashlib.sha256(research_board_image).hexdigest().upper(), RESEARCH_BOARD_SHA256)
            self.assertTrue(all(
                content_type == "image/jpeg"
                and content.startswith(b"\xff\xd8")
                and len(content) > 600_000
                for content_type, content in player_board_assets
            ))
            self.assertEqual(len({content for _content_type, content in player_board_assets}), 14)
            self.assertEqual(
                tuple(hashlib.sha256(content).hexdigest().upper() for _, content in player_board_assets),
                PLAYER_BOARD_SHA256,
            )
            self.assertIn("function factionPlayerBoardAsset", app_script)
            self.assertNotIn("function factionBoardAsset", app_script)
            self.assertNotIn("data-faction-board", app_script)
            self.assertIn("source=bga-260630-1810", app_script)
            self.assertNotIn("BGA 完整主板", app_script)
            self.assertIn("starting_credits", app_script)
            self.assertIn("starting_ore", app_script)
            self.assertIn("starting_knowledge", app_script)
            self.assertNotIn("TILE_ART_IDS", app_script)
            self.assertIn("const number = Number(id) + 1", app_script)
            self.assertIn("width: 12.1311%", styles)
            self.assertIn("top: 8.2380%", styles)
            self.assertIn(".research-tech-slot.advanced.track-0 { left: 4.2623%; }", styles)
            self.assertIn(".research-tech-slot.standard.track-0 { left: 2.2951%; }", styles)
            self.assertIn(".research-tech-slot.free-2 { left: 72.5410%; }", styles)
            for name, (content_type, content) in tile_assets.items():
                expected_type = "image/gif" if name.endswith(".gif") else "image/jpeg"
                self.assertEqual(content_type, expected_type)
                self.assertTrue(content.startswith(b"GIF" if expected_type == "image/gif" else b"\xff\xd8"))
                self.assertGreater(len(content), 1_000, name)
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
            self.assertFalse(game["can_undo"])
            initial_state = game["state"]
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
            first_log = game["history"][0]
            self.assertEqual(first_log["phase"], "starting_placement")
            self.assertTrue(any(
                component["kind"] == "planet"
                and component["code"].startswith("P-")
                for component in first_log["components"]
            ))
            self.assertIn("effects", first_log)
            self.assertIn("changes", first_log)
            self.assertEqual(first_log["automatic_steps"], [])
            self.assertEqual(game["current_role"], "ai")
            self.assertTrue(game["can_undo"])
            self.assertEqual(game["undo_count"], 1)

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
            self.assertTrue(game["can_undo"])
            self.assertEqual(game["undo_count"], 2)

            _, game = self.post_json(f"{base}/api/play/undo", {})
            self.assertEqual(game["undone_actions"], 2)
            self.assertEqual(game["move"], 0)
            self.assertEqual(game["history"], [])
            self.assertIsNone(game["last_action"])
            self.assertIsNone(game["last_search"])
            self.assertFalse(game["can_undo"])
            self.assertEqual(game["current_role"], "human")
            self.assertEqual(game["state"], initial_state)

            with urlopen(f"{base}/api/play", timeout=5) as response:
                restored = json.loads(response.read())
            self.assertEqual(restored["session_id"], game["session_id"])
            self.assertEqual(restored["move"], 0)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_interactive_action_ledger_marks_technology_tile_id(self) -> None:
        state = GaiaState.initial(
            2,
            seed=7,
            faction_indices=(0, 2),
            first_player=0,
        )
        state = replace(state, round_number=1, player_to_move=0)
        action = state.tech_action(Track.TERRAFORMING)
        after = state._apply_tech(Track.TERRAFORMING)

        entry = _interactive_action_record(
            state,
            after,
            action,
            move=1,
            player=0,
            role="human",
        )

        tile = state.standard_tech_tiles[Track.TERRAFORMING]
        self.assertTrue(any(
            component["kind"] == "standard_tech"
            and component["id"] == tile
            and component["code"] == f"TEC-S{tile + 1:02d}"
            for component in entry["components"]
        ))
        self.assertTrue(any(
            change["kind"] == "tech" and change["id"] == tile
            for change in entry["effects"][0]["changes"]
        ))

    def test_interactive_action_ledger_tracks_brainstone_selection_and_spending(self) -> None:
        state = GaiaState.initial(
            2,
            seed=7,
            faction_indices=(4, 0),
            first_player=0,
        )
        players = list(state.players)
        players[0] = replace(
            players[0],
            bowl_one=0,
            bowl_two=0,
            bowl_three=2,
            brainstone_bowl=3,
            ore=0,
        )
        state = replace(
            state,
            players=tuple(players),
            round_number=1,
            player_to_move=0,
        )

        selected = state.apply(BRAINSTONE_ACTION)
        selection_entry = _interactive_action_record(
            state,
            selected,
            BRAINSTONE_ACTION,
            move=1,
            player=0,
            role="human",
        )
        self.assertEqual(selection_entry["kind"], "brainstone")
        self.assertTrue(any(
            component["code"] == "TAK-BRAINSTONE"
            and component["relation"] == "selected"
            for component in selection_entry["components"]
        ))
        self.assertTrue(any(
            change["kind"] == "brainstone_selection" and change["after"]
            for change in selection_entry["effects"][0]["changes"]
        ))

        action = selected.power_action(PowerAction.ORE_TWO)
        after = selected.apply(action)
        spending_entry = _interactive_action_record(
            selected,
            after,
            action,
            move=2,
            player=0,
            role="human",
        )
        self.assertTrue(any(
            component["code"] == "TAK-BRAINSTONE"
            and component["relation"] == "spent"
            for component in spending_entry["components"]
        ))
        self.assertTrue(any(
            change["kind"] == "brainstone"
            and change["before"] == 3
            and change["after"] == 1
            for change in spending_entry["effects"][0]["changes"]
        ))

    def test_interactive_action_ledger_tracks_terrans_gaia_conversion(self) -> None:
        state = GaiaState.initial(
            2,
            seed=7,
            faction_indices=(0, 2),
            first_player=0,
        )
        players = list(state.players)
        players[0] = replace(players[0], ore=4, bowl_two=1, gaia_power=4)
        state = replace(
            state,
            players=tuple(players),
            round_number=2,
            player_to_move=0,
            pending_gaia_conversion_player=0,
            pending_gaia_conversion_power=4,
        )

        converted = state.apply(TERRANS_GAIA_ORE_ACTION)
        entry = _interactive_action_record(
            state,
            converted,
            TERRANS_GAIA_ORE_ACTION,
            move=1,
            player=0,
            role="human",
        )
        self.assertEqual(entry["phase"], "gaia_conversion")
        self.assertEqual(entry["kind"], "terrans_gaia_ore")
        self.assertEqual(
            entry["effects"][0]["costs"],
            [{"resource": "gaia_conversion_power", "amount": 3}],
        )
        self.assertIn({"resource": "ore", "amount": 1}, entry["effects"][0]["gains"])
        self.assertTrue(any(
            component["code"] == "TER-PI"
            for component in entry["components"]
        ))
        self.assertTrue(any(
            change["kind"] == "gaia_conversion_budget"
            and change["before"] == 4
            and change["after"] == 1
            for change in entry["effects"][0]["changes"]
        ))

        finished = converted.apply(TERRANS_GAIA_FINISH_ACTION)
        finish_entry = _interactive_action_record(
            converted,
            finished,
            TERRANS_GAIA_FINISH_ACTION,
            move=2,
            player=0,
            role="human",
        )
        self.assertEqual(finish_entry["kind"], "terrans_gaia_finish")
        self.assertEqual(finished.players[0].gaia_power, 0)
        self.assertEqual(finished.players[0].bowl_two, 5)

    def test_interactive_game_selects_starting_boosters_before_round_one(self) -> None:
        server = create_dashboard_server(self.metrics, port=0, quiet=True)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base = f"http://127.0.0.1:{server.server_port}"
            _, game = self.post_json(
                f"{base}/api/play/start",
                {
                    "players": 2,
                    "seed": 31,
                    "first_player": 0,
                    "factions": [0, 2],
                    "roles": ["human", "human"],
                    "simulations": 1,
                },
            )
            while game["state"]["phase"] == "starting_placement":
                _, game = self.post_json(
                    f"{base}/api/play/action",
                    {"action": game["legal_actions"][0]["id"]},
                )

            self.assertEqual(game["state"]["phase"], "booster_selection")
            self.assertEqual(game["state"]["round"], 0)
            self.assertEqual(game["state"]["current_player"], 1)
            self.assertEqual(game["state"]["booster_selection"]["order"], [1, 0])
            self.assertEqual(len(game["legal_actions"]), 5)
            self.assertTrue(all(
                action["kind"] == "select_booster"
                and action["target"] is None
                and isinstance(action["booster"], int)
                for action in game["legal_actions"]
            ))
            resources_before = [
                (player["credits"], player["ore"], player["knowledge"])
                for player in game["state"]["players"]
            ]

            selected = []
            while game["state"]["phase"] == "booster_selection":
                selected.append(game["legal_actions"][0]["booster"])
                _, game = self.post_json(
                    f"{base}/api/play/action",
                    {"action": game["legal_actions"][0]["id"]},
                )

            self.assertEqual(game["state"]["phase"], "round")
            self.assertEqual(game["state"]["round"], 1)
            self.assertEqual(game["state"]["current_player"], 0)
            self.assertEqual(len(set(selected)), 2)
            self.assertEqual(
                [player["booster"] for player in game["state"]["players"]],
                [selected[1], selected[0]],
            )
            resources_after = [
                (player["credits"], player["ore"], player["knowledge"])
                for player in game["state"]["players"]
            ]
            self.assertNotEqual(resources_after, resources_before)
            self.assertEqual(
                [item["kind"] for item in game["history"][-2:]],
                ["select_booster", "select_booster"],
            )
            final_booster_log = game["history"][-1]
            self.assertTrue(any(
                component["kind"] == "booster"
                and component["code"].startswith("BST-")
                for component in final_booster_log["components"]
            ))
            self.assertEqual(len(final_booster_log["automatic_steps"]), 1)
            income_step = final_booster_log["automatic_steps"][0]
            self.assertEqual(income_step["kind"], "round_income")
            self.assertEqual(income_step["round"], 1)
            self.assertFalse(income_step["gaia_phase"])
            self.assertTrue(any(
                component["kind"] == "round_scoring"
                and component["code"].startswith("RND-")
                for component in income_step["components"]
            ))
            self.assertEqual(
                {effect["player"] for effect in income_step["effects"]},
                {0, 1},
            )
            self.assertTrue(all(
                any(source["kind"] == "booster" for source in effect["sources"])
                for effect in income_step["effects"]
            ))
            self.assertTrue(all(effect["gains"] for effect in income_step["effects"]))

            build_action = next(
                action for action in game["legal_actions"] if action["kind"] == "build"
            )
            _, game = self.post_json(
                f"{base}/api/play/action",
                {"action": build_action["id"]},
            )
            build_log = game["history"][-1]
            build_costs = {
                item["resource"]: item["amount"]
                for item in build_log["effects"][0]["costs"]
            }
            self.assertEqual(build_costs["credits"], 2)
            self.assertGreaterEqual(build_costs["ore"], 1)
            self.assertTrue(any(
                change["kind"] == "building"
                and change["planet"] == build_action["target"]
                for change in build_log["changes"]
            ))

            research_action = next(
                action for action in game["legal_actions"] if action["kind"] == "research"
            )
            _, game = self.post_json(
                f"{base}/api/play/action",
                {"action": research_action["id"]},
            )
            research_log = game["history"][-1]
            self.assertIn(
                {"resource": "knowledge", "amount": 4},
                research_log["effects"][0]["costs"],
            )
            self.assertTrue(any(
                component["kind"] == "research_track"
                and component["code"].startswith("TRK-")
                for component in research_log["components"]
            ))

            live_state = server.play_session["state"]
            current_player = live_state.current_player
            forced_players = tuple(
                replace(player, passed=index != current_player)
                for index, player in enumerate(live_state.players)
            )
            server.play_session["state"] = replace(
                live_state,
                players=forced_players,
                player_to_move=current_player,
            )
            with urlopen(f"{base}/api/play", timeout=5) as response:
                game = json.loads(response.read())
            pass_action = next(
                action for action in game["legal_actions"] if action["kind"] == "pass_booster"
            )
            _, game = self.post_json(
                f"{base}/api/play/action",
                {"action": pass_action["id"]},
            )
            round_two_log = game["history"][-1]
            self.assertEqual(round_two_log["kind"], "pass_booster")
            self.assertEqual(round_two_log["automatic_steps"][0]["kind"], "round_income")
            self.assertEqual(round_two_log["automatic_steps"][0]["round"], 2)
            self.assertTrue(round_two_log["automatic_steps"][0]["gaia_phase"])
            self.assertEqual(game["state"]["round"], 2)
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
