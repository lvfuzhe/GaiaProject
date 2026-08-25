from __future__ import annotations

import gzip
import json
import shutil
import threading
import unittest
from http.cookiejar import Cookie
from pathlib import Path
from unittest.mock import patch
from urllib.request import Request, urlopen

from gaiazero.bga import (
    BgaClient,
    BgaReplayError,
    BgaSessionStore,
    _ReplayLinkParser,
    _extract_completesetup_data,
    _normalize_replay_address,
    convert_bga_replay,
    import_bga_replay,
)
from gaiazero.dashboard import create_dashboard_server
from gaiazero.telemetry import (
    build_local_history_index,
    read_local_game_trace,
    write_local_game,
)


TABLE_ID = 123456789
PLAYER_ONE = 101
PLAYER_TWO = 202


def player_payload(race_id: int, *, score: int = 10) -> dict[str, object]:
    return {
        "raceId": race_id,
        "gold": 15,
        "ore": 4,
        "knowledge": 3,
        "qic": 1,
        "score": score,
        "research": [0, 0, 0, 0, 0, 0, 0],
        "power": [0, 2, 4, 0],
        "techs": [],
        "fedTiles": [],
        "buildings": {"4": 8, "5": 4, "6": 3, "7": 1, "8": 1, "9": 1},
        "numAvailGaiaformers": 0,
        "boosterId": 0,
        "hasPassed": 0,
        "race5brainstonePos": -1,
        "numGaiaformersInGaiaArea": 0,
    }


def map_payload(*, satellite_players: tuple[int, ...] = ()) -> dict[str, object]:
    payload: dict[str, object] = {
        "0": {
            "0": {
                "tileNum": 1,
                "planetType": 1,
                "buildings": [
                    {"buildingId": 4, "playerId": str(PLAYER_ONE), "isPartOfFed": 0}
                ],
                "isTileCenter": 1,
                "q": 0,
                "r": 0,
            },
            "1": {
                "tileNum": 1,
                "planetType": 4,
                "buildings": [
                    {"buildingId": 4, "playerId": str(PLAYER_TWO), "isPartOfFed": 0}
                ],
                "isTileCenter": 0,
                "q": 0,
                "r": 1,
            },
        }
    }
    if satellite_players:
        payload["1"] = {
            "0": {
                "tileNum": 1,
                "planetType": 0,
                "buildings": [
                    {
                        "buildingId": 1,
                        "playerId": str(player_id),
                        "isPartOfFed": 1,
                    }
                    for player_id in satellite_players
                ],
                "isTileCenter": 0,
                "q": 1,
                "r": 0,
            }
        }
    return payload


def notice(kind: str, **args: object) -> dict[str, object]:
    return {"type": kind, "log": "", "args": args}


def replay_packets() -> list[dict[str, object]]:
    one = player_payload(1)
    two = player_payload(3)
    one_after = {**one, "gold": 13, "ore": 3, "buildings": {**one["buildings"], "4": 7}}
    packets = [
        {
            "packet_id": str(move),
            "move_id": str(move),
            "table_id": TABLE_ID,
            "time": str(1_700_000_000 + move),
            "data": data,
        }
        for move, data in enumerate(
            (
                [
                    notice(
                        "notifyChooseRace",
                        player_name="Alice",
                        raceId=1,
                        player=one,
                        playerId=PLAYER_ONE,
                    ),
                    notice(
                        "notifyChooseRace",
                        player_name="Bob",
                        raceId=3,
                        player=two,
                        playerId=PLAYER_TWO,
                    ),
                    notice("notifyPlayerOrder", playerList=[PLAYER_ONE, PLAYER_TWO]),
                ],
                [
                    notice(
                        "notifyPlaceStartingBldg",
                        player_name="Alice",
                        q=0,
                        r=0,
                        buildingId=4,
                        playerId=PLAYER_ONE,
                    )
                ],
                [
                    notice(
                        "notifyPlaceStartingBldg",
                        player_name="Bob",
                        q=0,
                        r=1,
                        buildingId=4,
                        playerId=PLAYER_TWO,
                    )
                ],
                [
                    notice(
                        "notifyChooseBoosterTile",
                        player_name="Bob",
                        boosterId=1,
                        playerId=PLAYER_TWO,
                    ),
                    notice(
                        "notifyChooseBoosterTile",
                        player_name="Alice",
                        boosterId=2,
                        playerId=PLAYER_ONE,
                    ),
                    notice("notifyGaiaDone", players={str(PLAYER_ONE): one, str(PLAYER_TWO): two}),
                ],
                [
                    notice(
                        "notifyBuild",
                        player_name="Alice",
                        payStr="[ORE1][GOLD2]",
                        q=0,
                        r=0,
                        buildingId=4,
                        player=one_after,
                        map=map_payload(),
                        playerId=PLAYER_ONE,
                    ),
                    notice(
                        "notifyScore",
                        player_name="Alice",
                        playerId=PLAYER_ONE,
                        vp=2,
                        desc="round scoring",
                    ),
                    notice(
                        "notifyFormFederation",
                        player_name="Alice",
                        playerId=PLAYER_ONE,
                        satellites=[{"q": 1, "r": 0}],
                        buildings=[{"q": 0, "r": 0}],
                    ),
                    notice(
                        "notifyTakeFedToken",
                        player_name="Alice",
                        playerId=PLAYER_ONE,
                        fedTokenId=2,
                    ),
                ],
                [
                    notice(
                        "notifyResearch",
                        player_name="Alice",
                        whichResearch=1,
                        knowledgeCost=4,
                        map=map_payload(
                            satellite_players=(PLAYER_ONE, PLAYER_TWO),
                        ),
                        playerId=PLAYER_ONE,
                    ),
                    notice(
                        "notifyPass",
                        player_name="Alice",
                        playerId=PLAYER_ONE,
                        boosterId=3,
                        vp=5,
                    ),
                    notice("notifyRoundEnd", roundNum=6, playerList=[PLAYER_ONE, PLAYER_TWO]),
                    {"type": "simpleNode", "log": "End of game", "args": []},
                ],
            ),
            start=1,
        )
    ]
    return packets


def game_data() -> dict[str, object]:
    return {
        "gamename": "gaiaproject",
        "tableId": str(TABLE_ID),
        "players": [{"id": PLAYER_ONE, "no": 1}, {"id": PLAYER_TWO, "no": 2}],
        "tableOptions": [
            {"id": 100, "value": 12, "value_displayed": "Random map"},
            {"id": 101, "value": 2, "value_displayed": "Reduced map"},
        ],
    }


def initial_setup_payload() -> dict[str, object]:
    return {
        "board": {
            "roundNum": 0,
            "techs": [5, 6, 2, 8, 3, 7, 9, 1, 4],
            "advTechs": [17, 18, 19, 23, 12, 11],
            "roundBonus": [0, 9, 8, 1, 10, 5, 3],
            "endGameBonus": [2, 6],
            "availFedTokens": [0, 3, 2, 3, 3, 3, 3, 0],
            "availBoosters": [1, 2, 3, 8, 9],
            "bonusFedToken": 2,
            "displayMap": 2,
        },
        "map": map_payload(),
        "players": {},
        "playerList": [PLAYER_ONE, PLAYER_TWO],
        "playerorder": [PLAYER_ONE, PLAYER_TWO],
        "passOrder": [],
        "gamestate": {"id": 1, "name": "gameSetup"},
    }


def review_html() -> str:
    return f"""
    <div class="score-entry">
      <div class="name"><a class="playername" href="/player?id={PLAYER_ONE}">Alice</a></div>
      <div class="score">(123 <img alt="pt"></div>
      <a href="/archive/replay/260101-1200/?table={TABLE_ID}&amp;player={PLAYER_ONE}">Replay</a>
    </div>
    <div class="score-entry">
      <div class="name"><a class="playername" href="/player?id={PLAYER_TWO}">Bob</a></div>
      <div class="score">(98 <img alt="pt"></div>
      <a href="/archive/replay/260101-1200/?table={TABLE_ID}&amp;player={PLAYER_TWO}">Replay</a>
    </div>
    """


def replay_html() -> str:
    logs = {"status": 1, "data": {"valid": 1, "data": replay_packets()}}
    return (
        "<script>globalThis.gameui.completesetup("
        "'gaiaproject', 'Gaia {Project}', 1, 2, 0, '0', '', "
        f"{json.dumps(initial_setup_payload())}, null);\n"
        f"globalThis.bgaGameData = {json.dumps(game_data())};\n"
        f"globalThis.g_gamelogs = {json.dumps(logs)};</script>"
    )


class StubBgaClient(BgaClient):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[str, bytes | None]] = []

    def _request_text(
        self,
        url: str,
        *,
        data: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> str:
        del headers
        self.calls.append((url, data))
        if url.endswith("/account"):
            return "<script>const cfg = {requestToken: 'token-123'};</script>"
        if "loginUserWithPassword" in url:
            return '{"status":1,"data":{"player_id":101}}'
        if "/gamereview" in url:
            return review_html()
        if "/archive/replay/" in url:
            return replay_html()
        raise AssertionError(url)


class BgaImportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(__file__).parent / ".artifacts" / "bga-history"
        shutil.rmtree(self.root, ignore_errors=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def test_review_parser_extracts_replay_links_players_and_scores(self) -> None:
        parser = _ReplayLinkParser(TABLE_ID)
        parser.feed(review_html())

        self.assertEqual(len(parser.links), 2)
        self.assertEqual(parser.players[PLAYER_ONE], {"name": "Alice", "score": 123})
        self.assertEqual(parser.players[PLAYER_TWO], {"name": "Bob", "score": 98})

    def test_only_bga_replay_addresses_are_accepted(self) -> None:
        address = f"https://boardgamearena.com/gamereview?table={TABLE_ID}"
        self.assertEqual(_normalize_replay_address(address), (address, TABLE_ID, "review"))

        localized_address = f"https://zh-cn.boardgamearena.com/gamereview?table={TABLE_ID}"
        self.assertEqual(
            _normalize_replay_address(localized_address),
            (localized_address, TABLE_ID, "review"),
        )

        with self.assertRaises(BgaReplayError):
            _normalize_replay_address("https://example.com/gamereview?table=1")
        with self.assertRaises(BgaReplayError):
            _normalize_replay_address("https://evilboardgamearena.com/gamereview?table=1")
        with self.assertRaises(BgaReplayError):
            _normalize_replay_address(
                "https://boardgamearena.com.evil.example/gamereview?table=1"
            )
        with self.assertRaises(BgaReplayError):
            _normalize_replay_address("http://zh-cn.boardgamearena.com/gamereview?table=1")
        with self.assertRaises(BgaReplayError):
            _normalize_replay_address("https://boardgamearena.com/player?id=1")

    def test_archive_completesetup_parser_recovers_move_zero_state(self) -> None:
        initial = _extract_completesetup_data(replay_html())

        self.assertEqual(initial["board"], initial_setup_payload()["board"])
        self.assertEqual(initial["map"], map_payload())

    def test_short_lived_client_logs_in_and_converts_a_review(self) -> None:
        client = StubBgaClient()
        client.login("alice", "secret")
        record = client.download(
            f"https://boardgamearena.com/gamereview?table={TABLE_ID}"
        )

        login_body = next(data for url, data in client.calls if "loginUserWithPassword" in url)
        self.assertIn(b"username=alice", login_body)
        self.assertIn(b"password=secret", login_body)
        self.assertNotIn("secret", json.dumps(record))
        self.assertEqual(record["trace"]["summary"]["scores"], [123, 98])
        self.assertEqual(record["trace"]["summary"]["moves"], 6)
        self.assertTrue(record["trace"]["steps"][-1]["state"]["terminal"])
        self.assertTrue(record["bga"]["initial_setup_complete"])
        self.assertEqual(record["bga"]["table_options"], game_data()["tableOptions"])
        self.assertEqual(record["bga"]["initial_setup"]["map"], map_payload())

        setup = record["trace"]["steps"][0]["state"]["setup"]
        self.assertEqual(setup["map"]["size"], "reduced")
        self.assertEqual([tile["id"] for tile in setup["boosters"]], [0, 1, 2, 7, 8])
        self.assertTrue(all(tile["owner"] == -1 for tile in setup["boosters"]))
        self.assertEqual(
            [tile["id"] for tile in setup["round_scoring"]],
            [8, 7, 0, 9, 4, 2],
        )
        self.assertEqual([tile["id"] for tile in setup["final_scoring"]], [1, 5])
        self.assertEqual(
            [tile["id"] for tile in setup["standard_tech"]],
            [4, 5, 1, 7, 2, 6, 8, 0, 3],
        )
        self.assertEqual(
            [tile["id"] for tile in setup["advanced_tech"]],
            [7, 8, 9, 13, 2, 1],
        )
        self.assertEqual(setup["terraforming_federation"]["id"], 1)
        self.assertEqual(setup["federation_supply"], [3, 2, 3, 3, 3, 3])
        federation_step = record["trace"]["steps"][5]["state"]
        self.assertEqual(
            federation_step["satellites"],
            [{"id": 2, "q": 1, "r": 0, "owners": [0]}],
        )
        self.assertEqual(federation_step["players"][0]["satellites"], 1)
        final_state = record["trace"]["steps"][-1]["state"]
        self.assertEqual(
            final_state["satellites"],
            [{"id": 2, "q": 1, "r": 0, "owners": [0, 1]}],
        )
        self.assertEqual(
            [player["satellites"] for player in final_state["players"]],
            [1, 1],
        )
        self.assertEqual(
            record["trace"]["steps"][-1]["state"]["setup"]["federation_supply"],
            [3, 1, 3, 3, 3, 3],
        )

    def test_client_requests_and_decodes_gzip_responses(self) -> None:
        class ResponseHeaders(dict[str, str]):
            def get_content_charset(self) -> str:
                return "utf-8"

        class Response:
            headers = ResponseHeaders({"Content-Encoding": "gzip"})

            def __enter__(self) -> Response:
                return self

            def __exit__(self, *_args: object) -> None:
                return None

            def read(self) -> bytes:
                return gzip.compress("BGA 压缩响应".encode("utf-8"))

        class Opener:
            request: Request | None = None

            def open(self, request: Request, *, timeout: float) -> Response:
                self.request = request
                self.timeout = timeout
                return Response()

        client = BgaClient(timeout=12.0)
        opener = Opener()
        client._opener = opener

        result = client._request_text("https://boardgamearena.com/account")

        self.assertEqual(result, "BGA 压缩响应")
        self.assertEqual(opener.request.get_header("Accept-encoding"), "gzip")
        self.assertEqual(opener.timeout, 12.0)

    def test_final_scoring_tiles_and_vp_reasons_are_recovered_from_bga_log(self) -> None:
        packets = replay_packets()
        final_notices = packets[-1]["data"]
        final_notices[-1:-1] = [
            notice(
                "notifyScore",
                player_name="Alice",
                playerId=PLAYER_ONE,
                vp=18,
                desc="Most structures",
            ),
            notice(
                "notifyScore",
                player_name="Bob",
                playerId=PLAYER_TWO,
                vp=12,
                desc="Most satellites",
            ),
            notice("notifyScore", player_name="Alice", playerId=PLAYER_ONE, vp=8),
            notice("notifyScore", player_name="Alice", playerId=PLAYER_ONE, vp=2),
        ]

        record = convert_bga_replay(
            table_id=TABLE_ID,
            source_url=f"https://boardgamearena.com/gamereview?table={TABLE_ID}",
            replay_url=f"https://boardgamearena.com/archive/replay/test/?table={TABLE_ID}",
            game_data=game_data(),
            packets=packets,
        )

        self.assertFalse(record["bga"]["initial_setup_complete"])
        self.assertIsNone(record["bga"]["initial_setup"])
        final_step = record["trace"]["steps"][-1]
        self.assertEqual(
            [tile["id"] for tile in final_step["state"]["setup"]["final_scoring"]],
            [1, 5],
        )
        self.assertEqual(
            [event["reason"] for event in final_step["record"]["vp"]["events"][-2:]],
            ["科研轨终局计分", "剩余资源计分"],
        )

    def test_cookie_cache_round_trips_through_client(self) -> None:
        cookie = Cookie(
            version=0,
            name="bga-session",
            value="cookie-secret",
            port=None,
            port_specified=False,
            domain=".boardgamearena.com",
            domain_specified=True,
            domain_initial_dot=True,
            path="/",
            path_specified=True,
            secure=True,
            expires=None,
            discard=True,
            comment=None,
            comment_url=None,
            rest={"HttpOnly": None},
            rfc2109=False,
        )
        source = BgaClient()
        source._cookies.set_cookie(cookie)

        restored = BgaClient(cookies=source.export_cookies())

        self.assertEqual(restored.export_cookies(), source.export_cookies())

    @unittest.skipUnless(__import__("os").name == "nt", "Windows DPAPI only")
    def test_session_store_encrypts_credentials_for_current_windows_user(self) -> None:
        path = self.root / ".bga-session.bin"
        cookies = [{"domain": ".boardgamearena.com", "name": "sid", "value": "cookie"}]
        store = BgaSessionStore(path)

        store.save(username="alice", password="secret", cookies=cookies)

        payload = store.load()
        self.assertEqual(payload["username"], "alice")
        self.assertEqual(payload["password"], "secret")
        self.assertEqual(payload["cookies"], cookies)
        self.assertNotIn(b"alice", path.read_bytes())
        self.assertNotIn(b"secret", path.read_bytes())
        self.assertEqual(
            store.metadata(),
            {
                "saved": True,
                "username": "alice",
                "cookie_count": 1,
                "updated_at": payload["updated_at"],
            },
        )
        store.clear()
        self.assertFalse(path.exists())

    @unittest.skipUnless(__import__("os").name == "nt", "Windows DPAPI only")
    def test_import_reuses_saved_credentials_and_cookie_without_login(self) -> None:
        session_path = self.root / ".bga-session.bin"
        cached_cookies = [
            {"domain": ".boardgamearena.com", "name": "sid", "value": "cached-cookie"}
        ]
        BgaSessionStore(session_path).save(
            username="alice",
            password="secret",
            cookies=cached_cookies,
        )
        review = _ReplayLinkParser(TABLE_ID)
        review.feed(review_html())
        record = convert_bga_replay(
            table_id=TABLE_ID,
            source_url=f"https://boardgamearena.com/gamereview?table={TABLE_ID}",
            replay_url=f"https://boardgamearena.com/archive/replay/test/?table={TABLE_ID}",
            game_data=game_data(),
            packets=replay_packets(),
            review_players=review.players,
        )
        calls: list[object] = []

        class CachedSessionClient:
            def __init__(self, *, timeout: float, cookies: list[dict[str, object]]) -> None:
                calls.append(("init", timeout, cookies))
                self.cookies = cookies

            def login(self, username: str, password: str) -> None:
                calls.append(("login", username, password))

            def download(self, replay_address: str) -> dict[str, object]:
                calls.append(("download", replay_address))
                return record

            def export_cookies(self) -> list[dict[str, object]]:
                return self.cookies

        archive = self.root / f"bga-{TABLE_ID}.json"
        with (
            patch("gaiazero.bga.BgaClient", CachedSessionClient),
            patch("gaiazero.bga.write_local_game", return_value=archive),
        ):
            result = import_bga_replay(
                username="",
                password="",
                replay_address=f"https://boardgamearena.com/gamereview?table={TABLE_ID}",
                history_path=self.root,
                session_path=session_path,
                remember=True,
            )

        self.assertEqual(calls[0], ("init", 30.0, cached_cookies))
        self.assertFalse(any(call[0] == "login" for call in calls))
        self.assertTrue(result["used_cached_session"])
        self.assertTrue(result["session_saved"])

    def test_converted_replay_round_trips_through_local_history(self) -> None:
        review = _ReplayLinkParser(TABLE_ID)
        review.feed(review_html())
        record = convert_bga_replay(
            table_id=TABLE_ID,
            source_url=f"https://boardgamearena.com/gamereview?table={TABLE_ID}",
            replay_url=f"https://boardgamearena.com/archive/replay/test/?table={TABLE_ID}",
            game_data=game_data(),
            packets=replay_packets(),
            review_players=review.players,
        )
        target = write_local_game(self.root, record)

        self.assertTrue(target.is_file())
        index = build_local_history_index(self.root)
        self.assertEqual(index["runs"][0]["source"], "bga")
        trace = read_local_game_trace(self.root, run_id=f"bga-{TABLE_ID}")
        self.assertIsNotNone(trace)
        self.assertEqual(trace["source"], "bga")
        self.assertEqual([step["move"] for step in trace["steps"]], list(range(7)))
        self.assertEqual(trace["summary"]["scores"], [123, 98])
        score_step = trace["steps"][5]
        self.assertEqual(score_step["record"]["vp"]["before"], [10, 10])
        self.assertEqual(score_step["record"]["vp"]["after"], [12, 10])
        self.assertEqual(score_step["record"]["vp"]["events"][0]["delta"], 2)
        final_ledger = trace["steps"][-1]["record"]["vp"]
        self.assertEqual(final_ledger["raw_after"], [17, 10])
        self.assertEqual(final_ledger["after"], [123, 98])
        self.assertFalse(final_ledger["matches_result_page"])
        self.assertEqual(
            [change["delta"] for change in final_ledger["reconciliation"]],
            [106, 88],
        )
        final = trace["steps"][-1]["state"]
        self.assertEqual([planet["terrain"] for planet in final["planets"]], [0, 1])
        self.assertEqual([player["faction"] for player in final["players"]], ["Terrans", "Xenos"])

    def test_dashboard_import_endpoint_does_not_echo_credentials(self) -> None:
        metrics = self.root.parent / "bga-dashboard.jsonl"
        metrics.parent.mkdir(parents=True, exist_ok=True)
        metrics.unlink(missing_ok=True)
        server = create_dashboard_server(
            metrics,
            port=0,
            history_path=self.root,
            quiet=True,
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        result = {
            "ok": True,
            "table_id": TABLE_ID,
            "run_id": f"bga-{TABLE_ID}",
            "archive_path": str(self.root / f"bga-{TABLE_ID}.json"),
            "moves": 6,
            "scores": [123, 98],
            "players": [],
            "imported_at": "2026-01-01T00:00:00+00:00",
        }
        try:
            with urlopen(
                f"http://127.0.0.1:{server.server_port}/import/bga",
                timeout=5,
            ) as page_response:
                page = page_response.read().decode("utf-8")
            self.assertIn("bga-import-form", page)
            self.assertIn("bga-import-username", page)
            self.assertIn("bga-import-password", page)
            self.assertIn("bga-import-address", page)
            self.assertNotIn('value="alice"', page)
            self.assertNotIn('value="secret"', page)
            with urlopen(
                f"http://127.0.0.1:{server.server_port}/api/bga/session",
                timeout=5,
            ) as session_response:
                session_payload = json.loads(session_response.read().decode("utf-8"))
            self.assertEqual(session_payload["saved"], False)
            body = json.dumps(
                {
                    "username": "alice",
                    "password": "secret",
                    "replay_address": f"https://boardgamearena.com/gamereview?table={TABLE_ID}",
                    "remember": True,
                }
            ).encode("utf-8")
            request = Request(
                f"http://127.0.0.1:{server.server_port}/api/bga/import",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with patch("gaiazero.dashboard.import_bga_replay", return_value=result) as importer:
                with urlopen(request, timeout=5) as response:
                    response_body = response.read().decode("utf-8")
                    payload = json.loads(response_body)
            self.assertEqual(response.status, 201)
            self.assertEqual(payload["run_id"], f"bga-{TABLE_ID}")
            self.assertNotIn("alice", response_body)
            self.assertNotIn("secret", response_body)
            importer.assert_called_once()
            self.assertEqual(importer.call_args.kwargs["session_path"], server.bga_session_path)
            self.assertTrue(importer.call_args.kwargs["remember"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
            metrics.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
