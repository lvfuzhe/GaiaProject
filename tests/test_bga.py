from __future__ import annotations

import json
import shutil
import threading
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.request import Request, urlopen

from gaiazero.bga import (
    BgaClient,
    BgaReplayError,
    _ReplayLinkParser,
    _normalize_replay_address,
    convert_bga_replay,
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


def map_payload() -> dict[str, object]:
    return {
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
                    )
                ],
                [
                    notice(
                        "notifyResearch",
                        player_name="Alice",
                        whichResearch=1,
                        knowledgeCost=4,
                        playerId=PLAYER_ONE,
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
        f"<script>globalThis.bgaGameData = {json.dumps(game_data())};\n"
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
        with self.assertRaises(BgaReplayError):
            _normalize_replay_address("https://example.com/gamereview?table=1")
        with self.assertRaises(BgaReplayError):
            _normalize_replay_address("https://boardgamearena.com/player?id=1")

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
            body = json.dumps(
                {
                    "username": "alice",
                    "password": "secret",
                    "replay_address": f"https://boardgamearena.com/gamereview?table={TABLE_ID}",
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
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
            metrics.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
