from __future__ import annotations

import copy
import gzip
import json
import os
import re
from collections import defaultdict
from datetime import UTC, datetime
from html.parser import HTMLParser
from http.cookiejar import Cookie, CookieJar
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlencode, urljoin, urlparse
from urllib.request import (
    HTTPCookieProcessor,
    HTTPRedirectHandler,
    Request,
    build_opener,
)

from gaiazero.game.gaia_state import (
    ADVANCED_TECH_TILES,
    BOOSTER_LABELS,
    FEDERATION_TILES,
    FINAL_SCORING_TILES,
    ROUND_SCORING_TILES,
    STANDARD_TECH_TILES,
    TRACK_COUNT,
    Track,
)
from gaiazero.telemetry import write_local_game


ACCOUNT_URL = "https://boardgamearena.com/account"
LOGIN_URL = "https://en.boardgamearena.com/account/auth/loginUserWithPassword.html"
BGA_ROOT_DOMAIN = "boardgamearena.com"
REQUEST_TOKEN_PATTERNS = (
    re.compile(r"requestToken\s*:\s*['\"]([^'\"]+)['\"]"),
    re.compile(r"['\"]request_token['\"]\s*:\s*['\"]([^'\"]+)['\"]"),
)

BGA_TERRAIN_TO_LOCAL = {
    1: 0,  # Terra
    4: 1,  # Desert
    5: 2,  # Swamp
    3: 3,  # Volcanic
    2: 4,  # Oxide
    6: 5,  # Titanium
    7: 6,  # Ice
    9: 7,  # Transdim
    8: 8,  # Gaia
    10: 9,  # Lost Planet
}
BGA_BUILDINGS = {
    4: "mine",
    5: "trading_station",
    6: "research_lab",
    7: "planetary_institute",
    8: "academy",
    9: "academy",
}
BUILDING_TOTALS = {
    "mine": 8,
    "trading_station": 4,
    "research_lab": 3,
    "planetary_institute": 1,
    "academy": 2,
}
FACTION_NAMES = (
    "Terrans",
    "Lantids",
    "Xenos",
    "Gleens",
    "Taklons",
    "Ambas",
    "Hadsch Hallas",
    "Ivits",
    "Geodens",
    "Bal T'aks",
    "Firaks",
    "Bescods",
    "Nevlas",
    "Itars",
)
FACTION_HOMES = (0, 0, 1, 1, 2, 2, 4, 4, 3, 3, 5, 5, 6, 6)
TRACK_NAMES = (
    "地形改造",
    "航行",
    "人工智能",
    "盖亚计划",
    "经济",
    "科学",
)

BGA_FINAL_SCORING = {
    "most structures in federations": {
        "id": 0,
        "key": "federation-structures",
        "label": "Structures in federations",
    },
    "most structures": {
        "id": 1,
        "key": "structures",
        "label": "Total structures",
    },
    "most planet types": {
        "id": 2,
        "key": "planet-types",
        "label": "Colonized planet types",
    },
    "most gaia planets": {
        "id": 3,
        "key": "gaia-planets",
        "label": "Colonized Gaia planets",
    },
    "most sectors": {
        "id": 4,
        "key": "sectors",
        "label": "Colonized sectors",
    },
    "most satellites": {
        "id": 5,
        "key": "satellites",
        "label": "Placed satellites and space stations",
    },
}


class BgaError(RuntimeError):
    """Base class for user-facing BGA import failures."""


class BgaAuthenticationError(BgaError):
    pass


class BgaRateLimitError(BgaError):
    pass


class BgaNetworkError(BgaError):
    pass


class BgaReplayError(BgaError):
    pass


class BgaSessionError(BgaError):
    pass


class BgaSessionStore:
    """Persist BGA credentials and cookies encrypted for the current Windows user."""

    HEADER = b"GAIAZERO-BGA-SESSION-V1\n"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()

    def load(self) -> dict[str, Any] | None:
        if not self.path.is_file():
            return None
        source = self.path.read_bytes()
        if not source.startswith(self.HEADER):
            raise BgaSessionError("BGA 本地会话文件格式无效，请清除后重新保存")
        try:
            payload = json.loads(_unprotect_session(source[len(self.HEADER):]))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise BgaSessionError(
                "无法读取 BGA 本地会话，可能由其他 Windows 用户创建"
            ) from error
        if not isinstance(payload, dict) or payload.get("version") != 1:
            raise BgaSessionError("BGA 本地会话版本不受支持")
        return payload

    def save(
        self,
        *,
        username: str,
        password: str,
        cookies: list[dict[str, Any]],
    ) -> None:
        payload = {
            "version": 1,
            "username": username.strip(),
            "password": password,
            "cookies": cookies,
            "updated_at": datetime.now(UTC).isoformat(),
        }
        encrypted = _protect_session(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f"{self.path.name}.tmp")
        temporary.write_bytes(self.HEADER + encrypted)
        temporary.replace(self.path)

    def clear(self) -> None:
        self.path.unlink(missing_ok=True)

    def metadata(self) -> dict[str, Any]:
        payload = self.load()
        if payload is None:
            return {"saved": False, "username": "", "cookie_count": 0, "updated_at": None}
        cookies = payload.get("cookies")
        return {
            "saved": True,
            "username": str(payload.get("username") or ""),
            "cookie_count": len(cookies) if isinstance(cookies, list) else 0,
            "updated_at": payload.get("updated_at"),
        }


class _BgaRedirectHandler(HTTPRedirectHandler):
    def redirect_request(
        self,
        request: Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> Request | None:
        _validate_bga_url(newurl)
        return super().redirect_request(request, fp, code, msg, headers, newurl)


class _ReplayLinkParser(HTMLParser):
    def __init__(self, table_id: int) -> None:
        super().__init__(convert_charrefs=True)
        self.table_id = table_id
        self.links: list[str] = []
        self.players: dict[int, dict[str, Any]] = {}
        self._entry_depth = 0
        self._current_player_id: int | None = None
        self._capture_name = False
        self._capture_score = False
        self._name_parts: list[str] = []
        self._score_parts: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        tag = tag.lower()
        attributes = dict(attrs)
        classes = set((attributes.get("class") or "").split())
        if tag == "div" and "score-entry" in classes:
            self._entry_depth = 1
            self._current_player_id = None
            self._name_parts = []
            self._score_parts = []
            return
        if self._entry_depth and tag == "div":
            self._entry_depth += 1
            if "score" in classes:
                self._capture_score = True
        if tag != "a":
            return
        href = attributes.get("href") or ""
        if self._entry_depth and "playername" in classes:
            values = parse_qs(urlparse(href).query)
            self._current_player_id = _player_id(values.get("id", [None])[0])
            self._capture_name = True
        parsed = urlparse(href)
        if not parsed.path.startswith("/archive/replay/"):
            return
        values = parse_qs(parsed.query)
        if values.get("table", [""])[0] == str(self.table_id):
            self.links.append(href)

    def handle_data(self, data: str) -> None:
        if self._capture_name:
            self._name_parts.append(data)
        if self._capture_score:
            self._score_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag == "a":
            self._capture_name = False
        if not self._entry_depth or tag != "div":
            return
        if self._capture_score:
            self._capture_score = False
        self._entry_depth -= 1
        if self._entry_depth != 0:
            return
        score_match = re.search(r"-?\d+", "".join(self._score_parts))
        if self._current_player_id is not None:
            self.players[self._current_player_id] = {
                "name": "".join(self._name_parts).strip(),
                "score": int(score_match.group()) if score_match else None,
            }


class BgaClient:
    """Short-lived authenticated BGA client used by one manual import."""

    def __init__(
        self,
        *,
        timeout: float = 30.0,
        cookies: list[dict[str, Any]] | None = None,
    ) -> None:
        self.timeout = timeout
        self._cookies = CookieJar()
        for cookie in cookies or []:
            restored = _restore_cookie(cookie)
            if restored is not None:
                self._cookies.set_cookie(restored)
        self._opener = build_opener(
            _BgaRedirectHandler(),
            HTTPCookieProcessor(self._cookies),
        )

    def export_cookies(self) -> list[dict[str, Any]]:
        return [_serialize_cookie(cookie) for cookie in self._cookies]

    def login(self, username: str, password: str) -> None:
        if not username.strip() or not password:
            raise BgaAuthenticationError("请输入 BGA 账号和密码")
        account_html = self._request_text(ACCOUNT_URL)
        request_token = _find_request_token(account_html)
        if not request_token:
            raise BgaAuthenticationError(
                "BGA 登录页没有返回请求令牌，可能触发了人机验证"
            )
        form = urlencode(
            {
                "username": username.strip(),
                "password": password,
                "remember_me": "false",
                "request_token": request_token,
            }
        ).encode("utf-8")
        response_text = self._request_text(
            LOGIN_URL,
            data=form,
            headers={
                "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
                "Referer": ACCOUNT_URL,
                "X-Requested-With": "XMLHttpRequest",
            },
        )
        try:
            response = json.loads(response_text)
        except json.JSONDecodeError as error:
            raise BgaAuthenticationError("BGA 登录响应无法解析") from error
        if not isinstance(response, dict) or _as_int(response.get("status"), 0) != 1:
            raise BgaAuthenticationError(_login_error_message(response))

    def download(self, replay_address: str) -> dict[str, Any]:
        address, table_id, path_kind = _normalize_replay_address(replay_address)
        review_url = f"https://boardgamearena.com/gamereview?table={table_id}"
        if path_kind == "review":
            review_html = self._request_text(address)
        else:
            review_html = self._request_text(review_url)
        review_parser = _ReplayLinkParser(table_id)
        review_parser.feed(review_html)
        if path_kind == "review":
            if not review_parser.links:
                raise BgaReplayError(
                    "该页面没有找到可下载的复盘，账号可能无权查看或对局尚未结束"
                )
            replay_url = urljoin(address, review_parser.links[0])
            _validate_bga_url(replay_url)
        else:
            replay_url = address

        replay_html = self._request_text(replay_url)
        try:
            game_data = _extract_json_assignment(replay_html, "globalThis.bgaGameData")
            game_logs = _extract_json_assignment(replay_html, "globalThis.g_gamelogs")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise BgaReplayError(
                "复盘页没有返回结构化行动日志，账号可能无权查看该复盘"
            ) from error
        try:
            initial_state = _extract_completesetup_data(replay_html)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            # Older archive pages may only contain the action log.
            initial_state = None
        game_name = str(game_data.get("gamename") or "").lower()
        if game_name != "gaiaproject":
            raise BgaReplayError("该地址不是《盖亚计划》复盘")
        try:
            packets = game_logs["data"]["data"]
        except (KeyError, TypeError) as error:
            raise BgaReplayError("BGA 复盘日志结构不完整") from error
        if not isinstance(packets, list) or not packets:
            raise BgaReplayError("BGA 复盘没有行动数据")
        return convert_bga_replay(
            table_id=table_id,
            source_url=address,
            replay_url=replay_url,
            game_data=game_data,
            packets=packets,
            review_players=review_parser.players,
            initial_state=initial_state,
        )

    def _request_text(
        self,
        url: str,
        *,
        data: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> str:
        _validate_bga_url(url)
        request_headers = {
            "Accept": "text/html,application/json;q=0.9,*/*;q=0.8",
            "Accept-Encoding": "gzip",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.7",
            "User-Agent": "GaiaZero-BGA-Importer/1.0",
        }
        request_headers.update(headers or {})
        request = Request(url, data=data, headers=request_headers)
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                body = response.read()
                charset = response.headers.get_content_charset() or "utf-8"
                content_encoding = response.headers.get("Content-Encoding", "").lower()
        except HTTPError as error:
            if error.code == 429:
                raise BgaRateLimitError("BGA 请求过于频繁，请稍后再试") from error
            if error.code in (401, 403):
                raise BgaAuthenticationError("BGA 拒绝了登录或复盘访问") from error
            raise BgaNetworkError(f"BGA 返回 HTTP {error.code}") from error
        except (TimeoutError, URLError, OSError) as error:
            raise BgaNetworkError("无法连接 BGA，请检查网络后重试") from error
        if content_encoding == "gzip":
            try:
                body = gzip.decompress(body)
            except (OSError, EOFError) as error:
                raise BgaNetworkError("BGA returned incomplete compressed data") from error
        return body.decode(charset, errors="replace")


def import_bga_replay(
    *,
    username: str,
    password: str,
    replay_address: str,
    history_path: str | Path,
    timeout: float = 30.0,
    session_path: str | Path | None = None,
    remember: bool = False,
) -> dict[str, Any]:
    """Download one BGA replay and atomically add it to the local archive."""

    store = BgaSessionStore(session_path) if session_path is not None else None
    saved: dict[str, Any] | None = None
    if store is not None:
        try:
            saved = store.load()
        except BgaSessionError:
            if not username.strip() or not password:
                raise

    supplied_username = username.strip()
    saved_username = str((saved or {}).get("username") or "").strip()
    effective_username = supplied_username or saved_username
    effective_password = password
    if not effective_password and saved is not None and effective_username == saved_username:
        effective_password = str(saved.get("password") or "")
    if not effective_username or not effective_password:
        raise BgaAuthenticationError("请输入 BGA 账号和密码，或先保存有效的本地会话")

    saved_cookies = (saved or {}).get("cookies")
    reusable_cookies = (
        saved_cookies
        if effective_username == saved_username and isinstance(saved_cookies, list)
        else []
    )
    client = BgaClient(timeout=timeout, cookies=reusable_cookies)
    used_cached_session = False
    if reusable_cookies:
        try:
            record = client.download(replay_address)
            used_cached_session = True
        except (BgaAuthenticationError, BgaReplayError):
            client.login(effective_username, effective_password)
            record = client.download(replay_address)
    else:
        client.login(effective_username, effective_password)
        record = client.download(replay_address)

    if remember and store is not None:
        store.save(
            username=effective_username,
            password=effective_password,
            cookies=client.export_cookies(),
        )
    target = write_local_game(history_path, record)
    trace = record["trace"]
    return {
        "ok": True,
        "table_id": record["bga"]["table_id"],
        "run_id": record["run_id"],
        "archive_path": str(target.resolve()),
        "moves": trace["summary"]["moves"],
        "scores": trace["summary"]["scores"],
        "players": record["bga"]["players"],
        "imported_at": record["updated_at"],
        "session_saved": bool(remember and store is not None),
        "used_cached_session": used_cached_session,
    }


def convert_bga_replay(
    *,
    table_id: int,
    source_url: str,
    replay_url: str,
    game_data: dict[str, Any],
    packets: list[dict[str, Any]],
    review_players: dict[int, dict[str, Any]] | None = None,
    initial_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert BGA packets into GaiaZero's persisted replay contract."""

    state = _BgaReplayState(table_id, game_data, packets, initial_state)
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    raw_packets: list[dict[str, Any]] = []
    for packet in packets:
        if not isinstance(packet, dict):
            continue
        raw_packets.append(packet)
        try:
            move = int(packet.get("move_id"))
        except (TypeError, ValueError):
            continue
        if move > 0:
            grouped[move].append(packet)
    if not grouped:
        raise BgaReplayError("BGA 复盘没有有效的行动编号")
    maximum_move = max(grouped)
    if set(grouped) != set(range(1, maximum_move + 1)):
        raise BgaReplayError("BGA 复盘行动编号不连续，下载内容可能不完整")

    steps: list[dict[str, Any]] = [
        {
            "move": 0,
            "player": None,
            "action": None,
            "action_label": "BGA 初始状态",
            "legal_actions": None,
            "state": state.snapshot(),
        }
    ]
    for move in range(1, maximum_move + 1):
        move_packets = grouped[move]
        notifications = [
            notice
            for packet in move_packets
            for notice in packet.get("data", [])
            if isinstance(notice, dict)
        ]
        actor = state.actor_for(notifications)
        label, kind = _action_label(notifications)
        vp_before = [player["vp"] for player in state.players]
        vp_events = _vp_events(notifications, state.player_index)
        for notice in notifications:
            state.apply_notification(notice)
        vp_after = [player["vp"] for player in state.players]
        vp_changes = [
            {
                "player": player,
                "bga_player_id": state.players[player]["bga_player_id"],
                "name": state.players[player].get("name") or f"P{player}",
                "before": vp_before[player],
                "delta": vp_after[player] - vp_before[player],
                "after": vp_after[player],
            }
            for player in range(len(state.players))
            if vp_after[player] != vp_before[player]
        ]
        record = {
            "role": "bga",
            "kind": kind,
            "label": label,
            "components": _notification_components(notifications),
            "effects": [],
            "changes": [],
            "vp": {
                "before": vp_before,
                "changes": vp_changes,
                "after": vp_after,
                "events": vp_events,
            },
            "bga": {
                "packet_ids": [packet.get("packet_id") for packet in move_packets],
                "notifications": [_compact_notification(item) for item in notifications],
            },
        }
        steps.append(
            {
                "move": move,
                "player": actor,
                "action": move,
                "action_label": label,
                "legal_actions": None,
                "record": record,
                "state": state.snapshot(),
            }
        )

    if review_players:
        final_players = steps[-1]["state"]["players"]
        final_vp_before = [player["vp"] for player in final_players]
        for player in final_players:
            result = review_players.get(player["bga_player_id"])
            if not result:
                continue
            if result.get("name"):
                player["name"] = str(result["name"])
            if result.get("score") is not None:
                player["vp"] = _as_int(result["score"], player["vp"])
        steps[-1]["state"]["scores"] = [player["vp"] for player in final_players]
        final_vp_after = [player["vp"] for player in final_players]
        reconciliation = [
            {
                "player": player,
                "bga_player_id": final_players[player]["bga_player_id"],
                "name": final_players[player].get("name") or f"P{player}",
                "before": final_vp_before[player],
                "delta": final_vp_after[player] - final_vp_before[player],
                "after": final_vp_after[player],
            }
            for player in range(len(final_players))
            if final_vp_after[player] != final_vp_before[player]
        ]
        final_ledger = steps[-1]["record"]["vp"]
        final_ledger["raw_after"] = list(final_ledger["after"])
        final_ledger["after"] = final_vp_after
        final_ledger["reconciliation"] = reconciliation
        final_ledger["matches_result_page"] = not reconciliation
        steps[-1]["record"]["components"].append(
            {
                "code": "BGA-FINAL-SCORE",
                "label": "使用 BGA 结果页终局分数",
            }
        )

    started_at = _packet_timestamp(raw_packets[0])
    completed_at = _packet_timestamp(raw_packets[-1])
    start_epoch = _packet_epoch(raw_packets[0])
    end_epoch = _packet_epoch(raw_packets[-1])
    duration = max(0, end_epoch - start_epoch) if start_epoch and end_epoch else None
    final_snapshot = steps[-1]["state"]
    imported_at = datetime.now(UTC).isoformat()
    players = [
        {
            "seat": player["id"],
            "bga_player_id": player["bga_player_id"],
            "name": player.get("name") or f"BGA {player['bga_player_id']}",
            "faction": player.get("faction") or "Unknown",
            "score": player["vp"],
        }
        for player in final_snapshot["players"]
    ]
    replay_complete = bool(final_snapshot["terminal"])
    return {
        "run_id": f"bga-{table_id}",
        "source": "bga",
        "status": "complete" if replay_complete else "active",
        "ruleset": "bga-gaiaproject",
        "engine": "bga-replay-importer-v1",
        "started_at": started_at,
        "updated_at": imported_at,
        "completed_at": completed_at if replay_complete else None,
        "roles": ["bga"] * len(players),
        "config": {
            "players": len(players),
            "table_id": table_id,
            "source_url": source_url,
        },
        "trace": {
            "run_id": f"bga-{table_id}",
            "iteration": 1,
            "game": 1,
            "started_at": started_at,
            "completed_at": completed_at if replay_complete else None,
            "summary": {
                "moves": maximum_move,
                "positions": maximum_move,
                "scores": final_snapshot["scores"],
                "returns": None,
                "duration_seconds": duration,
            },
            "trace_complete": set(grouped) == set(range(1, maximum_move + 1)),
            "captured_moves": maximum_move,
            "steps": steps,
        },
        "bga": {
            "table_id": table_id,
            "game": "gaiaproject",
            "source_url": source_url,
            "replay_url": replay_url,
            "players": players,
            "downloaded_at": imported_at,
            "initial_setup_complete": initial_state is not None,
            "initial_setup": _compact_initial_setup(initial_state),
            "table_options": copy.deepcopy(game_data.get("tableOptions") or []),
            "log_packets": raw_packets,
        },
    }


class _BgaReplayState:
    def __init__(
        self,
        table_id: int,
        game_data: dict[str, Any],
        packets: list[dict[str, Any]],
        initial_state: dict[str, Any] | None = None,
    ) -> None:
        self.table_id = table_id
        self.notifications = [
            notice
            for packet in packets
            if isinstance(packet, dict)
            for notice in packet.get("data", [])
            if isinstance(notice, dict)
        ]
        player_ids = _collect_player_ids(game_data, self.notifications)
        self.player_index = {player_id: index for index, player_id in enumerate(player_ids)}
        names, initial_players = _collect_initial_players(self.notifications)
        self.players = [
            _empty_player(index, player_id, names.get(player_id))
            for index, player_id in enumerate(player_ids)
        ]
        for player_id, payload in initial_players.items():
            if player_id in self.player_index:
                _apply_player_payload(self.players[self.player_index[player_id]], payload)

        self.placement_order = [
            self.player_index[player_id]
            for notice in self.notifications
            if notice.get("type") == "notifyPlaceStartingBldg"
            for player_id in [_player_id((notice.get("args") or {}).get("playerId"))]
            if player_id in self.player_index
        ]
        self.booster_order = [
            self.player_index[player_id]
            for notice in self.notifications
            if notice.get("type") == "notifyChooseBoosterTile"
            for player_id in [_player_id((notice.get("args") or {}).get("playerId"))]
            if player_id in self.player_index
        ][: len(self.players)]
        self.placement_step = 0
        self.booster_step = 0
        self.round = 0
        self.terminal = False
        self.current_player = self.placement_order[0] if self.placement_order else 0
        self.first_player = _find_first_player(self.notifications, self.player_index)
        board = initial_state.get("board") if isinstance(initial_state, dict) else None
        if not isinstance(board, dict):
            board = {}
        self.booster_owners = _initial_boosters(board)
        self.round_scoring = _initial_round_scoring(board)
        self.final_scoring = (
            _initial_final_scoring(board)
            or _collect_final_scoring(self.notifications)
        )
        self.standard_tech = _initial_standard_tech(board)
        self.advanced_tech = _initial_advanced_tech(board)
        self.terraforming_federation = _initial_terraforming_federation(board)
        self.federation_supply = _initial_federation_supply(board)

        initial_map = initial_state.get("map") if isinstance(initial_state, dict) else None
        first_map, all_planet_coordinates = _collect_map_topology(
            self.notifications,
            initial_map if isinstance(initial_map, dict) else None,
        )
        display_map = _as_int(board.get("displayMap"), 0)
        self.map_size = (
            "reduced"
            if display_map == 2
            or (display_map == 0 and len(_extract_sectors(first_map)) < 10)
            else "normal"
        )
        ordered_coordinates = sorted(all_planet_coordinates, key=lambda item: (item[1], item[0]))
        self.planet_ids = {
            coordinate: index for index, coordinate in enumerate(ordered_coordinates)
        }
        self.planets: dict[tuple[int, int], dict[str, Any]] = {}
        self.space_stations: list[dict[str, Any]] = []
        self.sectors = _extract_sectors(first_map)
        for coordinate, cell in first_map.items():
            planet_type = _as_int(cell.get("planetType"), 0)
            if planet_type <= 0:
                continue
            self.planets[coordinate] = self._empty_planet(coordinate, cell)
        self._refresh_structure_counts()

    def actor_for(self, notifications: list[dict[str, Any]]) -> int | None:
        priority_types = {
            "notifyPlaceStartingBldg",
            "notifyChooseBoosterTile",
            "notifyBuild",
            "notifyUpgrade",
            "notifyStartGaia",
            "notifyResearch",
            "notifyGainTech",
            "notifyFormFederation",
            "notifyAction",
            "notifyConvert",
            "notifyPass",
        }
        for notice in notifications:
            if notice.get("type") not in priority_types:
                continue
            player_id = _player_id((notice.get("args") or {}).get("playerId"))
            if player_id in self.player_index:
                return self.player_index[player_id]
        return self.current_player if self.current_player in range(len(self.players)) else None

    def apply_notification(self, notice: dict[str, Any]) -> None:
        notice_type = str(notice.get("type") or "")
        args = notice.get("args") or {}
        if not isinstance(args, dict):
            args = {}
        player_id = _player_id(args.get("playerId"))
        player = self._player(player_id)

        payload = args.get("player")
        if player is not None and isinstance(payload, dict):
            _apply_player_payload(player, payload)
        players_payload = args.get("players")
        if isinstance(players_payload, dict):
            for source_id, source_player in players_payload.items():
                target = self._player(_player_id(source_id))
                if target is not None and isinstance(source_player, dict):
                    _apply_player_payload(target, source_player)

        map_payload = args.get("map")
        if isinstance(map_payload, dict):
            self._apply_map(map_payload)

        if notice_type == "notifyPlaceStartingBldg" and player is not None:
            self._set_building(args, player, _as_int(args.get("buildingId"), 4))
            self.placement_step = min(len(self.placement_order), self.placement_step + 1)
        elif notice_type == "notifyChooseBoosterTile" and player is not None:
            self._set_booster(player, args.get("boosterId"))
            self.booster_step = min(len(self.booster_order), self.booster_step + 1)
        elif notice_type == "notifyBuild" and player is not None:
            if not isinstance(map_payload, dict):
                self._set_building(args, player, _as_int(args.get("buildingId"), 4))
        elif notice_type == "notifyUpgrade" and player is not None:
            player["credits"] = max(0, player["credits"] - _as_int(args.get("spentGold"), 0))
            player["ore"] = max(0, player["ore"] - _as_int(args.get("spentOre"), 0))
            self._set_building(args, player, _as_int(args.get("buildingId"), 0))
        elif notice_type == "notifyStartGaia" and player is not None:
            power = args.get("power")
            if isinstance(power, list) and len(power) >= 4:
                player["gaia_power"] = _as_int(power[0], 0)
                player["power"] = [_as_int(value, 0) for value in power[1:4]]
            player["qic"] = max(0, player["qic"] - _as_int(args.get("spentQics"), 0))
            self._set_building(args, player, 2)
        elif notice_type == "notifyResearch" and player is not None:
            track = _as_int(args.get("whichResearch"), 0) - 1
            if 0 <= track < 6:
                player["tracks"][track] = min(5, player["tracks"][track] + 1)
            player["knowledge"] = max(
                0,
                player["knowledge"] - _as_int(args.get("knowledgeCost"), 0),
            )
        elif notice_type == "notifyChargePower" and player is not None:
            power = args.get("power")
            if isinstance(power, list) and len(power) >= 4:
                player["gaia_power"] = _as_int(power[0], 0)
                player["power"] = [_as_int(value, 0) for value in power[1:4]]
            brainstone = _as_int(args.get("race5brainstonePos"), -1)
            player["brainstone_bowl"] = brainstone if brainstone > 0 else 0
        elif notice_type == "notifyIncome" and player is not None:
            income = args.get("income")
            if isinstance(income, list) and len(income) >= 6:
                player["round_income"] = {
                    "credits": _as_int(income[0], 0),
                    "ore": _as_int(income[1], 0),
                    "knowledge": _as_int(income[2], 0),
                    "qic": _as_int(income[3], 0),
                    "power_charge": sum(_as_int(value, 0) for value in income[4]),
                    "power_tokens": sum(_as_int(value, 0) for value in income[5]),
                }
        elif notice_type == "notifyScore" and player is not None:
            player["vp"] += _as_int(args.get("vp"), 0)
        elif notice_type == "notifyGainTech" and player is not None:
            tech_id = _as_int(args.get("techId"), 0)
            if 1 <= tech_id <= 9 and tech_id - 1 not in player["tech_tiles"]:
                player["tech_tiles"].append(tech_id - 1)
            elif tech_id > 9 and tech_id - 10 not in player["advanced_tech_tiles"]:
                player["advanced_tech_tiles"].append(tech_id - 10)
        elif notice_type == "notifyTakeFedToken" and player is not None:
            federation_tile = _as_int(args.get("fedTokenId"), 0)
            player["federation_tiles"].append(federation_tile)
            player["federations"] = len(player["federation_tiles"])
            supply_index = federation_tile - 1
            if (
                0 <= supply_index < len(self.federation_supply)
                and self.federation_supply[supply_index] > 0
            ):
                self.federation_supply[supply_index] -= 1
        elif notice_type == "notifyFormFederation" and player is not None:
            self._mark_federation(args, player)
        elif notice_type == "notifyPass" and player is not None:
            player["vp"] += _as_int(args.get("vp"), 0)
            player["passed"] = True
            self._set_booster(player, args.get("boosterId"))
        elif notice_type == "notifyRoundEnd":
            completed_round = _as_int(args.get("roundNum"), self.round)
            if completed_round >= 6:
                self.round = 6
                self.terminal = True
            else:
                self.round = max(self.round, completed_round + 1)
                for item in self.players:
                    item["passed"] = False
        elif notice_type == "notifyGaiaDone" and self.round == 0:
            self.round = 1
        elif notice_type == "simpleNode" and "end of game" in str(notice.get("log", "")).lower():
            self.round = max(6, self.round)
            self.terminal = True

        if notice_type == "gameStateChange":
            active_id = _player_id(args.get("active_player"))
            if active_id in self.player_index:
                self.current_player = self.player_index[active_id]
        elif notice_type == "notifyPlayerOrder":
            order = args.get("playerList") or []
            if order:
                first_id = _player_id(order[0])
                if first_id in self.player_index:
                    self.first_player = self.player_index[first_id]
        self._refresh_structure_counts()

    def snapshot(self) -> dict[str, Any]:
        phase = (
            "starting_placement"
            if self.placement_step < len(self.placement_order)
            else "booster_selection"
            if self.booster_step < len(self.booster_order)
            else "terminal"
            if self.terminal
            else "round"
        )
        planets = [
            copy.deepcopy(planet)
            for _coordinate, planet in sorted(
                self.planets.items(), key=lambda item: item[1]["id"]
            )
        ]
        setup_factions = [
            {
                "player": player["id"],
                "id": player["faction_id"],
                "name": player["faction"],
                "home_terrain": player["home_terrain"],
                "starting_planets": [],
            }
            for player in self.players
        ]
        setup = {
            "seed": self.table_id,
            "map": {
                "method": "bga-import",
                "size": self.map_size,
                "sector_count": len(self.sectors),
                "sectors": copy.deepcopy(self.sectors),
                "planet_sources": [
                    {
                        "id": planet["id"],
                        "q": planet["q"],
                        "r": planet["r"],
                        "terrain": planet["terrain"],
                        "sector": planet["sector"],
                    }
                    for planet in planets
                ],
            },
            "factions": setup_factions,
            "boosters": [
                {"id": booster, "label": BOOSTER_LABELS[booster], "owner": owner}
                for booster, owner in sorted(self.booster_owners.items())
                if 0 <= booster < len(BOOSTER_LABELS)
            ],
            "round_scoring": copy.deepcopy(self.round_scoring),
            "final_scoring": copy.deepcopy(self.final_scoring),
            "standard_tech": copy.deepcopy(self.standard_tech),
            "advanced_tech": copy.deepcopy(self.advanced_tech),
            "terraforming_federation": copy.deepcopy(self.terraforming_federation),
            "federation_supply": list(self.federation_supply),
        }
        return {
            "ruleset": "bga-gaiaproject",
            "source": "bga",
            "round": max(0, min(self.round, 6)),
            "max_rounds": 6,
            "phase": phase,
            "placement": {
                "active": phase == "starting_placement",
                "step": self.placement_step,
                "total": len(self.placement_order),
                "order": list(self.placement_order),
                "remaining": max(0, len(self.placement_order) - self.placement_step),
            },
            "booster_selection": {
                "active": phase == "booster_selection",
                "step": self.booster_step,
                "total": len(self.booster_order),
                "order": list(self.booster_order),
                "remaining": max(0, len(self.booster_order) - self.booster_step),
            },
            "round_scoring": None,
            "current_player": None if self.terminal else self.current_player,
            "first_player": self.first_player,
            "terminal": self.terminal,
            "scores": [player["vp"] for player in self.players],
            "players": copy.deepcopy(self.players),
            "planets": planets,
            "space_stations": copy.deepcopy(self.space_stations),
            "setup": setup,
        }

    def _empty_planet(
        self,
        coordinate: tuple[int, int],
        cell: dict[str, Any],
    ) -> dict[str, Any]:
        q, r = coordinate
        tile = _as_int(cell.get("tileNum"), 0)
        return {
            "id": self.planet_ids[coordinate],
            "q": q,
            "r": r,
            "source_q": q,
            "source_r": r,
            "source_id": self.planet_ids[coordinate],
            "sector": max(-1, tile - 1),
            "terrain": BGA_TERRAIN_TO_LOCAL.get(_as_int(cell.get("planetType"), 0), 7),
            "owner": -1,
            "building": "empty",
            "building_id": 0,
            "coexisting_mine_owner": -1,
            "coexisting_mine_federated": False,
            "gaiaformer": -1,
            "federated": False,
        }

    def _player(self, player_id: int | None) -> dict[str, Any] | None:
        index = self.player_index.get(player_id)
        return self.players[index] if index is not None else None

    def _set_booster(self, player: dict[str, Any], value: Any) -> None:
        booster = _as_int(value, 0) - 1
        if booster < 0:
            return
        for key, owner in list(self.booster_owners.items()):
            if owner == player["id"]:
                self.booster_owners[key] = -1
        self.booster_owners[booster] = player["id"]
        player["booster"] = booster

    def _set_building(
        self,
        args: dict[str, Any],
        player: dict[str, Any],
        building_id: int,
    ) -> None:
        coordinate = (_as_int(args.get("q"), 0), _as_int(args.get("r"), 0))
        if building_id == 3:
            self.space_stations = [
                station
                for station in self.space_stations
                if (station["q"], station["r"]) != coordinate
            ]
            self.space_stations.append(
                {
                    "id": len(self.space_stations),
                    "q": coordinate[0],
                    "r": coordinate[1],
                    "owner": player["id"],
                    "federated": False,
                }
            )
            return
        planet = self.planets.get(coordinate)
        if planet is None:
            if coordinate not in self.planet_ids:
                self.planet_ids[coordinate] = len(self.planet_ids)
            cell = {
                "tileNum": 0,
                "planetType": _local_to_bga_terrain(player["home_terrain"]),
            }
            planet = self._empty_planet(coordinate, cell)
            self.planets[coordinate] = planet
        if building_id == 2:
            planet["gaiaformer"] = player["id"]
            return
        building = BGA_BUILDINGS.get(building_id)
        if not building:
            return
        if planet["owner"] >= 0 and planet["owner"] != player["id"] and player["faction_id"] == 1:
            planet["coexisting_mine_owner"] = player["id"]
            planet["coexisting_mine_federated"] = False
        else:
            planet["owner"] = player["id"]
            planet["building"] = building
            planet["building_id"] = building_id
            planet["gaiaformer"] = -1

    def _apply_map(self, map_payload: dict[str, Any]) -> None:
        cells = _flatten_map(map_payload)
        self.planets = {}
        self.space_stations = []
        satellite_counts = [0] * len(self.players)
        for coordinate, cell in cells.items():
            planet_type = _as_int(cell.get("planetType"), 0)
            buildings = cell.get("buildings") or []
            if planet_type > 0:
                if coordinate not in self.planet_ids:
                    self.planet_ids[coordinate] = len(self.planet_ids)
                planet = self._empty_planet(coordinate, cell)
                structures: list[tuple[int, int, bool]] = []
                for item in buildings:
                    if not isinstance(item, dict):
                        continue
                    building_id = _as_int(item.get("buildingId"), 0)
                    source_player_id = _player_id(item.get("playerId"))
                    owner = self.player_index.get(source_player_id)
                    if owner is None:
                        continue
                    federated = bool(_as_int(item.get("isPartOfFed"), 0))
                    if building_id == 1:
                        satellite_counts[owner] += 1
                    elif building_id == 2:
                        planet["gaiaformer"] = owner
                    elif building_id in BGA_BUILDINGS:
                        structures.append((owner, building_id, federated))
                if structures:
                    primary_index = next(
                        (
                            index
                            for index, (owner, _building, _fed) in enumerate(structures)
                            if self.players[owner]["faction_id"] != 1
                        ),
                        0,
                    )
                    owner, building_id, federated = structures.pop(primary_index)
                    planet.update(
                        owner=owner,
                        building=BGA_BUILDINGS[building_id],
                        building_id=building_id,
                        federated=federated,
                        gaiaformer=-1,
                    )
                    for coexist_owner, coexist_building, coexist_federated in structures:
                        if coexist_building == 4:
                            planet["coexisting_mine_owner"] = coexist_owner
                            planet["coexisting_mine_federated"] = coexist_federated
                            break
                self.planets[coordinate] = planet
                continue

            for item in buildings:
                if not isinstance(item, dict):
                    continue
                building_id = _as_int(item.get("buildingId"), 0)
                source_player_id = _player_id(item.get("playerId"))
                owner = self.player_index.get(source_player_id)
                if owner is None:
                    continue
                if building_id == 1:
                    satellite_counts[owner] += 1
                elif building_id == 3:
                    self.space_stations.append(
                        {
                            "id": len(self.space_stations),
                            "q": coordinate[0],
                            "r": coordinate[1],
                            "owner": owner,
                            "federated": bool(_as_int(item.get("isPartOfFed"), 0)),
                        }
                    )
        for index, count in enumerate(satellite_counts):
            self.players[index]["satellites"] = count

    def _mark_federation(self, args: dict[str, Any], player: dict[str, Any]) -> None:
        for item in args.get("buildings") or []:
            if not isinstance(item, dict):
                continue
            coordinate = (_as_int(item.get("q"), 0), _as_int(item.get("r"), 0))
            planet = self.planets.get(coordinate)
            if planet is None:
                continue
            if planet["owner"] == player["id"]:
                planet["federated"] = True
            if planet["coexisting_mine_owner"] == player["id"]:
                planet["coexisting_mine_federated"] = True
        for item in args.get("satellites") or []:
            if not isinstance(item, dict):
                continue
            player["satellites"] += 1

    def _refresh_structure_counts(self) -> None:
        counts = [defaultdict(int) for _player in self.players]
        academy_types = [[0, 0] for _player in self.players]
        colonized = [set() for _player in self.players]
        gaiaformers = [0] * len(self.players)
        for planet in self.planets.values():
            owner = _as_int(planet.get("owner"), -1)
            if 0 <= owner < len(self.players) and planet.get("building") != "empty":
                counts[owner][planet["building"]] += 1
                colonized[owner].add(planet["terrain"])
                if planet.get("building_id") == 8:
                    academy_types[owner][0] += 1
                elif planet.get("building_id") == 9:
                    academy_types[owner][1] += 1
            coexist = _as_int(planet.get("coexisting_mine_owner"), -1)
            if 0 <= coexist < len(self.players):
                counts[coexist]["mine"] += 1
                colonized[coexist].add(planet["terrain"])
            gaiaformer = _as_int(planet.get("gaiaformer"), -1)
            if 0 <= gaiaformer < len(self.players):
                gaiaformers[gaiaformer] += 1
        for index, player in enumerate(self.players):
            player["structures"] = {
                building: {
                    "built": counts[index][building],
                    "supply": max(0, total - counts[index][building]),
                }
                for building, total in BUILDING_TOTALS.items()
            }
            player["knowledge_academies"] = academy_types[index][0]
            player["qic_academies"] = academy_types[index][1]
            player["gaiaformers_on_board"] = gaiaformers[index]
            player["space_stations"] = sum(
                station["owner"] == index for station in self.space_stations
            )
            player["satellites_and_space_stations"] = (
                player["satellites"] + player["space_stations"]
            )
            mask = 0
            for terrain in colonized[index]:
                if 0 <= terrain <= 9:
                    mask |= 1 << terrain
            player["colonized_types"] = mask


def _empty_player(index: int, player_id: int, name: str | None) -> dict[str, Any]:
    return {
        "id": index,
        "bga_player_id": player_id,
        "name": name or f"BGA {player_id}",
        "faction_id": -1,
        "faction": "Unknown",
        "home_terrain": 0,
        "faction_ability": "",
        "credits": 0,
        "ore": 0,
        "knowledge": 0,
        "qic": 0,
        "vp": 0,
        "power": [0, 0, 0],
        "gaia_power": 0,
        "brainstone_bowl": 0,
        "brainstone_selected": False,
        "round_income": {
            "credits": None,
            "ore": None,
            "knowledge": None,
            "qic": None,
            "power_tokens": None,
            "power_charge": None,
        },
        "gaiaformers": 0,
        "gaiaformers_in_gaia": 0,
        "gaiaformers_on_board": 0,
        "structures": {
            key: {"built": 0, "supply": total}
            for key, total in BUILDING_TOTALS.items()
        },
        "tracks": [0] * 6,
        "satellites": 0,
        "space_stations": 0,
        "satellites_and_space_stations": 0,
        "colonized_types": 0,
        "tech_tiles": [],
        "covered_tech_tiles": [],
        "advanced_tech_tiles": [],
        "knowledge_academies": 0,
        "qic_academies": 0,
        "federations": 0,
        "federation_tiles": [],
        "booster": -1,
        "passed": False,
    }


def _apply_player_payload(player: dict[str, Any], payload: dict[str, Any]) -> None:
    race_id = _as_int(payload.get("raceId"), 0)
    if 1 <= race_id <= len(FACTION_NAMES):
        player["faction_id"] = race_id - 1
        player["faction"] = FACTION_NAMES[race_id - 1]
        player["home_terrain"] = FACTION_HOMES[race_id - 1]
    player["credits"] = _as_int(payload.get("gold"), player["credits"])
    player["ore"] = _as_int(payload.get("ore"), player["ore"])
    player["knowledge"] = _as_int(payload.get("knowledge"), player["knowledge"])
    player["qic"] = _as_int(payload.get("qic"), player["qic"])
    player["vp"] = _as_int(payload.get("score"), player["vp"])
    research = payload.get("research")
    if isinstance(research, list):
        values = research[1:7] if len(research) >= 7 else research[:6]
        player["tracks"] = [_as_int(value, 0) for value in values]
        player["tracks"] += [0] * (6 - len(player["tracks"]))
    power = payload.get("power")
    if isinstance(power, list) and len(power) >= 4:
        player["gaia_power"] = _as_int(power[0], 0)
        player["power"] = [_as_int(value, 0) for value in power[1:4]]
    player["gaiaformers"] = _as_int(
        payload.get("numAvailGaiaformers"), player["gaiaformers"]
    )
    player["gaiaformers_in_gaia"] = _as_int(
        payload.get("numGaiaformersInGaiaArea"), player["gaiaformers_in_gaia"]
    )
    booster_id = _as_int(payload.get("boosterId"), 0)
    player["booster"] = booster_id - 1 if booster_id > 0 else -1
    player["passed"] = bool(_as_int(payload.get("hasPassed"), 0))
    brainstone = _as_int(payload.get("race5brainstonePos"), -1)
    player["brainstone_bowl"] = brainstone if brainstone > 0 else 0
    techs = payload.get("techs")
    if isinstance(techs, list):
        standard: list[int] = []
        advanced: list[int] = []
        covered: list[int] = []
        for tech in techs:
            if not isinstance(tech, dict):
                continue
            tech_id = _as_int(tech.get("techId"), 0)
            if 1 <= tech_id <= 9:
                standard.append(tech_id - 1)
                if _as_int(tech.get("isCovered"), 0):
                    covered.append(tech_id - 1)
            elif tech_id > 9:
                advanced.append(tech_id - 10)
        player["tech_tiles"] = standard
        player["advanced_tech_tiles"] = advanced
        player["covered_tech_tiles"] = covered
    federation_tiles = payload.get("fedTiles")
    if isinstance(federation_tiles, list):
        player["federation_tiles"] = [
            _as_int(tile.get("fedTokenId"), 0)
            for tile in federation_tiles
            if isinstance(tile, dict)
        ]
        player["federations"] = len(player["federation_tiles"])


def _collect_player_ids(
    game_data: dict[str, Any],
    notifications: list[dict[str, Any]],
) -> list[int]:
    ordered: list[int] = []
    game_players = game_data.get("players") or []
    if isinstance(game_players, dict):
        game_players = list(game_players.values())
    if isinstance(game_players, list):
        sorted_players = sorted(
            (player for player in game_players if isinstance(player, dict)),
            key=lambda player: _as_int(player.get("no"), len(game_players)),
        )
        for player in sorted_players:
            player_id = _player_id(player.get("id"))
            if player_id is not None and player_id not in ordered:
                ordered.append(player_id)
    for notice in notifications:
        args = notice.get("args") or {}
        if not isinstance(args, dict):
            continue
        candidates = [args.get("playerId"), args.get("active_player")]
        candidates.extend((args.get("players") or {}).keys() if isinstance(args.get("players"), dict) else [])
        for value in candidates:
            player_id = _player_id(value)
            if player_id is not None and player_id not in ordered:
                ordered.append(player_id)
    if not 1 <= len(ordered) <= 4:
        raise BgaReplayError("BGA 复盘的玩家数据不完整")
    return ordered


def _collect_initial_players(
    notifications: list[dict[str, Any]],
) -> tuple[dict[int, str], dict[int, dict[str, Any]]]:
    names: dict[int, str] = {}
    initial: dict[int, dict[str, Any]] = {}
    for notice in notifications:
        args = notice.get("args") or {}
        if not isinstance(args, dict):
            continue
        player_id = _player_id(args.get("playerId"))
        player_name = args.get("player_name")
        if player_id is not None and isinstance(player_name, str) and player_name:
            names.setdefault(player_id, player_name)
        player = args.get("player")
        if player_id is not None and isinstance(player, dict):
            initial.setdefault(player_id, player)
        players = args.get("players")
        if isinstance(players, dict):
            for source_id, payload in players.items():
                source_player_id = _player_id(source_id)
                if source_player_id is not None and isinstance(payload, dict):
                    initial.setdefault(source_player_id, payload)
    return names, initial


def _collect_map_topology(
    notifications: list[dict[str, Any]],
    initial_map: dict[str, Any] | None = None,
) -> tuple[dict[tuple[int, int], dict[str, Any]], set[tuple[int, int]]]:
    first_map = _flatten_map(initial_map) if isinstance(initial_map, dict) else {}
    active_coordinates = {
        coordinate
        for coordinate, cell in first_map.items()
        if _as_int(cell.get("planetType"), 0) > 0
    }
    for notice in notifications:
        args = notice.get("args") or {}
        map_payload = args.get("map") if isinstance(args, dict) else None
        if not isinstance(map_payload, dict):
            continue
        cells = _flatten_map(map_payload)
        if not first_map:
            first_map = cells
        active_coordinates.update(
            coordinate
            for coordinate, cell in cells.items()
            if _as_int(cell.get("planetType"), 0) > 0
        )
    if not first_map:
        raise BgaReplayError("BGA 复盘没有星图数据")
    return first_map, active_coordinates


def _flatten_map(map_payload: dict[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
    cells: dict[tuple[int, int], dict[str, Any]] = {}
    for column in map_payload.values():
        if not isinstance(column, dict):
            continue
        for cell in column.values():
            if not isinstance(cell, dict):
                continue
            coordinate = (_as_int(cell.get("q"), 0), _as_int(cell.get("r"), 0))
            cells[coordinate] = cell
    return cells


def _extract_sectors(
    cells: dict[tuple[int, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    sectors: dict[int, tuple[int, int]] = {}
    for coordinate, cell in cells.items():
        tile = _as_int(cell.get("tileNum"), 0)
        if tile > 0 and _as_int(cell.get("isTileCenter"), 0):
            sectors[tile] = coordinate
    return [
        {
            "position": position,
            "tile": tile,
            "side": "solid",
            "rotation": 0,
            "q": coordinate[0],
            "r": coordinate[1],
        }
        for position, (tile, coordinate) in enumerate(sorted(sectors.items()))
    ]


def _find_first_player(
    notifications: list[dict[str, Any]],
    player_index: dict[int, int],
) -> int:
    for notice in notifications:
        if notice.get("type") != "notifyPlayerOrder":
            continue
        order = (notice.get("args") or {}).get("playerList") or []
        if order:
            player_id = _player_id(order[0])
            if player_id in player_index:
                return player_index[player_id]
    return 0


def _collect_final_scoring(
    notifications: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    tiles: list[dict[str, Any]] = []
    seen: set[int] = set()
    for notice in notifications:
        if notice.get("type") != "notifyScore":
            continue
        args = notice.get("args") or {}
        if not isinstance(args, dict):
            continue
        description = re.sub(r"\s+", " ", str(args.get("desc") or "").strip().lower())
        tile = BGA_FINAL_SCORING.get(description)
        if tile is None or tile["id"] in seen:
            continue
        seen.add(tile["id"])
        tiles.append(copy.deepcopy(tile))
    return tiles


def _initial_boosters(board: dict[str, Any]) -> dict[int, int]:
    values = board.get("availBoosters")
    if not isinstance(values, list):
        return {}
    return {
        booster - 1: -1
        for value in values
        for booster in [_as_int(value, 0)]
        if 1 <= booster <= len(BOOSTER_LABELS)
    }


def _initial_round_scoring(board: dict[str, Any]) -> list[dict[str, Any]]:
    values = board.get("roundBonus")
    if not isinstance(values, list):
        return []
    if values and _as_int(values[0], -1) == 0:
        values = values[1:]
    result: list[dict[str, Any]] = []
    for round_index, value in enumerate(values[:6]):
        tile = _as_int(value, 0) - 1
        if not 0 <= tile < len(ROUND_SCORING_TILES):
            continue
        spec = ROUND_SCORING_TILES[tile]
        result.append(
            {
                "round": round_index + 1,
                "id": tile,
                "key": spec.key,
                "label": spec.label,
                "points": spec.points,
            }
        )
    return result


def _initial_final_scoring(board: dict[str, Any]) -> list[dict[str, Any]]:
    values = board.get("endGameBonus")
    if not isinstance(values, list):
        return []
    result: list[dict[str, Any]] = []
    for value in values[:2]:
        tile = _as_int(value, 0) - 1
        if not 0 <= tile < len(FINAL_SCORING_TILES):
            continue
        spec = FINAL_SCORING_TILES[tile]
        result.append({"id": tile, "key": spec.key, "label": spec.label})
    return result


def _initial_standard_tech(board: dict[str, Any]) -> list[dict[str, Any]]:
    values = board.get("techs")
    if not isinstance(values, list):
        return []
    result: list[dict[str, Any]] = []
    for position, value in enumerate(values[:9]):
        tile = _as_int(value, 0) - 1
        if not 0 <= tile < len(STANDARD_TECH_TILES):
            continue
        spec = STANDARD_TECH_TILES[tile]
        result.append(
            {
                "space": position,
                "track": Track(position).name.lower() if position < TRACK_COUNT else None,
                "id": tile,
                "key": spec.key,
                "label": spec.label,
            }
        )
    return result


def _initial_advanced_tech(board: dict[str, Any]) -> list[dict[str, Any]]:
    values = board.get("advTechs")
    if not isinstance(values, list):
        return []
    result: list[dict[str, Any]] = []
    for position, value in enumerate(values[:TRACK_COUNT]):
        tile = _as_int(value, 0) - 10
        if not 0 <= tile < len(ADVANCED_TECH_TILES):
            continue
        spec = ADVANCED_TECH_TILES[tile]
        result.append(
            {
                "track": Track(position).name.lower(),
                "id": tile,
                "key": spec.key,
                "label": spec.label,
            }
        )
    return result


def _initial_terraforming_federation(
    board: dict[str, Any],
) -> dict[str, Any] | None:
    tile = _as_int(board.get("bonusFedToken"), 0) - 1
    if not 0 <= tile < len(FEDERATION_TILES):
        return None
    spec = FEDERATION_TILES[tile]
    return {"id": tile, "key": spec.key, "label": spec.label}


def _initial_federation_supply(board: dict[str, Any]) -> list[int]:
    values = board.get("availFedTokens")
    if not isinstance(values, list):
        return []
    return [
        max(0, _as_int(value, 0))
        for value in values[1 : 1 + len(FEDERATION_TILES)]
    ]


def _action_label(notifications: list[dict[str, Any]]) -> tuple[str, str]:
    priorities = (
        "notifyPlaceStartingBldg",
        "notifyChooseBoosterTile",
        "notifyBuild",
        "notifyStartGaia",
        "notifyUpgrade",
        "notifyFormFederation",
        "notifyResearch",
        "notifyGainTech",
        "notifyConvert",
        "notifyAction",
        "notifyPass",
        "notifyChooseRace",
        "notifyTakeFedToken",
        "notifyRoundEnd",
        "notifyGaiaDone",
        "notifyIncome",
        "notifyScore",
    )
    by_type = {str(notice.get("type")): notice for notice in notifications}
    notice = next((by_type[item] for item in priorities if item in by_type), None)
    if notice is None:
        return "BGA 状态同步", "bga_state"
    notice_type = str(notice.get("type"))
    args = notice.get("args") or {}
    name = str(args.get("player_name") or "")
    prefix = f"{name} · " if name else ""
    building_id = _as_int(args.get("buildingId"), 0)
    building_names = {
        2: "盖亚塑形者",
        3: "空间站",
        4: "矿场",
        5: "贸易站",
        6: "研究所",
        7: "行星研究院",
        8: "知识学院",
        9: "Q.I.C. 学院",
    }
    pay = _resource_tags(str(args.get("payStr") or ""))
    gain = _resource_tags(str(args.get("gainStr") or ""))
    suffix = f" · 支付 {pay}" if pay else ""
    if gain:
        suffix += f" · 获得 {gain}"
    pass_vp = _as_int(args.get("vp"), 0)
    pass_label = "过轮" + (f" · 计分 {_signed(pass_vp)} VP" if pass_vp else "")
    labels = {
        "notifyPlaceStartingBldg": (f"放置起始{building_names.get(building_id, '建筑')}", "starting_building"),
        "notifyChooseBoosterTile": (f"选择助推板块 #{_as_int(args.get('boosterId'), 0)}", "choose_booster"),
        "notifyBuild": (f"建造{building_names.get(building_id, '建筑')}{suffix}", "build"),
        "notifyStartGaia": (f"启动盖亚计划{suffix}", "start_gaia"),
        "notifyUpgrade": (f"升级为{building_names.get(building_id, '建筑')}{suffix}", "upgrade"),
        "notifyFormFederation": (f"组建联邦{suffix}", "federation"),
        "notifyResearch": (
            f"推进{_track_label(args.get('whichResearch'))}科研",
            "research",
        ),
        "notifyGainTech": (f"获得科技板块 #{_as_int(args.get('techId'), 0)}", "technology"),
        "notifyConvert": (f"自由兑换{suffix}", "convert"),
        "notifyAction": (f"执行行动 #{_as_int(args.get('actionId'), 0)}", "special_action"),
        "notifyPass": (pass_label, "pass"),
        "notifyChooseRace": (f"选择{_faction_label(args.get('raceId'))}", "choose_faction"),
        "notifyTakeFedToken": (f"获得联邦板块 #{_as_int(args.get('fedTokenId'), 0)}", "federation_tile"),
        "notifyRoundEnd": (f"第 {_as_int(args.get('roundNum'), 0)} 轮结束", "round_end"),
        "notifyGaiaDone": ("盖亚阶段完成", "gaia_phase"),
        "notifyIncome": ("更新回合收入", "income"),
        "notifyScore": (f"计分 {_signed(_as_int(args.get('vp'), 0))} VP", "score"),
    }
    label, kind = labels[notice_type]
    return prefix + label, kind


def _notification_components(notifications: list[dict[str, Any]]) -> list[dict[str, str]]:
    ignored = {"gameStateChange", "gameStateMultipleActiveUpdate", "updateReflexionTime"}
    components: list[dict[str, str]] = []
    for notice in notifications:
        notice_type = str(notice.get("type") or "")
        if notice_type in ignored:
            continue
        label, _kind = _action_label([notice])
        components.append({"code": notice_type, "label": label})
    return components


def _vp_events(
    notifications: list[dict[str, Any]],
    player_index: dict[int, int],
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    final_scoring_started = False
    unlabeled_final_scores: dict[int, int] = defaultdict(int)
    for notice in notifications:
        notice_type = str(notice.get("type") or "")
        if notice_type not in ("notifyScore", "notifyPass"):
            continue
        args = notice.get("args") or {}
        if not isinstance(args, dict) or "vp" not in args:
            continue
        bga_player_id = _player_id(args.get("playerId"))
        if bga_player_id not in player_index:
            continue
        description = str(args.get("desc") or "").strip()
        normalized_description = re.sub(r"\s+", " ", description.lower())
        if normalized_description in BGA_FINAL_SCORING:
            final_scoring_started = True
        if description:
            reason = description
        elif notice_type == "notifyPass":
            reason = "过轮计分"
        elif final_scoring_started:
            occurrence = unlabeled_final_scores[bga_player_id]
            reason = "科研轨终局计分" if occurrence == 0 else "剩余资源计分"
            unlabeled_final_scores[bga_player_id] += 1
        else:
            reason = "BGA 计分"
        events.append(
            {
                "player": player_index[bga_player_id],
                "bga_player_id": bga_player_id,
                "delta": _as_int(args.get("vp"), 0),
                "source": notice_type,
                "reason": reason,
            }
        )
    return events


def _compact_notification(notice: dict[str, Any]) -> dict[str, Any]:
    args = notice.get("args") or {}
    if not isinstance(args, dict):
        args = {}
    omitted = {"map", "player", "players", "reflexion", "initialprivate", "options"}
    return {
        "type": notice.get("type"),
        "log": notice.get("log"),
        "args": {key: value for key, value in args.items() if key not in omitted},
    }


def _normalize_replay_address(address: str) -> tuple[str, int, str]:
    value = address.strip()
    if not value:
        raise BgaReplayError("请输入 BGA 复盘地址")
    parsed = _validate_bga_url(value)
    path = parsed.path.rstrip("/") or "/"
    if path == "/gamereview":
        kind = "review"
    elif path.startswith("/archive/replay/"):
        kind = "replay"
    else:
        raise BgaReplayError("仅支持 BGA 的 gamereview 或 archive/replay 地址")
    table_value = parse_qs(parsed.query).get("table", [""])[0]
    try:
        table_id = int(table_value)
    except ValueError as error:
        raise BgaReplayError("复盘地址缺少有效的 table 编号") from error
    if table_id <= 0:
        raise BgaReplayError("复盘地址缺少有效的 table 编号")
    return value, table_id, kind


def _validate_bga_url(url: str):
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    is_bga_host = host == BGA_ROOT_DOMAIN or host.endswith(f".{BGA_ROOT_DOMAIN}")
    if parsed.scheme != "https" or not is_bga_host:
        raise BgaReplayError("只允许访问 boardgamearena.com 的 HTTPS 地址")
    if parsed.username or parsed.password:
        raise BgaReplayError("复盘地址不能包含账号信息")
    return parsed


def _serialize_cookie(cookie: Cookie) -> dict[str, Any]:
    return {
        "version": cookie.version,
        "name": cookie.name,
        "value": cookie.value,
        "port": cookie.port,
        "port_specified": cookie.port_specified,
        "domain": cookie.domain,
        "domain_specified": cookie.domain_specified,
        "domain_initial_dot": cookie.domain_initial_dot,
        "path": cookie.path,
        "path_specified": cookie.path_specified,
        "secure": cookie.secure,
        "expires": cookie.expires,
        "discard": cookie.discard,
        "comment": cookie.comment,
        "comment_url": cookie.comment_url,
        "rest": dict(getattr(cookie, "_rest", {})),
        "rfc2109": cookie.rfc2109,
    }


def _restore_cookie(payload: dict[str, Any]) -> Cookie | None:
    if not isinstance(payload, dict):
        return None
    domain = str(payload.get("domain") or "").lstrip(".").lower()
    if domain != BGA_ROOT_DOMAIN and not domain.endswith(f".{BGA_ROOT_DOMAIN}"):
        return None
    name = str(payload.get("name") or "")
    if not name:
        return None
    return Cookie(
        version=_as_int(payload.get("version"), 0),
        name=name,
        value=str(payload.get("value") or ""),
        port=payload.get("port"),
        port_specified=bool(payload.get("port_specified")),
        domain=str(payload.get("domain") or ""),
        domain_specified=bool(payload.get("domain_specified")),
        domain_initial_dot=bool(payload.get("domain_initial_dot")),
        path=str(payload.get("path") or "/"),
        path_specified=bool(payload.get("path_specified", True)),
        secure=bool(payload.get("secure", True)),
        expires=(
            _as_int(payload.get("expires"), 0)
            if payload.get("expires") is not None
            else None
        ),
        discard=bool(payload.get("discard")),
        comment=payload.get("comment"),
        comment_url=payload.get("comment_url"),
        rest=payload.get("rest") if isinstance(payload.get("rest"), dict) else {},
        rfc2109=bool(payload.get("rfc2109")),
    )


def _protect_session(source: bytes) -> bytes:
    return _windows_dpapi(source, protect=True)


def _unprotect_session(source: bytes) -> str:
    return _windows_dpapi(source, protect=False).decode("utf-8")


def _windows_dpapi(source: bytes, *, protect: bool) -> bytes:
    if os.name != "nt":
        raise BgaSessionError("当前系统不支持 Windows DPAPI，无法安全保存 BGA 会话")
    import ctypes
    from ctypes import wintypes

    class DataBlob(ctypes.Structure):
        _fields_ = [
            ("size", wintypes.DWORD),
            ("data", ctypes.POINTER(ctypes.c_ubyte)),
        ]

    buffer = (ctypes.c_ubyte * len(source)).from_buffer_copy(source)
    source_blob = DataBlob(len(source), buffer)
    result_blob = DataBlob()
    crypt32 = ctypes.windll.crypt32
    kernel32 = ctypes.windll.kernel32
    flags = 0x1  # CRYPTPROTECT_UI_FORBIDDEN
    if protect:
        succeeded = crypt32.CryptProtectData(
            ctypes.byref(source_blob),
            "GaiaZero BGA session",
            None,
            None,
            None,
            flags,
            ctypes.byref(result_blob),
        )
    else:
        succeeded = crypt32.CryptUnprotectData(
            ctypes.byref(source_blob),
            None,
            None,
            None,
            None,
            flags,
            ctypes.byref(result_blob),
        )
    if not succeeded:
        raise OSError(ctypes.get_last_error(), "Windows DPAPI operation failed")
    try:
        return ctypes.string_at(result_blob.data, result_blob.size)
    finally:
        kernel32.LocalFree(result_blob.data)


def _extract_json_assignment(source: str, name: str) -> dict[str, Any]:
    position = source.index(name) + len(name)
    position = source.index("=", position) + 1
    value, _end = json.JSONDecoder().raw_decode(source[position:].lstrip())
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be an object")
    return value


def _extract_completesetup_data(source: str) -> dict[str, Any]:
    """Extract the authoritative move-zero state embedded in a BGA archive page."""

    names = ("globalThis.gameui.completesetup", "gameui.completesetup")
    call_position = -1
    for name in names:
        call_position = source.find(name)
        if call_position >= 0:
            break
    if call_position < 0:
        raise KeyError("gameui.completesetup")
    position = source.index("(", call_position) + 1
    quote: str | None = None
    escaped = False
    decoder = json.JSONDecoder()
    while position < len(source):
        character = source[position]
        if quote is not None:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = None
            position += 1
            continue
        if character in ('"', "'", "`"):
            quote = character
            position += 1
            continue
        if character == ")":
            break
        if character != "{":
            position += 1
            continue
        try:
            value, consumed = decoder.raw_decode(source[position:])
        except json.JSONDecodeError:
            position += 1
            continue
        if (
            isinstance(value, dict)
            and isinstance(value.get("board"), dict)
            and isinstance(value.get("map"), dict)
        ):
            return value
        position += max(1, consumed)
    raise KeyError("gameui.completesetup initial state")


def _compact_initial_setup(
    initial_state: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(initial_state, dict):
        return None
    keys = (
        "board",
        "map",
        "players",
        "playerList",
        "playerorder",
        "passOrder",
        "gamestate",
    )
    return {
        "source": "gameui.completesetup",
        **{
            key: copy.deepcopy(initial_state[key])
            for key in keys
            if key in initial_state
        },
    }


def _find_request_token(source: str) -> str | None:
    for pattern in REQUEST_TOKEN_PATTERNS:
        match = pattern.search(source)
        if match:
            return match.group(1)
    return None


def _login_error_message(response: Any) -> str:
    if isinstance(response, dict):
        data = response.get("data")
        if isinstance(data, dict):
            if data.get("captcha") or data.get("challenge"):
                return "BGA 要求完成人机验证，请稍后在浏览器登录后重试"
        error = response.get("error")
        if isinstance(error, str) and error:
            lowered = error.lower()
            if "captcha" in lowered or "robot" in lowered:
                return "BGA 要求完成人机验证，请稍后在浏览器登录后重试"
    return "BGA 登录失败，请检查账号、密码或人机验证状态"


def _packet_epoch(packet: dict[str, Any]) -> int:
    return _as_int(packet.get("time"), 0)


def _packet_timestamp(packet: dict[str, Any]) -> str | None:
    value = _packet_epoch(packet)
    return datetime.fromtimestamp(value, UTC).isoformat() if value > 0 else None


def _player_id(value: Any) -> int | None:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result > 0 else None


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _local_to_bga_terrain(terrain: int) -> int:
    return next(
        (source for source, target in BGA_TERRAIN_TO_LOCAL.items() if target == terrain),
        9,
    )


def _track_label(value: Any) -> str:
    index = _as_int(value, 0) - 1
    return TRACK_NAMES[index] if 0 <= index < len(TRACK_NAMES) else "未知"


def _faction_label(value: Any) -> str:
    index = _as_int(value, 0) - 1
    return FACTION_NAMES[index] if 0 <= index < len(FACTION_NAMES) else "未知种族"


def _resource_tags(value: str) -> str:
    labels = {
        "GOLD": "信用点",
        "ORE": "矿石",
        "KNOWLEDGE": "知识",
        "QIC": "Q.I.C.",
        "POWER": "能量",
        "VP": "VP",
    }
    items = [
        f"{amount} {labels.get(kind, kind)}"
        for kind, amount in re.findall(r"\[([A-Z]+)(-?\d+)\]", value)
    ]
    return "、".join(items)


def _signed(value: int) -> str:
    return f"+{value}" if value > 0 else str(value)
