from __future__ import annotations

import copy
import gzip
import html
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
    7: "academy",
    8: "academy",
    9: "planetary_institute",
}
# BGA orders federation tokens by its image/token ids. GaiaZero keeps the
# canonical rules-engine order used by FEDERATION_TILES.
BGA_FEDERATION_TO_LOCAL = {
    1: 5,  # 12 VP
    2: 2,  # 8 VP + 1 Q.I.C.
    3: 3,  # 8 VP + 2 power tokens
    4: 1,  # 7 VP + 2 ore
    5: 4,  # 7 VP + 6 credits
    6: 0,  # 6 VP + 2 knowledge
}
LOCAL_FEDERATION_TO_BGA = tuple(
    next(bga_id for bga_id, local_id in BGA_FEDERATION_TO_LOCAL.items() if local_id == tile)
    for tile in range(len(FEDERATION_TILES))
)
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

BGA_NOTIFICATION_FUNCTIONS: dict[str, tuple[str, str]] = {
    "gameStateChange": ("状态机切换", "更新当前阶段、行动玩家和可选操作"),
    "gameStateMultipleActiveUpdate": ("并行行动玩家更新", "更新同时需要作出选择的玩家集合"),
    "notifyAction": ("公共或特殊行动", "标记行动工位或板块行动已使用"),
    "notifyAddRaceDraft": ("加入种族竞选池", "把种族加入本局竞选候选"),
    "notifyBanRaceDraft": ("禁用竞选种族", "从本局种族竞选中移除一个种族"),
    "notifyBuild": ("建造", "更新星球、建筑、资源和建筑供应"),
    "notifyChargePower": ("充能", "按 BGA 权威能量状态更新能量碗和脑石"),
    "notifyChooseBoosterTile": ("选择助推板块", "更新助推板块归属"),
    "notifyChooseRace": ("选择种族", "确定种族并载入初始资源、科研和 VP"),
    "notifyConvert": ("自由兑换", "更新自由兑换或盖亚区兑换后的资源"),
    "notifyFormFederation": ("组建联邦", "标记联邦建筑并放置卫星"),
    "notifyGaiaDone": ("盖亚阶段结束", "结算盖亚阶段并进入收入或行动阶段"),
    "notifyGaiaformed": ("盖亚化完成", "把对应的超空间星球翻为盖亚星球"),
    "notifyGainResource": ("获得或支付资源", "应用资源及可能包含的即时 VP 变化"),
    "notifyGainTech": ("获得科技板块", "获得基础或高级科技、覆盖基础科技并消耗未使用联邦片"),
    "notifyGeneric": ("规则消息", "记录竞拍、全局补分或不属于专用通知的规则结果"),
    "notifyIncome": ("回合收入", "更新当前建筑与科技产生的回合收入明细"),
    "notifyPass": ("过轮", "结算助推/高级科技过轮 VP，选择下轮助推并更新顺位"),
    "notifyPlaceStartingBldg": ("放置起始建筑", "更新蛇形起始建筑摆放"),
    "notifyPlayerOrder": ("玩家顺位", "更新首家与行动顺序"),
    "notifyRace6Swap": ("Ambas 建筑交换", "交换一个矿场与行星研究院的位置"),
    "notifyResearch": ("推进科研", "更新科研轨、知识开销和未使用联邦片数量"),
    "notifyRoundEnd": ("大轮结束", "推进轮次或进入终局结算"),
    "notifyScore": ("计分", "应用轮次、科技、种族、被动充能或终局 VP"),
    "notifySetOptions": ("玩家选项", "更新 BGA 自动确认等玩家偏好，不改变规则资源"),
    "notifyStartGaia": ("启动盖亚计划", "放置盖亚塑形者并结算航程 QIC 与盖亚能量"),
    "notifyTakeFedToken": ("获得联邦板块", "更新玩家联邦板块与公共供应"),
    "notifyUpgrade": ("升级建筑", "替换建筑并应用信用点、矿石开销"),
    "simpleNode": ("流程标记", "标记复盘流程节点，包括游戏结束"),
    "updateReflexionTime": ("计时器更新", "仅更新 BGA 思考时间，不改变游戏规则状态"),
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
        href = (
            attributes.get("href")
            or attributes.get("data-href")
            or attributes.get("data-url")
            or attributes.get("data-replay-url")
            or ""
        )
        if not href and "choosePlayerLink" in classes:
            href = attributes.get("onclick") or ""
        href = html.unescape(href)
        replay_match = re.search(r"((?:https?://[^\"'<>\s]+)?/archive/replay/[^\"'<>\s]+)", href)
        if replay_match:
            href = replay_match.group(1)
        if self._entry_depth and "playername" in classes:
            values = parse_qs(urlparse(href).query)
            self._current_player_id = _player_id(values.get("id", [None])[0])
            self._capture_name = True
        parsed = urlparse(href)
        if not parsed.path.startswith("/archive/replay/"):
            return
        values = parse_qs(parsed.query)
        if values.get("table", [""])[0] == str(self.table_id) and href not in self.links:
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


def _extract_replay_links(source: str, table_id: int) -> list[str]:
    """Extract archive links from both rendered anchors and embedded BGA markup."""

    parser = _ReplayLinkParser(table_id)
    parser.feed(source)
    links = list(parser.links)
    pattern = re.compile(
        r"((?:https?://(?:[^/\"'<>\s]+\.)?boardgamearena\.com)?"
        r"/archive/replay/[^\"'<>\s]+)",
        re.IGNORECASE,
    )
    for match in pattern.finditer(html.unescape(source)):
        candidate = match.group(1)
        parsed = urlparse(candidate)
        if parse_qs(parsed.query).get("table", [""])[0] != str(table_id):
            continue
        if candidate not in links:
            links.append(candidate)
    return links


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
        replay_links = _extract_replay_links(review_html, table_id)
        if path_kind == "review":
            if not replay_links:
                lowered = review_html.lower()
                if any(
                    marker in lowered
                    for marker in (
                        "loginuserwithpassword",
                        "bga-login",
                        "please log in",
                        "登录",
                        "connexion",
                    )
                ):
                    raise BgaAuthenticationError(
                        "BGA 当前会话未登录或已失效，请重新输入账号密码后重试"
                    )
                raise BgaReplayError(
                    "该页面没有找到可下载的复盘，账号可能无权查看或对局尚未结束"
                )
            replay_url = urljoin(address, replay_links[0])
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

    requested_address, table_id, address_kind = _normalize_replay_address(replay_address)
    cached_replay_address = (
        _cached_replay_address(history_path, table_id)
        if address_kind == "review"
        else None
    )
    used_cached_replay = False

    def download_record(client: BgaClient) -> dict[str, Any]:
        nonlocal used_cached_replay
        try:
            return client.download(requested_address)
        except BgaReplayError:
            if not cached_replay_address:
                raise
            record = client.download(cached_replay_address)
            bga = record.get("bga")
            if isinstance(bga, dict):
                bga["source_url"] = requested_address
                bga["requested_url"] = requested_address
            used_cached_replay = True
            return record

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
            record = download_record(client)
            used_cached_session = True
        except (BgaAuthenticationError, BgaReplayError):
            client.login(effective_username, effective_password)
            record = download_record(client)
    else:
        client.login(effective_username, effective_password)
        record = download_record(client)

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
        "used_cached_replay_url": used_cached_replay,
    }


def _cached_replay_address(history_path: str | Path, table_id: int) -> str | None:
    """Return a previously verified archive URL for a table, if available."""

    path = Path(history_path).resolve() / f"bga-{table_id}.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    bga = payload.get("bga") if isinstance(payload, dict) else None
    candidate = bga.get("replay_url") if isinstance(bga, dict) else None
    if not isinstance(candidate, str) or not candidate.strip():
        return None
    try:
        normalized, cached_table, kind = _normalize_replay_address(candidate)
    except BgaReplayError:
        return None
    return normalized if kind == "replay" and cached_table == table_id else None


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
        vp_transitions: list[tuple[list[int], list[int]]] = []
        for notice in notifications:
            notice_before = [player["vp"] for player in state.players]
            state.apply_notification(notice)
            notice_after = [player["vp"] for player in state.players]
            vp_transitions.append((notice_before, notice_after))
        vp_events = _vp_events(
            notifications,
            state.player_index,
            transitions=vp_transitions,
        )
        vp_after = [player["vp"] for player in state.players]
        event_deltas = [0] * len(state.players)
        for event in vp_events:
            event_deltas[event["player"]] += event["delta"]
        state_deltas = [after - before for before, after in zip(vp_before, vp_after, strict=True)]
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
                "audit": {
                    "event_deltas": event_deltas,
                    "state_deltas": state_deltas,
                    "matches_state": event_deltas == state_deltas,
                },
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
    observed_notification_types = sorted(
        {
            str(notice.get("type") or "")
            for packet in raw_packets
            for notice in packet.get("data", [])
            if isinstance(notice, dict) and notice.get("type")
        }
    )
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
            "notification_catalog": [
                {
                    "code": code,
                    "function": function,
                    "state_effect": state_effect,
                }
                for code, (function, state_effect) in BGA_NOTIFICATION_FUNCTIONS.items()
            ],
            "notification_coverage": {
                "observed": observed_notification_types,
                "unknown": [
                    code
                    for code in observed_notification_types
                    if code not in BGA_NOTIFICATION_FUNCTIONS
                ],
            },
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
        # Future choose-race payloads are used to recover faction/resources, but
        # their score already contains the auction result. Start the replay at
        # the rules-defined 10 VP so the auction settlement remains visible.
        for player in self.players:
            player["vp"] = 10

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
        self.board_space_ids = {
            coordinate: index
            for index, coordinate in enumerate(
                sorted(first_map, key=lambda item: (item[1], item[0]))
            )
        }
        self.planets: dict[tuple[int, int], dict[str, Any]] = {}
        self.satellites: dict[tuple[int, int], set[int]] = {}
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
        has_player_payload = player is not None and isinstance(payload, dict)
        if has_player_payload:
            _apply_player_payload(player, payload)
        players_payload = args.get("players")
        if isinstance(players_payload, dict):
            for source_id, source_player in players_payload.items():
                target = self._player(_player_id(source_id))
                if target is not None and isinstance(source_player, dict):
                    _apply_player_payload(target, source_player)
                    if target is player:
                        has_player_payload = True

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
            if _as_int(args.get("fedTokenId"), 0) > 0 and not has_player_payload:
                player["federation_keys"] = max(
                    0, player["federation_keys"] - 1
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
            covered = _as_int(args.get("coverupTechId"), 0) - 1
            if 0 <= covered < 9 and covered not in player["covered_tech_tiles"]:
                player["covered_tech_tiles"].append(covered)
            player["qic"] = max(0, player["qic"] - _as_int(args.get("qicCost"), 0))
            if _as_int(args.get("fedTokenId"), 0) > 0 and not has_player_payload:
                player["federation_keys"] = max(
                    0, player["federation_keys"] - 1
                )
        elif notice_type == "notifyTakeFedToken" and player is not None:
            bga_federation_tile = _as_int(args.get("fedTokenId"), 0)
            federation_tile = _local_federation_tile(bga_federation_tile)
            is_gleens_tile = bga_federation_tile == 7
            if federation_tile < 0 and not is_gleens_tile:
                return
            if not has_player_payload:
                if is_gleens_tile:
                    player["gleens_federation_tokens"] += 1
                else:
                    player["federation_tiles"].append(federation_tile)
                player["federations"] = (
                    len(player["federation_tiles"])
                    + player["gleens_federation_tokens"]
                )
                if bga_federation_tile != 1:
                    player["federation_keys"] += 1
            if (
                not is_gleens_tile
                and federation_tile < len(self.federation_supply)
                and self.federation_supply[federation_tile] > 0
            ):
                self.federation_supply[federation_tile] -= 1
        elif notice_type == "notifyFormFederation" and player is not None:
            self._mark_federation(args, player)
        elif notice_type == "notifyRace6Swap" and player is not None:
            self._swap_ambas_buildings(args, player)
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
        for item in self.players:
            _refresh_federation_usage(item)
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
            "satellites": [
                {
                    "id": self.board_space_ids[coordinate],
                    "q": coordinate[0],
                    "r": coordinate[1],
                    "owners": sorted(owners),
                }
                for coordinate, owners in sorted(
                    self.satellites.items(),
                    key=lambda item: self.board_space_ids[item[0]],
                )
            ],
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
        self.satellites = {}
        self.space_stations = []
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
                        self._set_satellite(coordinate, owner)
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
                    self._set_satellite(coordinate, owner)
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
            coordinate = (_as_int(item.get("q"), 0), _as_int(item.get("r"), 0))
            self._set_satellite(coordinate, player["id"])

    def _swap_ambas_buildings(
        self,
        args: dict[str, Any],
        player: dict[str, Any],
    ) -> None:
        mine = self.planets.get((_as_int(args.get("mineQ"), 0), _as_int(args.get("mineR"), 0)))
        institute = self.planets.get((_as_int(args.get("piQ"), 0), _as_int(args.get("piR"), 0)))
        if mine is None or institute is None:
            return
        if mine.get("owner") != player["id"] or institute.get("owner") != player["id"]:
            return
        mine["building"] = "planetary_institute"
        mine["building_id"] = 9
        institute["building"] = "mine"
        institute["building_id"] = 4

    def _set_satellite(self, coordinate: tuple[int, int], owner: int) -> None:
        if coordinate not in self.board_space_ids:
            self.board_space_ids[coordinate] = len(self.board_space_ids)
        self.satellites.setdefault(coordinate, set()).add(owner)

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
                if planet.get("building_id") == 7:
                    academy_types[owner][0] += 1
                elif planet.get("building_id") == 8:
                    academy_types[owner][1] += 1
            coexist = _as_int(planet.get("coexisting_mine_owner"), -1)
            if 0 <= coexist < len(self.players):
                counts[coexist]["mine"] += 1
                colonized[coexist].add(planet["terrain"])
            gaiaformer = _as_int(planet.get("gaiaformer"), -1)
            if 0 <= gaiaformer < len(self.players):
                gaiaformers[gaiaformer] += 1
        for index, player in enumerate(self.players):
            player["satellites"] = sum(
                index in owners for owners in self.satellites.values()
            )
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
        "federation_keys": 0,
        "federation_unused": 0,
        "federation_used": 0,
        "gleens_federation_tokens": 0,
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
            local_tile
            for tile in federation_tiles
            if isinstance(tile, dict)
            for local_tile in [_local_federation_tile(tile.get("fedTokenId"))]
            if local_tile >= 0
        ]
        player["gleens_federation_tokens"] = sum(
            1
            for tile in federation_tiles
            if isinstance(tile, dict) and _as_int(tile.get("fedTokenId"), 0) == 7
        )
        player["federations"] = (
            len(player["federation_tiles"])
            + player["gleens_federation_tokens"]
        )
        player["federation_keys"] = sum(
            1
            for tile in federation_tiles
            if isinstance(tile, dict) and _as_int(tile.get("isGreen"), 0) > 0
        )
        _refresh_federation_usage(player)


def _refresh_federation_usage(player: dict[str, Any]) -> None:
    total = max(0, _as_int(player.get("federations"), 0))
    unused = max(0, min(total, _as_int(player.get("federation_keys"), 0)))
    player["federation_keys"] = unused
    player["federation_unused"] = unused
    player["federation_used"] = total - unused


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
    tile = _local_federation_tile(board.get("bonusFedToken"))
    if not 0 <= tile < len(FEDERATION_TILES):
        return None
    spec = FEDERATION_TILES[tile]
    return {"id": tile, "key": spec.key, "label": spec.label}


def _initial_federation_supply(board: dict[str, Any]) -> list[int]:
    values = board.get("availFedTokens")
    if not isinstance(values, list):
        return []
    supply = [0] * len(FEDERATION_TILES)
    for bga_id in range(1, len(FEDERATION_TILES) + 1):
        local_id = _local_federation_tile(bga_id)
        if local_id >= 0 and bga_id < len(values):
            supply[local_id] = max(0, _as_int(values[bga_id], 0))
    return supply


def _action_label(notifications: list[dict[str, Any]]) -> tuple[str, str]:
    priorities = (
        "notifyPlaceStartingBldg",
        "notifyChooseBoosterTile",
        "notifyBuild",
        "notifyStartGaia",
        "notifyUpgrade",
        "notifyRace6Swap",
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
        "notifyGaiaformed",
        "notifyGainResource",
        "notifyChargePower",
        "notifyIncome",
        "notifyScore",
        "notifyBanRaceDraft",
        "notifyAddRaceDraft",
        "notifyPlayerOrder",
        "notifyGeneric",
        "notifySetOptions",
        "simpleNode",
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
        7: "知识学院",
        8: "Q.I.C. 学院",
        9: "行星研究院",
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
        "notifyRace6Swap": ("交换矿场与行星研究院", "faction_action"),
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
        "notifyGaiaformed": ("完成盖亚化", "gaia_phase"),
        "notifyGainResource": (
            (f"资源结算{suffix}" if suffix else "资源结算"),
            "resource",
        ),
        "notifyChargePower": ("充能并更新能量碗", "power"),
        "notifyIncome": ("更新回合收入", "income"),
        "notifyScore": (f"计分 {_signed(_as_int(args.get('vp'), 0))} VP", "score"),
        "notifyBanRaceDraft": (f"禁用{_faction_label(args.get('raceId'))}", "faction_draft"),
        "notifyAddRaceDraft": (f"加入{_faction_label(args.get('raceId'))}竞选", "faction_draft"),
        "notifyPlayerOrder": ("更新玩家顺位", "turn_order"),
        "notifyGeneric": ("规则消息", "bga_rule"),
        "notifySetOptions": ("更新玩家选项", "bga_option"),
        "simpleNode": ("流程节点", "bga_state"),
    }
    label, kind = labels.get(
        notice_type,
        (
            BGA_NOTIFICATION_FUNCTIONS.get(
                notice_type,
                (f"未识别通知 {notice_type}", "没有对应的本地复盘语义"),
            )[0],
            "bga_state",
        ),
    )
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
    *,
    transitions: list[tuple[list[int], list[int]]] | None = None,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    final_scoring_started = False
    unlabeled_final_scores: dict[int, int] = defaultdict(int)
    transitions = transitions or [([], []) for _notice in notifications]
    name_to_player = {
        str(args.get("player_name")): player_index[player_id]
        for notice in notifications
        for args in [notice.get("args") or {}]
        if isinstance(args, dict)
        for player_id in [_player_id(args.get("playerId"))]
        if player_id in player_index and args.get("player_name")
    }
    has_auction_settlement = any(
        notice.get("type") == "notifyGeneric"
        and (
            "wins the auction" in str(notice.get("log") or "").lower()
            or "all players receive" in str(notice.get("log") or "").lower()
        )
        for notice in notifications
    )
    for notice_index, notice in enumerate(notifications):
        notice_type = str(notice.get("type") or "")
        args = notice.get("args") or {}
        if not isinstance(args, dict):
            continue
        description = str(args.get("desc") or "").strip()
        normalized_description = re.sub(r"\s+", " ", description.lower())
        if normalized_description in BGA_FINAL_SCORING:
            final_scoring_started = True
        before, after = transitions[notice_index]
        changed_players = [
            player
            for player in range(min(len(before), len(after)))
            if before[player] != after[player]
        ]
        explicit_player_id = _player_id(args.get("playerId"))
        explicit_player = player_index.get(explicit_player_id)
        if notice_type == "notifyChooseRace" and has_auction_settlement:
            changed_players = []
        explicit_vp = notice_type in ("notifyScore", "notifyPass") and "vp" in args
        if explicit_vp and explicit_player is not None and explicit_player not in changed_players:
            changed_players.append(explicit_player)
        for player in changed_players:
            bga_player_id = next(
                source_id
                for source_id, target_player in player_index.items()
                if target_player == player
            )
            if before and after and before[player] != after[player]:
                delta = after[player] - before[player]
                event_before = before[player]
                event_after = after[player]
            else:
                delta = _as_int(args.get("vp"), 0)
                event_before = before[player] if before else None
                event_after = event_before + delta if event_before is not None else None
            if description:
                tile = BGA_FINAL_SCORING.get(normalized_description)
                reason = (
                    f"终局计分：{tile['label']}"
                    if tile is not None
                    else description
                )
            elif notice_type == "notifyPass":
                reason = "过轮计分（助推板块/高级科技）"
            elif final_scoring_started and notice_type == "notifyScore":
                occurrence = unlabeled_final_scores[bga_player_id]
                reason = "科研轨终局计分" if occurrence == 0 else "剩余资源计分"
                unlabeled_final_scores[bga_player_id] += 1
            else:
                reason = _vp_reason(notice, notifications, notice_index)
            events.append(
                {
                    "player": player,
                    "bga_player_id": bga_player_id,
                    "delta": delta,
                    "before": event_before,
                    "after": event_after,
                    "source": notice_type,
                    "reason": reason,
                }
            )
        if notice_type == "notifyGeneric" and "wins the auction" in str(notice.get("log") or "").lower():
            player = name_to_player.get(str(args.get("player_name") or ""))
            spent = _as_int(args.get("vp"), 0)
            if player is not None and spent:
                bga_player_id = next(
                    source_id
                    for source_id, target_player in player_index.items()
                    if target_player == player
                )
                events.append(
                    {
                        "player": player,
                        "bga_player_id": bga_player_id,
                        "delta": -spent,
                        "before": None,
                        "after": None,
                        "source": notice_type,
                        "reason": "种族竞拍获胜支付",
                    }
                )
        elif notice_type == "notifyGeneric" and "all players receive" in str(notice.get("log") or "").lower():
            gained = _as_int(args.get("vp"), 0)
            for bga_player_id, player in player_index.items():
                events.append(
                    {
                        "player": player,
                        "bga_player_id": bga_player_id,
                        "delta": gained,
                        "before": None,
                        "after": None,
                        "source": notice_type,
                        "reason": "种族竞拍起始 VP 补偿",
                    }
                )
    return events


def _vp_reason(
    notice: dict[str, Any],
    notifications: list[dict[str, Any]],
    notice_index: int,
) -> str:
    notice_type = str(notice.get("type") or "")
    log = str(notice.get("log") or "").lower()
    args = notice.get("args") or {}
    if notice_type == "notifyChooseRace":
        return "种族竞拍与起始 VP 净结算"
    if notice_type == "notifyScore":
        if " loses " in f" {log} ":
            return "接受被动充能扣分"
        if "round bonus" in log:
            return "轮次计分板块"
        if "faction bonus" in log:
            return "种族能力计分"
        if "technology" in log:
            return "科技板块计分"
        if "research" in log:
            return "科研轨终局计分"
        if "resources/power" in log:
            return "剩余资源终局计分"
        return "BGA 显式计分"
    if notice_type == "notifyGainResource":
        if "federation token" in log:
            return "联邦板块即时收益"
        if _as_int(args.get("actionId"), 0) == 6:
            return "Q.I.C. 行星种类公共行动"
        preceding = notifications[: notice_index + 1]
        if any(item.get("type") == "notifyGainTech" for item in preceding):
            return "科技板块即时计分"
        if any(item.get("type") == "notifyResearch" for item in preceding):
            return "科研轨顶即时收益"
        return "资源通知中的 VP 收益"
    return BGA_NOTIFICATION_FUNCTIONS.get(notice_type, ("BGA 状态变化", ""))[0]


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


def _local_federation_tile(value: Any) -> int:
    return BGA_FEDERATION_TO_LOCAL.get(_as_int(value, 0), -1)


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
