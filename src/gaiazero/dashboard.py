from __future__ import annotations

import json
import threading
import uuid
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from time import perf_counter
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np

from gaiazero.bga import (
    BgaAuthenticationError,
    BgaError,
    BgaNetworkError,
    BgaRateLimitError,
    BgaReplayError,
    BgaSessionError,
    BgaSessionStore,
    import_bga_replay,
)
from gaiazero.game import GaiaHeuristicEvaluator, GaiaState
from gaiazero.game.gaia_state import (
    ADVANCED_TECH_ACTION_OFFSET,
    BAL_TAKS_GAIAFORMER_QIC_ACTION,
    BESCODS_RESEARCH_LIMIT,
    BESCODS_RESEARCH_OFFSET,
    BRAINSTONE_ACTION,
    BOOSTER_LABELS,
    BOOSTER_RANGE_ACTION,
    BOOSTER_TERRAFORM_ACTION,
    ADVANCED_TECH_TILES,
    BUILD_OFFSET,
    FACTIONS,
    FEDERATION_TILES,
    FEDERATION_OFFSET,
    IVITS_SPACE_STATION_OFFSET,
    IVITS_SPACE_STATION_LIMIT,
    ITARS_BURN_POWER_ACTION,
    ITARS_GAIA_FINISH_ACTION,
    ITARS_GAIA_TECH_ACTION,
    LOST_PLANET_LIMIT,
    LOST_PLANET_OFFSET,
    LOST_PLANET_SLOT,
    NEVLAS_CREDITS_ACTION,
    NEVLAS_CREDIT_ORE_ACTION,
    NEVLAS_KNOWLEDGE_ACTION,
    NEVLAS_ORE_ACTION,
    NEVLAS_POWER_TO_GAIA_ACTION,
    NEVLAS_QIC_ACTION,
    QIC_ACADEMY_ACTION,
    QIC_FEDERATION_ACTION_OFFSET,
    QIC_PLANET_TYPES_ACTION,
    QIC_TECH_ACTION,
    GAIA_OFFSET,
    MAX_ROUNDS,
    PASS_BOOSTER_OFFSET,
    PASS_FINAL_ACTION,
    POWER_OFFSET,
    RESEARCH_OFFSET,
    ROUND_SCORING_TILES,
    SKIP_TECH_RESEARCH_ACTION,
    STANDARD_TECH_TILES,
    STANDARD_TECH_COUNT,
    STANDARD_TECH_ACTION,
    TECH_OFFSET,
    TERRANS_GAIA_CREDIT_ACTION,
    TERRANS_GAIA_FINISH_ACTION,
    TERRANS_GAIA_KNOWLEDGE_ACTION,
    TERRANS_GAIA_ORE_ACTION,
    TERRANS_GAIA_QIC_ACTION,
    TAKLONS_PASSIVE_AFTER_ACTION,
    TAKLONS_PASSIVE_BEFORE_ACTION,
    UPGRADE_ACADEMY_OFFSET,
    UPGRADE_QIC_ACADEMY_OFFSET,
    UPGRADE_LAB_OFFSET,
    UPGRADE_PI_OFFSET,
    UPGRADE_TRADING_OFFSET,
    Building,
    Terrain,
    Track,
)
from gaiazero.mcts import PUCTSearch, SearchConfig
from gaiazero.model import (
    NetworkEvaluator,
    architecture_for_players,
    load_checkpoint,
)
from gaiazero.telemetry import (
    JsonlTelemetry,
    build_history_index,
    build_local_history_index,
    read_events,
    read_game_trace,
    read_local_game_trace,
    write_local_game,
)

WEB_ROOT = Path(__file__).with_name("web")
ASSETS = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/setup/random": ("index.html", "text/html; charset=utf-8"),
    "/setup/manual": ("index.html", "text/html; charset=utf-8"),
    "/play": ("index.html", "text/html; charset=utf-8"),
    "/import/bga": ("index.html", "text/html; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
}
for number in range(1, 11):
    name = f"sector-{number:02d}-solid.gif"
    ASSETS[f"/assets/sectors/{name}"] = (f"assets/sectors/{name}", "image/gif")
for number in range(5, 8):
    name = f"sector-{number:02d}-outlined.gif"
    ASSETS[f"/assets/sectors/{name}"] = (f"assets/sectors/{name}", "image/gif")
for number in range(1, 10):
    name = f"tech-standard-{number:02d}.jpg"
    ASSETS[f"/assets/tiles/{name}"] = (f"assets/tiles/{name}", "image/jpeg")
for number in range(1, 16):
    name = f"tech-advanced-{number:02d}.jpg"
    ASSETS[f"/assets/tiles/{name}"] = (f"assets/tiles/{name}", "image/jpeg")
for number in range(1, 11):
    name = f"round-scoring-{number:02d}.gif"
    ASSETS[f"/assets/tiles/{name}"] = (f"assets/tiles/{name}", "image/gif")
for number in range(1, 7):
    name = f"final-scoring-{number:02d}.jpg"
    ASSETS[f"/assets/tiles/{name}"] = (f"assets/tiles/{name}", "image/jpeg")
for number in range(1, 11):
    name = f"booster-{number:02d}.jpg"
    ASSETS[f"/assets/tiles/{name}"] = (f"assets/tiles/{name}", "image/jpeg")
for number in range(1, 15):
    board_name = f"player-board-{number:02d}.jpg"
    ASSETS[f"/assets/factions/{board_name}"] = (
        f"assets/factions/{board_name}",
        "image/jpeg",
    )
ASSETS["/assets/boards/research-board.png"] = (
    "assets/boards/research-board.png",
    "image/png",
)


class DashboardServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        address: tuple[str, int],
        metrics_path: str | Path,
        *,
        history_path: str | Path | None = None,
        quiet: bool = False,
    ) -> None:
        self.metrics_path = Path(metrics_path).resolve()
        self.history_path = (
            Path(history_path).resolve()
            if history_path is not None
            else (self.metrics_path.parent / "history").resolve()
        )
        self.bga_session_path = self.history_path / ".bga-session.bin"
        self.quiet = quiet
        self.simulation_lock = threading.Lock()
        self.simulation: dict[str, Any] = {"status": "idle"}
        self.play_lock = threading.Lock()
        self.play_session: dict[str, Any] = {"status": "idle"}
        self.bga_import_lock = threading.Lock()
        super().__init__(address, DashboardRequestHandler)


class DashboardRequestHandler(BaseHTTPRequestHandler):
    server: DashboardServer

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        request = urlparse(self.path)
        if request.path == "/api/events":
            self._serve_events(parse_qs(request.query))
            return
        if request.path == "/api/health":
            self._serve_health()
            return
        if request.path == "/api/history":
            self._serve_history()
            return
        if request.path == "/api/bga/session":
            self._serve_bga_session()
            return
        if request.path == "/api/game":
            self._serve_game(parse_qs(request.query))
            return
        if request.path == "/api/simulation":
            self._send_json(self._simulation_status())
            return
        if request.path == "/api/play":
            with self.server.play_lock:
                self._send_json(_interactive_session_snapshot(self.server.play_session))
            return
        asset = ASSETS.get(request.path)
        if asset is None:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        filename, content_type = asset
        path = WEB_ROOT / filename
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self._send(path.read_bytes(), content_type)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        request = urlparse(self.path)
        if request.path == "/api/bga/import":
            self._handle_bga_import()
            return
        if request.path == "/api/bga/session/clear":
            self._handle_bga_session_clear()
            return
        if request.path.startswith("/api/play/"):
            self._handle_play_request(request.path)
            return
        if request.path not in ("/api/setup/preview", "/api/simulation"):
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            payload = self._read_json_body()
            config = _normalize_manual_config(payload)
            requested_random_setup = config["random_setup"]
            initial = _manual_initial_state(config)
            resolved_random_setup = _resolved_random_setup(initial)
            if requested_random_setup:
                resolved_random_setup.update(requested_random_setup)
                if (
                    "planet_positions" in requested_random_setup
                    and "planet_layout" not in requested_random_setup
                ):
                    resolved_random_setup.pop("planet_layout", None)
            config["random_setup"] = resolved_random_setup
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            self._send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
            return

        if request.path == "/api/setup/preview":
            self._send_json({"config": config, "state": _public_setup_snapshot(initial)})
            return

        with self.server.simulation_lock:
            if self.server.simulation.get("status") == "running":
                self._send_json(
                    {"error": "a manual simulation is already running"},
                    HTTPStatus.CONFLICT,
                )
                return
            run_id = f"manual-{uuid.uuid4().hex[:10]}"
            self.server.simulation = {
                "status": "running",
                "run_id": run_id,
                "move": 0,
                "config": config,
            }
        worker = threading.Thread(
            target=_run_single_simulation,
            args=(self.server, initial, config, run_id),
            daemon=True,
            name=f"gaiazero-{run_id}",
        )
        worker.start()
        self._send_json(self._simulation_status(), HTTPStatus.ACCEPTED)

    def _handle_bga_import(self) -> None:
        if not self.server.bga_import_lock.acquire(blocking=False):
            self._send_json(
                {"error": "已有 BGA 复盘正在下载"},
                HTTPStatus.CONFLICT,
            )
            return
        try:
            payload = self._read_json_body()
            username = payload.get("username")
            password = payload.get("password")
            replay_address = payload.get("replay_address")
            remember = payload.get("remember", True)
            if not isinstance(username, str) or not isinstance(password, str):
                raise TypeError("BGA username and password must be strings")
            if not isinstance(replay_address, str):
                raise TypeError("BGA replay address must be a string")
            if not isinstance(remember, bool):
                raise TypeError("BGA remember must be a boolean")
            result = import_bga_replay(
                username=username,
                password=password,
                replay_address=replay_address,
                history_path=self.server.history_path,
                session_path=self.server.bga_session_path,
                remember=remember,
            )
            self._send_json(result, HTTPStatus.CREATED)
        except BgaRateLimitError as error:
            self._send_json({"error": str(error)}, HTTPStatus.TOO_MANY_REQUESTS)
        except BgaAuthenticationError as error:
            self._send_json({"error": str(error)}, HTTPStatus.UNAUTHORIZED)
        except BgaReplayError as error:
            self._send_json({"error": str(error)}, HTTPStatus.UNPROCESSABLE_ENTITY)
        except BgaNetworkError as error:
            self._send_json({"error": str(error)}, HTTPStatus.BAD_GATEWAY)
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            self._send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
        except BgaError as error:
            self._send_json({"error": str(error)}, HTTPStatus.BAD_GATEWAY)
        finally:
            self.server.bga_import_lock.release()

    def _serve_bga_session(self) -> None:
        try:
            metadata = BgaSessionStore(self.server.bga_session_path).metadata()
            self._send_json(metadata)
        except BgaSessionError as error:
            self._send_json(
                {"saved": False, "error": str(error)},
                HTTPStatus.UNPROCESSABLE_ENTITY,
            )

    def _handle_bga_session_clear(self) -> None:
        BgaSessionStore(self.server.bga_session_path).clear()
        self._send_json({"ok": True, "saved": False})

    def _handle_play_request(self, path: str) -> None:
        try:
            payload = self._read_json_body()
            if path == "/api/play/start":
                config = _normalize_manual_config(payload)
                roles = _normalize_player_roles(payload.get("roles"), config["players"])
                initial = _manual_initial_state(config)
                session = _create_interactive_session(initial, config, roles)
                with self.server.play_lock:
                    if self.server.play_session.get("busy"):
                        self._send_json(
                            {"error": "the current AI move is still running"},
                            HTTPStatus.CONFLICT,
                        )
                        return
                    _save_interactive_session(self.server, session)
                    self.server.play_session = session
                    response = _interactive_session_snapshot(session)
                self._send_json(response, HTTPStatus.CREATED)
                return
            if path == "/api/play/action":
                self._handle_human_action(payload)
                return
            if path == "/api/play/ai":
                self._handle_ai_action()
                return
            if path == "/api/play/undo":
                self._handle_play_undo()
                return
            if path == "/api/play/roles":
                self._handle_role_change(payload)
                return
            self.send_error(HTTPStatus.NOT_FOUND)
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            self._send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)

    def _handle_human_action(self, payload: dict[str, Any]) -> None:
        action = int(payload.get("action", -1))
        with self.server.play_lock:
            session = self.server.play_session
            state = _active_interactive_state(session)
            if session.get("busy"):
                self._send_json({"error": "AI move is running"}, HTTPStatus.CONFLICT)
                return
            if session["roles"][state.current_player] != "human":
                self._send_json(
                    {"error": "the current player is controlled by AI"},
                    HTTPStatus.CONFLICT,
                )
                return
            if action not in state.legal_actions():
                raise ValueError(f"illegal action {action}")
            _apply_interactive_action(session, action, "human")
            _save_interactive_session(self.server, session)
            response = _interactive_session_snapshot(session)
        self._send_json(response)

    def _handle_role_change(self, payload: dict[str, Any]) -> None:
        with self.server.play_lock:
            session = self.server.play_session
            state = session.get("state")
            if not isinstance(state, GaiaState):
                raise ValueError("no interactive game has been started")
            if session.get("busy"):
                self._send_json({"error": "AI move is running"}, HTTPStatus.CONFLICT)
                return
            session["roles"] = _normalize_player_roles(
                payload.get("roles"),
                state.num_players,
            )
            session["revision"] += 1
            _save_interactive_session(self.server, session)
            response = _interactive_session_snapshot(session)
        self._send_json(response)

    def _handle_play_undo(self) -> None:
        with self.server.play_lock:
            session = self.server.play_session
            state = session.get("state")
            if not isinstance(state, GaiaState):
                raise ValueError("no interactive game has been started")
            if session.get("busy"):
                self._send_json({"error": "AI move is running"}, HTTPStatus.CONFLICT)
                return
            undone = _undo_interactive_action(session)
            _save_interactive_session(self.server, session)
            response = {**_interactive_session_snapshot(session), "undone_actions": undone}
        self._send_json(response)

    def _handle_ai_action(self) -> None:
        with self.server.play_lock:
            session = self.server.play_session
            state = _active_interactive_state(session)
            if session.get("busy"):
                self._send_json({"error": "AI move is already running"}, HTTPStatus.CONFLICT)
                return
            if session["roles"][state.current_player] != "ai":
                self._send_json(
                    {"error": "the current player is controlled by a human"},
                    HTTPStatus.CONFLICT,
                )
                return
            session["busy"] = True
            session_id = session["session_id"]
            player = state.current_player
            search = session["searches"][player]

        try:
            result = search.run(state, add_root_noise=False, temperature=0.0)
            action = int(np.argmax(result.policy))
            top_actions = np.argsort(result.policy)[-3:][::-1]
            search_summary = {
                "root_value": result.root_value.tolist(),
                "candidates": [
                    {
                        **_interactive_action_snapshot(state, int(candidate)),
                        "probability": float(result.policy[candidate]),
                        "visits": int(result.visits[candidate]),
                    }
                    for candidate in top_actions
                    if result.policy[candidate] > 0
                ],
            }
        except Exception as error:
            with self.server.play_lock:
                if self.server.play_session.get("session_id") == session_id:
                    self.server.play_session["busy"] = False
                    self.server.play_session["error"] = f"{type(error).__name__}: {error}"
            self._send_json(
                {"error": f"{type(error).__name__}: {error}"},
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            return

        with self.server.play_lock:
            session = self.server.play_session
            if session.get("session_id") != session_id or session.get("state") is not state:
                session["busy"] = False
                self._send_json({"error": "game state changed during AI search"}, HTTPStatus.CONFLICT)
                return
            _apply_interactive_action(session, action, "ai", search_summary)
            session["busy"] = False
            _save_interactive_session(self.server, session)
            response = _interactive_session_snapshot(session)
        self._send_json(response)

    def _serve_events(self, query: dict[str, list[str]]) -> None:
        try:
            after = max(0, int(query.get("after", ["0"])[0]))
            limit = min(5_000, max(1, int(query.get("limit", ["5000"])[0])))
        except ValueError:
            self._send_json({"error": "after and limit must be integers"}, HTTPStatus.BAD_REQUEST)
            return
        events = read_events(self.server.metrics_path, after=after, limit=limit)
        self._send_json(
            {
                "events": events,
                "last_sequence": events[-1]["sequence"] if events else after,
                "source": str(self.server.metrics_path),
                "exists": self.server.metrics_path.exists(),
            }
        )

    def _serve_health(self) -> None:
        path = self.server.metrics_path
        stat = path.stat() if path.exists() else None
        self._send_json(
            {
                "ok": True,
                "source": str(path),
                "exists": stat is not None,
                "size": stat.st_size if stat else 0,
                "modified": stat.st_mtime if stat else None,
            }
        )

    def _serve_history(self) -> None:
        training = build_history_index(self.server.metrics_path)
        for run in training["runs"]:
            run["source"] = "training"
        local = build_local_history_index(self.server.history_path)
        runs = [*training["runs"], *local["runs"]]
        runs.sort(key=lambda run: str(run.get("started_at") or ""))
        self._send_json(
            {
                "runs": runs,
                "latest_sequence": training["latest_sequence"],
                "source": str(self.server.metrics_path),
                "local_source": str(self.server.history_path),
            }
        )

    def _serve_game(self, query: dict[str, list[str]]) -> None:
        run_id = query.get("run_id", [""])[0]
        try:
            iteration = int(query.get("iteration", [""])[0])
            game = int(query.get("game", [""])[0])
        except ValueError:
            self._send_json(
                {"error": "iteration and game must be integers"},
                HTTPStatus.BAD_REQUEST,
            )
            return
        if not run_id:
            self._send_json({"error": "run_id is required"}, HTTPStatus.BAD_REQUEST)
            return
        trace = read_local_game_trace(
            self.server.history_path,
            run_id=run_id,
            iteration=iteration,
            game=game,
        )
        if trace is None:
            trace = read_game_trace(
                self.server.metrics_path,
                run_id=run_id,
                iteration=iteration,
                game=game,
            )
        if trace is None:
            self._send_json({"error": "game not found"}, HTTPStatus.NOT_FOUND)
            return
        self._send_json(trace)

    def _read_json_body(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as error:
            raise ValueError("invalid Content-Length") from error
        if length < 1 or length > 16_384:
            raise ValueError("request body must contain at most 16384 bytes")
        body = self.rfile.read(length).decode("utf-8")
        payload = json.loads(body)
        if not isinstance(payload, dict):
            raise TypeError("request body must be a JSON object")
        return payload

    def _simulation_status(self) -> dict[str, Any]:
        with self.server.simulation_lock:
            return dict(self.server.simulation)

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self._send(body, "application/json; charset=utf-8", status)

    def _send(
        self,
        body: bytes,
        content_type: str,
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; connect-src 'self'; img-src 'self' data:; "
            "style-src 'self'; script-src 'self'",
        )
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, message: str, *args: object) -> None:
        if not self.server.quiet:
            super().log_message(message, *args)


def create_dashboard_server(
    metrics_path: str | Path,
    host: str = "127.0.0.1",
    port: int = 8765,
    *,
    history_path: str | Path | None = None,
    quiet: bool = False,
) -> DashboardServer:
    return DashboardServer(
        (host, port),
        metrics_path,
        history_path=history_path,
        quiet=quiet,
    )


def _normalize_manual_config(payload: dict[str, Any]) -> dict[str, Any]:
    players = int(payload.get("players", 2))
    seed = int(payload.get("seed", 0))
    first_player = int(payload.get("first_player", 0))
    simulations = int(payload.get("simulations", 8))
    factions_value = payload.get("factions")
    if not 2 <= players <= 4:
        raise ValueError("players must be between two and four")
    if not 0 <= seed <= 2_147_483_647:
        raise ValueError("seed must be between 0 and 2147483647")
    if not 0 <= first_player < players:
        raise ValueError("first_player is out of range")
    if not 1 <= simulations <= 128:
        raise ValueError("simulations must be between 1 and 128")
    if not isinstance(factions_value, list):
        raise TypeError("factions must be an array")
    factions = [int(faction) for faction in factions_value]
    if len(factions) != players:
        raise ValueError("one faction is required for each player")
    random_setup = _normalize_random_setup(payload.get("random_setup"))
    return {
        "players": players,
        "seed": seed,
        "first_player": first_player,
        "factions": factions,
        "simulations": simulations,
        "random_setup": random_setup,
    }


def _normalize_random_setup(value: object) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TypeError("random_setup must be an object")
    array_fields = (
        "sector_tiles",
        "sector_rotations",
        "booster_tiles",
        "round_scoring_tiles",
        "final_scoring_tiles",
        "standard_tech_tiles",
        "advanced_tech_tiles",
    )
    allowed_fields = {
        *array_fields,
        "planet_positions",
        "planet_layout",
        "terraforming_federation_tile",
        "map_mode",
    }
    unknown = set(value) - allowed_fields
    if unknown:
        raise ValueError(f"unknown random_setup field: {sorted(unknown)[0]}")
    normalized: dict[str, Any] = {}
    for field in array_fields:
        if field not in value:
            continue
        items = value[field]
        if not isinstance(items, list):
            raise TypeError(f"random_setup.{field} must be an array")
        normalized[field] = [int(item) for item in items]
    if "planet_positions" in value:
        positions = value["planet_positions"]
        if not isinstance(positions, list):
            raise TypeError("random_setup.planet_positions must be an array")
        normalized_positions: list[dict[str, int]] = []
        for position in positions:
            if not isinstance(position, dict):
                raise TypeError("each planet position must be an object")
            if set(position) != {"id", "q", "r"}:
                raise ValueError("each planet position must contain only id, q and r")
            normalized_positions.append({
                "id": int(position["id"]),
                "q": int(position["q"]),
                "r": int(position["r"]),
            })
        normalized["planet_positions"] = normalized_positions
    if "planet_layout" in value:
        layout = value["planet_layout"]
        if not isinstance(layout, list):
            raise TypeError("random_setup.planet_layout must be an array")
        normalized_layout: list[dict[str, int]] = []
        for item in layout:
            if not isinstance(item, dict):
                raise TypeError("each planet layout item must be an object")
            if set(item) != {"id", "q", "r", "source_id"}:
                raise ValueError(
                    "each planet layout item must contain only id, q, r and source_id"
                )
            normalized_layout.append({
                "id": int(item["id"]),
                "q": int(item["q"]),
                "r": int(item["r"]),
                "source_id": int(item["source_id"]),
            })
        normalized["planet_layout"] = normalized_layout
    if "planet_positions" in normalized and "planet_layout" in normalized:
        raise ValueError("planet_positions and planet_layout cannot both be provided")
    if "terraforming_federation_tile" in value:
        normalized["terraforming_federation_tile"] = int(
            value["terraforming_federation_tile"]
        )
    if "map_mode" in value:
        map_mode = str(value["map_mode"])
        if map_mode not in ("bga-random", "manual"):
            raise ValueError("random_setup.map_mode must be 'bga-random' or 'manual'")
        normalized["map_mode"] = map_mode
    if ("sector_tiles" in normalized) != ("sector_rotations" in normalized):
        raise ValueError("sector tiles and rotations must be provided together")
    return normalized


def _manual_initial_state(config: dict[str, Any]) -> GaiaState:
    random_setup = config.get("random_setup") or {}
    tuple_fields = (
        "sector_tiles",
        "sector_rotations",
        "booster_tiles",
        "round_scoring_tiles",
        "final_scoring_tiles",
        "standard_tech_tiles",
        "advanced_tech_tiles",
    )
    overrides = {
        field: tuple(random_setup[field])
        for field in tuple_fields
        if field in random_setup
    }
    if "terraforming_federation_tile" in random_setup:
        overrides["terraforming_federation_tile"] = random_setup[
            "terraforming_federation_tile"
        ]
    if "map_mode" in random_setup:
        overrides["map_mode"] = random_setup["map_mode"]
    if "planet_positions" in random_setup:
        overrides["planet_positions"] = tuple(
            (position["id"], position["q"], position["r"])
            for position in random_setup["planet_positions"]
        )
    if "planet_layout" in random_setup:
        overrides["planet_layout"] = tuple(
            (item["id"], item["q"], item["r"], item["source_id"])
            for item in random_setup["planet_layout"]
        )
    return GaiaState.initial(
        config["players"],
        config["seed"],
        faction_indices=tuple(config["factions"]),
        first_player=config["first_player"],
        **overrides,
    )


def _resolved_random_setup(state: GaiaState) -> dict[str, Any]:
    available_boosters = [
        booster
        for booster, owner in enumerate(state.booster_owner)
        if owner != -2
    ]
    resolved = {
        "map_mode": state.map_mode,
        "sector_tiles": list(state.sector_tiles),
        "sector_rotations": list(state.sector_rotations),
        "booster_tiles": available_boosters,
        "round_scoring_tiles": list(state.round_scoring_tiles),
        "final_scoring_tiles": list(state.final_scoring_tiles),
        "standard_tech_tiles": list(state.standard_tech_tiles),
        "advanced_tech_tiles": list(state.advanced_tech_tiles),
        "terraforming_federation_tile": state.terraforming_federation_tile,
    }
    if state.map_mode == "manual":
        resolved["planet_layout"] = [
            {
                "id": planet,
                "q": state.planet_q[planet],
                "r": state.planet_r[planet],
                "source_id": state.planet_source_ids[planet],
            }
            for planet, active in enumerate(state.active_planets)
            if active
        ]
    return resolved


def _public_setup_snapshot(state: GaiaState) -> dict[str, object]:
    """Return public setup components without resolving player placement choices."""
    snapshot = state.snapshot()
    for planet in snapshot["planets"]:
        planet["owner"] = -1
        planet["building"] = "empty"
        planet["coexisting_mine_owner"] = -1
        planet["coexisting_mine_federated"] = False
        planet["gaiaformer"] = -1
        planet["federated"] = False
    for player in snapshot["players"]:
        player["booster"] = None
        player["gaiaformers_on_board"] = 0
        player["colonized_types"] = 0
        for inventory in player["structures"].values():
            inventory["supply"] += inventory["built"]
            inventory["built"] = 0
    for faction in snapshot["setup"]["factions"]:
        faction.pop("starting_planets", None)
    for booster in snapshot["setup"]["boosters"]:
        booster["owner"] = -1
    snapshot["setup"]["player_choices_resolved"] = False
    return snapshot


def _normalize_player_roles(value: object, players: int) -> list[str]:
    if value is None:
        return ["human", *(["ai"] * (players - 1))]
    if not isinstance(value, list) or len(value) != players:
        raise ValueError("roles must contain one human or ai entry per player")
    roles = [str(role).lower() for role in value]
    if any(role not in ("human", "ai") for role in roles):
        raise ValueError("roles may only contain human or ai")
    return roles


def _interactive_action_snapshot(state: GaiaState, action: int) -> dict[str, Any]:
    target: int | None = None
    booster: int | None = None
    track: int | None = None
    tech_space: int | None = None
    federation_tile: int | None = None
    power_action: int | None = None
    advanced_action_tile: int | None = None
    space_station_slot: int | None = None
    space_q: int | None = None
    space_r: int | None = None
    hadsch_credit_actions = state._hadsch_hallas_credit_actions(state.player_to_move)
    if (
        action == SKIP_TECH_RESEARCH_ACTION
        and state.pending_research_optional
    ):
        kind = "skip_tech_research"
    elif action == BRAINSTONE_ACTION:
        kind = "brainstone"
    elif action == BAL_TAKS_GAIAFORMER_QIC_ACTION:
        kind = "bal_taks_gaiaformer_qic"
    elif BESCODS_RESEARCH_OFFSET <= action < BESCODS_RESEARCH_LIMIT:
        kind = "bescods_research"
        track = action - BESCODS_RESEARCH_OFFSET
    elif action == ITARS_BURN_POWER_ACTION:
        kind = "itars_burn_power"
    elif action == ITARS_GAIA_TECH_ACTION:
        kind = "itars_gaia_tech"
    elif action == ITARS_GAIA_FINISH_ACTION:
        kind = "itars_gaia_finish"
    elif action == NEVLAS_POWER_TO_GAIA_ACTION:
        kind = "nevlas_power_to_gaia"
    elif action == NEVLAS_CREDITS_ACTION:
        kind = "nevlas_convert_credits"
    elif action == NEVLAS_CREDIT_ORE_ACTION:
        kind = "nevlas_convert_credit_ore"
    elif action == NEVLAS_ORE_ACTION:
        kind = "nevlas_convert_ore"
    elif action == NEVLAS_QIC_ACTION:
        kind = "nevlas_convert_qic"
    elif action == NEVLAS_KNOWLEDGE_ACTION:
        kind = "nevlas_convert_knowledge"
    elif action == TAKLONS_PASSIVE_BEFORE_ACTION:
        kind = "taklons_passive_before"
    elif action == TAKLONS_PASSIVE_AFTER_ACTION:
        kind = "taklons_passive_after"
    elif IVITS_SPACE_STATION_OFFSET <= action < IVITS_SPACE_STATION_LIMIT:
        kind = "ivits_space_station"
        space_station_slot = action - IVITS_SPACE_STATION_OFFSET
        board_spaces = state._board_spaces()
        if space_station_slot < len(board_spaces):
            space_q, space_r = board_spaces[space_station_slot]
    elif LOST_PLANET_OFFSET <= action < LOST_PLANET_LIMIT:
        kind = "lost_planet"
        space_station_slot = action - LOST_PLANET_OFFSET
        board_spaces = state._board_spaces()
        if space_station_slot < len(board_spaces):
            space_q, space_r = board_spaces[space_station_slot]
    elif action == TERRANS_GAIA_CREDIT_ACTION:
        kind = "terrans_gaia_credit"
    elif action == TERRANS_GAIA_ORE_ACTION:
        kind = (
            "hadsch_credit_ore"
            if action in hadsch_credit_actions
            else "terrans_gaia_ore"
        )
    elif action == TERRANS_GAIA_KNOWLEDGE_ACTION:
        kind = (
            "hadsch_credit_knowledge"
            if action in hadsch_credit_actions
            else "terrans_gaia_knowledge"
        )
    elif action == TERRANS_GAIA_QIC_ACTION:
        kind = (
            "hadsch_credit_qic"
            if action in hadsch_credit_actions
            else "terrans_gaia_qic"
        )
    elif action == TERRANS_GAIA_FINISH_ACTION:
        kind = "terrans_gaia_finish"
    elif BUILD_OFFSET <= action < GAIA_OFFSET:
        kind = "starting_placement" if state.is_starting_placement else "build"
        target = action - BUILD_OFFSET
    elif GAIA_OFFSET <= action < UPGRADE_TRADING_OFFSET:
        kind, target = "gaia", action - GAIA_OFFSET
    elif UPGRADE_TRADING_OFFSET <= action < UPGRADE_LAB_OFFSET:
        target = action - UPGRADE_TRADING_OFFSET
        kind = (
            "firaks_downgrade"
            if state._is_firaks_downgrade_action(target)
            else "upgrade_trading"
        )
    elif UPGRADE_LAB_OFFSET <= action < UPGRADE_PI_OFFSET:
        kind, target = "upgrade_lab", action - UPGRADE_LAB_OFFSET
    elif UPGRADE_PI_OFFSET <= action < UPGRADE_ACADEMY_OFFSET:
        target = action - UPGRADE_PI_OFFSET
        kind = "ambas_swap" if state._is_ambas_swap_action(target) else "upgrade_pi"
    elif UPGRADE_ACADEMY_OFFSET <= action < UPGRADE_QIC_ACADEMY_OFFSET:
        kind, target = "upgrade_academy", action - UPGRADE_ACADEMY_OFFSET
    elif UPGRADE_QIC_ACADEMY_OFFSET <= action < RESEARCH_OFFSET:
        kind = (
            "upgrade_credits_academy"
            if FACTIONS[
                state.players[state.player_to_move].faction
            ].qic_academy_credit_action
            else "upgrade_qic_academy"
        )
        target = action - UPGRADE_QIC_ACADEMY_OFFSET
    elif RESEARCH_OFFSET <= action < POWER_OFFSET:
        kind = "research"
        track = action - RESEARCH_OFFSET
    elif POWER_OFFSET <= action < TECH_OFFSET:
        kind = "power"
        power_action = action - POWER_OFFSET
    elif TECH_OFFSET <= action < FEDERATION_OFFSET:
        kind = "technology"
        tech_space = action - TECH_OFFSET
        if tech_space >= STANDARD_TECH_COUNT:
            track = tech_space - STANDARD_TECH_COUNT
        elif tech_space < len(Track):
            track = tech_space
    elif FEDERATION_OFFSET <= action < QIC_ACADEMY_ACTION:
        kind = "federation"
        federation_tile = action - FEDERATION_OFFSET
    elif action == QIC_ACADEMY_ACTION:
        kind = (
            "credits_academy_action"
            if FACTIONS[
                state.players[state.player_to_move].faction
            ].qic_academy_credit_action
            else "qic_academy_action"
        )
    elif action == STANDARD_TECH_ACTION:
        kind = "standard_tech_action"
    elif action == QIC_TECH_ACTION:
        kind = "qic_tech_action"
    elif QIC_FEDERATION_ACTION_OFFSET <= action < QIC_PLANET_TYPES_ACTION:
        kind = "qic_federation_action"
        federation_tile = action - QIC_FEDERATION_ACTION_OFFSET
    elif action == QIC_PLANET_TYPES_ACTION:
        kind = "qic_planet_types_action"
    elif action == BOOSTER_TERRAFORM_ACTION:
        kind = "booster_terraform_action"
    elif action == BOOSTER_RANGE_ACTION:
        kind = "booster_range_action"
    elif ADVANCED_TECH_ACTION_OFFSET <= action < PASS_BOOSTER_OFFSET:
        kind = "advanced_tech_action"
        advanced_action_tile = action - ADVANCED_TECH_ACTION_OFFSET
    elif PASS_BOOSTER_OFFSET <= action < PASS_FINAL_ACTION:
        kind = "select_booster" if state.is_booster_selection else "pass_booster"
        booster = action - PASS_BOOSTER_OFFSET
    elif action == PASS_FINAL_ACTION:
        kind = "pass_final"
    else:
        kind = "other"
    return {
        "id": int(action),
        "label": state.describe_action(action),
        "kind": kind,
        "target": target,
        "booster": booster,
        "track": track,
        "tech_space": tech_space,
        "federation_tile": federation_tile,
        "power_action": power_action,
        "advanced_action_tile": advanced_action_tile,
        "space_station_slot": space_station_slot,
        "space_q": space_q,
        "space_r": space_r,
    }


_LOG_RESOURCE_FIELDS = ("credits", "ore", "knowledge", "qic", "vp")


def _interactive_phase(state: GaiaState) -> str:
    if state.is_starting_placement:
        return "starting_placement"
    if state.is_booster_selection:
        return "booster_selection"
    if state.pending_taklons_charge_player >= 0:
        return "taklons_passive_charge"
    if state.pending_gaia_conversion_player >= 0:
        return "gaia_conversion"
    if state.pending_lost_planet_player >= 0:
        return "lost_planet_placement"
    if state.pending_itars_gaia_player >= 0:
        return "itars_gaia_technology"
    if state.is_terminal:
        return "terminal"
    return "round"


def _component_ref(
    kind: str,
    component_id: int,
    label: str,
    code: str,
    *,
    relation: str = "uses",
) -> dict[str, Any]:
    return {
        "kind": kind,
        "id": int(component_id),
        "code": code,
        "label": label,
        "relation": relation,
    }


def _interactive_action_components(
    state: GaiaState,
    action: dict[str, Any],
    player: int,
) -> list[dict[str, Any]]:
    components: list[dict[str, Any]] = []
    if action["kind"] == "brainstone" or (
        action["kind"] == "power" and state.brainstone_selected
    ):
        components.append(
            _component_ref(
                "brainstone",
                0,
                "Brainstone",
                "TAK-BRAINSTONE",
                relation=("selected" if action["kind"] == "brainstone" else "spent"),
            )
        )
    if action["kind"] == "lost_planet":
        components.append(
            _component_ref(
                "research_track",
                int(Track.NAVIGATION),
                "Navigation level 5: Lost Planet",
                "RES-NAV-L5",
                relation="gained",
            )
        )
        components.append(
            _component_ref(
                "planet",
                LOST_PLANET_SLOT,
                "Lost Planet",
                f"P-{LOST_PLANET_SLOT}",
                relation="gained",
            )
        )
    if action["kind"].startswith("terrans_gaia_"):
        components.append(
            _component_ref(
                "faction_ability",
                0,
                "Terrans planetary institute",
                "TER-PI",
                relation="uses",
            )
        )
    if action["kind"].startswith("hadsch_credit_"):
        components.append(
            _component_ref(
                "faction_ability",
                6,
                "Hadsch Hallas planetary institute credit conversion",
                "HAD-PI-CREDIT",
                relation="uses",
            )
        )
    if action["kind"] == "bal_taks_gaiaformer_qic":
        components.append(
            _component_ref(
                "faction_ability",
                9,
                "Bal T'aks Gaiaformer conversion",
                "BAL-GF-QIC",
                relation="uses",
            )
        )
    if action["kind"] == "ambas_swap":
        components.append(
            _component_ref(
                "faction_ability",
                5,
                "Ambas planetary institute swap",
                "AMB-PI-SWAP",
                relation="uses",
            )
        )
    if action["kind"] == "firaks_downgrade":
        components.append(
            _component_ref(
                "faction_ability",
                10,
                "Firaks planetary institute: downgrade a research lab",
                "FIR-PI-DOWNGRADE",
                relation="uses",
            )
        )
    if action["kind"] == "bescods_research":
        components.append(
            _component_ref(
                "faction_ability",
                11,
                "Bescods: advance a tied lowest research area",
                "BES-LOWEST-RESEARCH",
                relation="uses",
            )
        )
    if action["kind"] == "itars_burn_power":
        components.append(
            _component_ref(
                "faction_ability",
                13,
                "Itars: burned power token moves to the Gaia area",
                "ITA-BURN",
                relation="uses",
            )
        )
    if action["kind"] in ("itars_gaia_tech", "itars_gaia_finish") or (
        action["kind"] == "technology"
        and state.pending_itars_gaia_player >= 0
    ):
        components.append(
            _component_ref(
                "faction_ability",
                13,
                "Itars planetary institute: 4 Gaia power for a technology tile",
                "ITA-PI-TECH",
                relation="uses",
            )
        )
    if action["kind"] == "nevlas_power_to_gaia":
        components.append(
            _component_ref(
                "faction_ability",
                12,
                "Nevlas: move power from bowl III to the Gaia area for knowledge",
                "NEV-GAIA-K",
                relation="uses",
            )
        )
    if action["kind"].startswith("nevlas_convert_"):
        components.append(
            _component_ref(
                "faction_ability",
                12,
                "Nevlas planetary institute: improved power conversion",
                "NEV-PI-CONVERT",
                relation="uses",
            )
        )
    if (
        action["kind"] == "power"
        and FACTIONS[state.players[player].faction].name == "Nevlas"
        and state._has_pi(player)
    ):
        components.append(
            _component_ref(
                "faction_ability",
                12,
                "Nevlas planetary institute: each bowl III token counts as 2 power for public power actions",
                "NEV-PI-POWER",
                relation="uses",
            )
        )
    if action["kind"] == "ivits_space_station":
        slot = action.get("space_station_slot")
        q = action.get("space_q")
        r = action.get("space_r")
        components.append(
            _component_ref(
                "space_station",
                int(slot or 0),
                f"Ivits space station at ({q}, {r})",
                f"IVI-SS-{int(slot or 0):03d}",
                relation="placed",
            )
        )
    if action["kind"] in ("taklons_passive_before", "taklons_passive_after"):
        components.append(
            _component_ref(
                "faction_ability",
                4,
                "Taklons planetary institute: passive charge token",
                "TAK-PI",
                relation="gained",
            )
        )
    target = action.get("target")
    if target is not None:
        components.append(
            _component_ref("planet", target, f"Planet {target}", f"P-{target}")
        )
    if (
        action["kind"] == "build"
        and target is not None
        and state._geodens_new_type_knowledge(player, target)
    ):
        components.append(
            _component_ref(
                "faction_ability",
                8,
                "Geodens planetary institute: first mine on this planet type gains 3 knowledge",
                "GEO-PI",
                relation="gained",
            )
        )
    if (
        action["kind"] == "upgrade_pi"
        and FACTIONS[state.players[player].faction].name == "Gleens"
    ):
        components.append(
            _component_ref(
                "gleens_federation",
                0,
                "Gleens federation tile: 2 credits, 1 ore, and 1 knowledge",
                "GLE-FED",
                relation="gained",
            )
        )

    booster = action.get("booster")
    if booster is not None and booster >= 0:
        components.append(
            _component_ref(
                "booster",
                booster,
                BOOSTER_LABELS[booster],
                f"BST-{booster + 1:02d}",
                relation="selected",
            )
        )
    if action["kind"] == "pass_booster":
        returned = state._player_booster(player)
        if returned >= 0:
            components.append(
                _component_ref(
                    "booster",
                    returned,
                    BOOSTER_LABELS[returned],
                    f"BST-{returned + 1:02d}",
                    relation="returned",
                )
            )

    track = action.get("track")
    tech_space = action.get("tech_space")
    if track is not None:
        track_name = Track(track).name.replace("_", " ").title()
        components.append(
            _component_ref("research_track", track, track_name, f"TRK-{track + 1:02d}")
        )
        if (
            track == Track.TERRAFORMING
            and state.players[player].tracks[track] == 4
            and (
                action["kind"] in ("research", "bescods_research")
                or (
                    action["kind"] == "technology"
                    and state.pending_advanced_tech < 0
                    and tech_space is not None
                    and tech_space < len(Track)
                )
            )
        ):
            tile = state.terraforming_federation_tile
            components.append(
                _component_ref(
                    "federation",
                    tile,
                    FEDERATION_TILES[tile].label,
                    f"FED-{tile + 1:02d}",
                    relation="gained",
                )
            )

    if action["kind"] == "technology" and tech_space is not None:
        if state.pending_advanced_tech >= 0:
            tile = state.standard_tech_tiles[tech_space]
            components.append(
                _component_ref(
                    "standard_tech",
                    tile,
                    STANDARD_TECH_TILES[tile].label,
                    f"TEC-S{tile + 1:02d}",
                    relation="covered",
                )
            )
            tile = state.pending_advanced_tech
            components.append(
                _component_ref(
                    "advanced_tech",
                    tile,
                    ADVANCED_TECH_TILES[tile].label,
                    f"TEC-A{tile + 1:02d}",
                    relation="gained",
                )
            )
        elif tech_space < STANDARD_TECH_COUNT:
            tile = state.standard_tech_tiles[tech_space]
            components.append(
                _component_ref(
                    "standard_tech",
                    tile,
                    STANDARD_TECH_TILES[tile].label,
                    f"TEC-S{tile + 1:02d}",
                    relation="gained",
                )
            )
        else:
            tile = state.advanced_tech_tiles[tech_space - STANDARD_TECH_COUNT]
            components.append(
                _component_ref(
                    "advanced_tech",
                    tile,
                    ADVANCED_TECH_TILES[tile].label,
                    f"TEC-A{tile + 1:02d}",
                    relation="selected",
                )
            )

    power_action = action.get("power_action")
    if power_action is not None:
        power_labels = (
            "Gain 3 knowledge",
            "Build a mine with 2 free terraforming steps; pay ore for remaining steps",
            "Gain 2 ore",
            "Gain 7 credits",
            "Gain 2 knowledge",
            "Build a mine with 1 free terraforming step; pay ore for remaining steps",
            "Gain 2 power tokens",
        )
        components.append(
            _component_ref(
                "power_action",
                power_action,
                power_labels[power_action],
                f"PWR-{power_action + 1:02d}",
            )
        )

    if action["kind"] == "federation":
        tile = action.get("federation_tile")
        if tile is None:
            tile = 0
        components.append(
            _component_ref(
                "federation",
                tile,
                FEDERATION_TILES[tile].label,
                f"FED-{tile + 1:02d}",
                relation="gained",
            )
        )
    elif action["kind"] in ("qic_academy_action", "credits_academy_action"):
        faction = FACTIONS[state.players[player].faction]
        label = (
            f"Gain {faction.qic_academy_credit_action} credits"
            if faction.qic_academy_credit_action
            else "Q.I.C. academy: gain 1 Q.I.C."
        )
        components.append(
            _component_ref(
                "faction_action",
                player,
                label,
                f"FAC-{player + 1:02d}",
                relation="used",
            )
        )
    elif action["kind"] == "standard_tech_action":
        components.append(
            _component_ref(
                "standard_tech",
                8,
                STANDARD_TECH_TILES[8].label,
                "TEC-S09",
                relation="used",
            )
        )
    elif action["kind"] == "qic_tech_action":
        components.append(
            _component_ref(
                "qic_action",
                0,
                "Q.I.C. action: take a tech tile",
                "QIC-01",
                relation="used",
            )
        )
    elif action["kind"] == "qic_federation_action":
        tile = action.get("federation_tile", 0)
        if tile == len(FEDERATION_TILES):
            components.append(
                _component_ref(
                    "gleens_federation",
                    0,
                    "Repeat Gleens federation reward",
                    "GLE-FED",
                    relation="repeated",
                )
            )
        else:
            components.append(
                _component_ref(
                    "federation",
                    tile,
                    "Repeat federation reward",
                    f"FED-{tile + 1:02d}",
                    relation="repeated",
                )
            )
    elif action["kind"] == "qic_planet_types_action":
        components.append(
            _component_ref(
                "qic_action",
                2,
                "Q.I.C. action: score planet types",
                "QIC-03",
                relation="used",
            )
        )
    elif action["kind"] == "booster_terraform_action":
        components.append(
            _component_ref(
                "booster",
                0,
                BOOSTER_LABELS[0],
                "BST-01",
                relation="used",
            )
        )
    elif action["kind"] == "booster_range_action":
        components.append(
            _component_ref(
                "booster",
                1,
                BOOSTER_LABELS[1],
                "BST-02",
                relation="used",
            )
        )
    elif action["kind"] == "advanced_tech_action":
        tile = action["advanced_action_tile"]
        components.append(
            _component_ref(
                "advanced_tech",
                tile,
                ADVANCED_TECH_TILES[tile].label,
                f"TEC-A{tile + 1:02d}",
                relation="used",
            )
        )

    if 1 <= state.round_number <= MAX_ROUNDS:
        tile = state.round_scoring_tiles[state.round_number - 1]
        scoring_kind = ROUND_SCORING_TILES[tile].kind
        scored_kinds: set[str] = set()
        if action["kind"] == "lost_planet":
            scored_kinds.add("mine")
        elif action["kind"] == "build" and target is not None:
            terrain = Terrain(state.terrains[target])
            scored_kinds.add("mine")
            if terrain == Terrain.GAIA:
                scored_kinds.add("gaia")
            elif terrain != Terrain.TRANSDIM:
                home = FACTIONS[state.players[player].faction].home
                if state._terrain_steps(home, terrain):
                    scored_kinds.add("terraform")
        elif action["kind"] in ("upgrade_trading", "firaks_downgrade"):
            scored_kinds.add("trading")
        elif action["kind"] in (
            "upgrade_pi",
            "upgrade_academy",
            "upgrade_qic_academy",
            "upgrade_credits_academy",
        ):
            scored_kinds.add("big")
            if (
                action["kind"] == "upgrade_pi"
                and FACTIONS[state.players[player].faction].name == "Gleens"
            ):
                scored_kinds.add("federation")
        elif action["kind"] in ("research", "technology", "bescods_research"):
            scored_kinds.add("research")
            if track == Track.TERRAFORMING and state.players[player].tracks[track] == 4:
                scored_kinds.add("federation")
        elif action["kind"] == "federation":
            scored_kinds.add("federation")
        if scoring_kind in scored_kinds:
            components.append(
                _component_ref(
                    "round_scoring",
                    tile,
                    ROUND_SCORING_TILES[tile].label,
                    f"RND-{tile + 1:02d}",
                    relation="scored",
                )
            )
    return components


def _interactive_action_costs(
    state: GaiaState,
    action: dict[str, Any],
    player: int,
) -> list[dict[str, Any]]:
    kind = action["kind"]
    target = action.get("target")
    costs: dict[str, int] = {}
    if kind == "build" and target is not None:
        credits, ore, qic = state._build_cost(
            player,
            target,
            free_steps=(
                state.pending_power_terraform_steps
                if state.pending_power_terraform_player >= 0
                else 1
                if state.pending_booster_terraform_player >= 0
                else 0
            ),
            range_bonus=3 if state.pending_booster_range_player >= 0 else 0,
        )
        costs.update(credits=credits, ore=ore, qic=qic)
    elif kind == "gaia":
        costs["power_to_gaia"] = state._gaia_cost(state.players[player])
        qic = state._range_qic_cost(
            player,
            target,
            range_bonus=3 if state.pending_booster_range_player >= 0 else 0,
        )
        if qic:
            costs["qic"] = qic
    elif kind == "lost_planet":
        q = action.get("space_q")
        r = action.get("space_r")
        if q is not None and r is not None:
            qic = state._coordinate_range_qic_cost(player, int(q), int(r))
            if qic:
                costs["qic"] = qic
    elif kind == "upgrade_trading" and target is not None:
        costs.update(
            credits=3 if state._has_nearby_opponent(player, target) else 6,
            ore=2,
        )
    elif kind == "upgrade_lab":
        costs.update(credits=5, ore=3)
    elif kind == "upgrade_pi":
        costs.update(credits=6, ore=4)
    elif kind in (
        "upgrade_academy",
        "upgrade_qic_academy",
        "upgrade_credits_academy",
    ):
        costs.update(credits=6, ore=6)
    elif kind == "qic_tech_action":
        costs["qic"] = 4
    elif kind == "qic_federation_action":
        costs["qic"] = 3
    elif kind == "qic_planet_types_action":
        costs["qic"] = 2
    elif kind == "research":
        if state.pending_research_player < 0:
            costs["knowledge"] = 4
    elif kind == "power":
        costs["power"] = state._power_action_cost(player, action["power_action"])
    elif kind == "hadsch_credit_ore":
        costs["credits"] = 3
    elif kind in ("hadsch_credit_knowledge", "hadsch_credit_qic"):
        costs["credits"] = 4
    elif kind == "bal_taks_gaiaformer_qic":
        costs["gaiaformers"] = 1
    elif kind == "itars_gaia_tech":
        costs["gaia_power"] = 4
    elif kind == "nevlas_power_to_gaia":
        costs["power_to_gaia"] = 1
    elif kind == "nevlas_convert_credits":
        costs["power"] = 1
    elif kind in (
        "nevlas_convert_credit_ore",
        "nevlas_convert_qic",
        "nevlas_convert_knowledge",
    ):
        costs["power"] = 2
    elif kind == "nevlas_convert_ore":
        costs["power"] = 3
    elif kind == "terrans_gaia_credit":
        costs["gaia_conversion_power"] = 1
    elif kind == "terrans_gaia_ore":
        costs["gaia_conversion_power"] = 3
    elif kind in ("terrans_gaia_knowledge", "terrans_gaia_qic"):
        costs["gaia_conversion_power"] = 4
    elif kind == "federation":
        plan = state._federation_plan(player)
        if plan is not None and plan[1] > 0:
            if FACTIONS[state.players[player].faction].name == "Ivits":
                costs["qic"] = plan[1]
            else:
                costs["power_tokens"] = plan[1]

    track = action.get("track")
    if (
        track is not None
        and state.players[player].tracks[track] == 4
        and kind in ("research", "technology")
    ):
        costs["federation_key"] = 1
    return [
        {"resource": resource, "amount": int(amount)}
        for resource, amount in costs.items()
        if amount > 0
    ]


def _interactive_player_changes(
    before: GaiaState,
    after: GaiaState,
    player: int,
) -> list[dict[str, Any]]:
    old = before.players[player]
    new = after.players[player]
    changes: list[dict[str, Any]] = []
    old_power = [old.bowl_one, old.bowl_two, old.bowl_three]
    new_power = [new.bowl_one, new.bowl_two, new.bowl_three]
    if old_power != new_power:
        changes.append({"kind": "power", "before": old_power, "after": new_power})
    if old.brainstone_bowl != new.brainstone_bowl:
        changes.append({
            "kind": "brainstone",
            "before": old.brainstone_bowl,
            "after": new.brainstone_bowl,
        })
    if (
        player == before.player_to_move
        and before.brainstone_selected != after.brainstone_selected
    ):
        changes.append({
            "kind": "brainstone_selection",
            "before": before.brainstone_selected,
            "after": after.brainstone_selected,
        })
    if (
        player == before.player_to_move
        and before.pending_gaia_conversion_power
        != after.pending_gaia_conversion_power
    ):
        changes.append({
            "kind": "gaia_conversion_budget",
            "before": before.pending_gaia_conversion_power,
            "after": after.pending_gaia_conversion_power,
        })
    for counter in (
        "gaia_power",
        "gaiaformers",
        "gaiaformers_in_gaia",
        "federation_tokens",
        "federation_keys",
        "gleens_federation_tokens",
        "satellites",
    ):
        old_value = int(getattr(old, counter))
        new_value = int(getattr(new, counter))
        if old_value != new_value:
            changes.append({
                "kind": "counter",
                "counter": counter,
                "before": old_value,
                "after": new_value,
            })
    if old.knowledge_academies != new.knowledge_academies:
        changes.append({
            "kind": "academy",
            "type": "knowledge",
            "before": old.knowledge_academies,
            "after": new.knowledge_academies,
        })
    if old.qic_academies != new.qic_academies:
        changes.append({
            "kind": "academy",
            "type": "qic",
            "before": old.qic_academies,
            "after": new.qic_academies,
        })
    for track, (old_level, new_level) in enumerate(zip(old.tracks, new.tracks, strict=True)):
        if old_level != new_level:
            changes.append({
                "kind": "track",
                "track": track,
                "before": int(old_level),
                "after": int(new_level),
            })
    gained_tech = new.tech_tiles & ~old.tech_tiles
    for tile in range(len(STANDARD_TECH_TILES)):
        if gained_tech & (1 << tile):
            changes.append({"kind": "tech", "id": tile})
    gained_advanced = new.advanced_tech_tiles & ~old.advanced_tech_tiles
    for tile in range(len(ADVANCED_TECH_TILES)):
        if gained_advanced & (1 << tile):
            changes.append({"kind": "advanced_tech", "id": tile})
    old_booster = before._player_booster(player)
    new_booster = after._player_booster(player)
    if old_booster != new_booster:
        changes.append({
            "kind": "booster",
            "before": old_booster,
            "after": new_booster,
        })
    if old.passed != new.passed:
        changes.append({"kind": "passed", "before": old.passed, "after": new.passed})
    return changes


def _interactive_board_changes(
    before: GaiaState,
    after: GaiaState,
) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    for planet in range(len(before.active_planets)):
        if before.owners[planet] != after.owners[planet] or before.buildings[planet] != after.buildings[planet]:
            changes.append({
                "kind": "building",
                "planet": planet,
                "owner_before": before.owners[planet],
                "owner_after": after.owners[planet],
                "building_before": Building(before.buildings[planet]).name.lower(),
                "building_after": Building(after.buildings[planet]).name.lower(),
            })
        if before.coexisting_mine_owner[planet] != after.coexisting_mine_owner[planet]:
            changes.append({
                "kind": "coexisting_mine",
                "planet": planet,
                "owner_before": before.coexisting_mine_owner[planet],
                "owner_after": after.coexisting_mine_owner[planet],
            })
        if (
            before.coexisting_mine_federated[planet]
            != after.coexisting_mine_federated[planet]
        ):
            changes.append({
                "kind": "coexisting_federated",
                "planet": planet,
                "owner": after.coexisting_mine_owner[planet],
                "after": after.coexisting_mine_federated[planet],
            })
        if before.gaiaformer_owner[planet] != after.gaiaformer_owner[planet]:
            changes.append({
                "kind": "gaiaformer",
                "planet": planet,
                "before": before.gaiaformer_owner[planet],
                "after": after.gaiaformer_owner[planet],
            })
        if before.terrains[planet] != after.terrains[planet]:
            changes.append({
                "kind": "terrain",
                "planet": planet,
                "before": before.terrains[planet],
                "after": after.terrains[planet],
            })
    federated = sum(after.federated) - sum(before.federated)
    if federated:
        changes.append({"kind": "federated", "amount": int(federated)})
    return changes


def _interactive_player_effects(
    before: GaiaState,
    after: GaiaState,
    actor: int,
    costs: list[dict[str, Any]],
    *,
    adjustments: dict[tuple[int, str], int] | None = None,
    change_kinds: set[str] | None = None,
) -> list[dict[str, Any]]:
    adjustments = adjustments or {}
    effects: list[dict[str, Any]] = []
    actor_costs = {item["resource"]: item["amount"] for item in costs}
    player_order = (actor, *(player for player in range(before.num_players) if player != actor))
    for player in player_order:
        old = before.players[player]
        new = after.players[player]
        player_costs = list(costs) if player == actor else []
        player_gains: list[dict[str, Any]] = []
        for resource in _LOG_RESOURCE_FIELDS:
            delta = (
                int(getattr(new, resource))
                - int(getattr(old, resource))
                + adjustments.get((player, resource), 0)
            )
            paid = actor_costs.get(resource, 0) if player == actor else 0
            received = delta + paid
            if received > 0:
                player_gains.append({"resource": resource, "amount": received})
            elif received < 0:
                player_costs.append({"resource": resource, "amount": -received})
        changes = _interactive_player_changes(before, after, player)
        if change_kinds is not None:
            changes = [change for change in changes if change["kind"] in change_kinds]
        if player == actor or player_costs or player_gains or changes:
            effects.append({
                "player": player,
                "costs": player_costs,
                "gains": player_gains,
                "changes": changes,
            })
    return effects


def _income_sources(state: GaiaState, player: int) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    booster = state._player_booster(player)
    if booster >= 0:
        sources.append(
            _component_ref(
                "booster",
                booster,
                BOOSTER_LABELS[booster],
                f"BST-{booster + 1:02d}",
                relation="income",
            )
        )
    info = state.players[player]
    for track in (Track.ECONOMY, Track.SCIENCE):
        level = info.tracks[track]
        if level:
            name = Track(track).name.replace("_", " ").title()
            sources.append(
                _component_ref(
                    "research_track",
                    int(track),
                    f"{name} level {level}",
                    f"TRK-{int(track) + 1:02d}",
                    relation="income",
                )
            )
    for tile in (5, 6, 7):
        if state._has_active_standard_tech(info, tile):
            sources.append(
                _component_ref(
                    "standard_tech",
                    tile,
                    STANDARD_TECH_TILES[tile].label,
                    f"TEC-S{tile + 1:02d}",
                    relation="income",
                )
            )
    return sources


def _interactive_action_record(
    before: GaiaState,
    after: GaiaState,
    action: int,
    move: int,
    player: int,
    role: str,
) -> dict[str, Any]:
    summary = _interactive_action_snapshot(before, action)
    costs = _interactive_action_costs(before, summary, player)
    components = _interactive_action_components(before, summary, player)
    round_advanced = (
        after.round_number == before.round_number + 1
        and 1 <= after.round_number <= MAX_ROUNDS
    )
    automatic_steps: list[dict[str, Any]] = []
    if round_advanced:
        pass_points = 0
        if summary["kind"] == "pass_booster":
            pass_points = before._booster_pass_points(
                player,
                before._player_booster(player),
            )
        action_changes = [
            change
            for change in _interactive_player_changes(before, after, player)
            if change["kind"] == "booster"
        ]
        effects = [{
            "player": player,
            "costs": costs,
            "gains": ([{"resource": "vp", "amount": pass_points}] if pass_points else []),
            "changes": action_changes,
        }]
        adjustments = {(player, "vp"): -pass_points} if pass_points else {}
        income_effects = _interactive_player_effects(
            before,
            after,
            player,
            [],
            adjustments=adjustments,
            change_kinds={"power", "brainstone", "counter"},
        )
        for effect in income_effects:
            effect["sources"] = _income_sources(after, effect["player"])
        round_tile = after.round_scoring_tiles[after.round_number - 1]
        automatic_steps.append({
            "kind": "round_income",
            "round": after.round_number,
            "label": f"Round {after.round_number} automatic income",
            "gaia_phase": before.round_number >= 1,
            "components": [
                _component_ref(
                    "round_scoring",
                    round_tile,
                    ROUND_SCORING_TILES[round_tile].label,
                    f"RND-{round_tile + 1:02d}",
                    relation="round",
                )
            ],
            "effects": income_effects,
            "changes": _interactive_board_changes(before, after),
        })
        board_changes: list[dict[str, Any]] = []
    else:
        effects = _interactive_player_effects(before, after, player, costs)
        board_changes = _interactive_board_changes(before, after)
    return {
        "move": move,
        "player": player,
        "role": role,
        "round": before.round_number,
        "phase": _interactive_phase(before),
        **summary,
        "components": components,
        "effects": effects,
        "changes": board_changes,
        "automatic_steps": automatic_steps,
    }


def _interactive_ai_components(
    state: GaiaState,
) -> tuple[object, str]:
    expected_architecture = architecture_for_players(state.num_players)
    model_directory = Path.cwd() / "runs" / "models"
    checkpoints = (
        model_directory
        / f"gaia-standard-{state.num_players}p-{expected_architecture}.pt",
        model_directory / f"gaia-standard-{state.num_players}p.pt",
    )
    for checkpoint in checkpoints:
        if not checkpoint.is_file():
            continue
        try:
            model, _metadata = load_checkpoint(checkpoint, "cpu")
            expected = (state.observation_size, state.action_size, state.num_players)
            actual = (
                model.config.observation_size,
                model.config.action_size,
                model.config.num_players,
            )
            if actual == expected and model.architecture == expected_architecture:
                label = "NNUE" if model.architecture == "nnue" else "KataGo"
                return NetworkEvaluator(model, "cpu"), f"{label} + PIMCTS"
        except Exception:
            pass
    return GaiaHeuristicEvaluator(), "Heuristic PIMCTS"


def _create_interactive_session(
    initial: GaiaState,
    config: dict[str, Any],
    roles: list[str],
) -> dict[str, Any]:
    created_at = datetime.now(UTC).isoformat()
    initial_snapshot = initial.snapshot()
    evaluator, engine = _interactive_ai_components(initial)
    searches = [
        PUCTSearch(
            evaluator,
            SearchConfig(
                simulations=config["simulations"],
                c_puct=1.5,
                root_noise_fraction=0.0,
                seed=config["seed"] + player,
            ),
        )
        for player in range(initial.num_players)
    ]
    return {
        "status": "active",
        "session_id": f"play-{uuid.uuid4().hex[:10]}",
        "created_at": created_at,
        "completed_at": None,
        "started_clock": perf_counter(),
        "duration_seconds": 0.0,
        "config": dict(config),
        "roles": roles,
        "state": initial,
        "searches": searches,
        "engine": engine,
        "move": 0,
        "history": [],
        "trace_steps": [
            {
                "sequence": 0,
                "timestamp": created_at,
                "move": 0,
                "player": initial_snapshot.get("current_player"),
                "role": None,
                "action": None,
                "action_label": "initial state",
                "legal_actions": len(initial.legal_actions()),
                "search_sampled": False,
                "root_value": None,
                "candidates": [],
                "record": None,
                "state": initial_snapshot,
            }
        ],
        "undo_stack": [],
        "last_action": None,
        "last_search": None,
        "busy": False,
        "error": None,
        "archive_error": None,
        "revision": 0,
    }


def _active_interactive_state(session: dict[str, Any]) -> GaiaState:
    state = session.get("state")
    if not isinstance(state, GaiaState):
        raise ValueError("no interactive game has been started")
    if state.is_terminal:
        raise ValueError("the interactive game is already complete")
    return state


def _apply_interactive_action(
    session: dict[str, Any],
    action: int,
    role: str,
    search_summary: dict[str, Any] | None = None,
) -> None:
    before = _active_interactive_state(session)
    player = before.current_player
    if action not in before.legal_actions():
        raise ValueError(f"illegal action {action}")
    legal_action_count = len(before.legal_actions())
    session["undo_stack"].append(
        {
            "state": before,
            "status": session["status"],
            "last_search": session.get("last_search"),
        }
    )
    after = before.apply(action)
    session["state"] = after
    session["move"] += 1
    session["revision"] += 1
    session["last_action"] = _interactive_action_record(
        before,
        after,
        action,
        session["move"],
        player,
        role,
    )
    session["last_search"] = search_summary
    session["history"].append(session["last_action"])
    session["trace_steps"].append(
        {
            "sequence": session["move"],
            "timestamp": datetime.now(UTC).isoformat(),
            "move": session["move"],
            "player": player,
            "role": role,
            "action": action,
            "action_label": session["last_action"]["label"],
            "legal_actions": legal_action_count,
            "search_sampled": search_summary is not None,
            "root_value": (search_summary or {}).get("root_value"),
            "candidates": (search_summary or {}).get("candidates") or [],
            "record": session["last_action"],
            "state": after.snapshot(),
        }
    )
    session["error"] = None
    if after.is_terminal:
        session["status"] = "complete"
        session["completed_at"] = datetime.now(UTC).isoformat()


def _undo_interactive_action(session: dict[str, Any]) -> int:
    history = session.get("history", [])
    undo_stack = session.get("undo_stack", [])
    human_index = next(
        (index for index in range(len(history) - 1, -1, -1) if history[index]["role"] == "human"),
        None,
    )
    if human_index is None:
        raise ValueError("no human action is available to undo")
    if len(undo_stack) != len(history):
        raise ValueError("interactive undo history is inconsistent")

    frame = undo_stack[human_index]
    undone = len(history) - human_index
    del history[human_index:]
    del undo_stack[human_index:]
    del session["trace_steps"][human_index + 1 :]
    session["state"] = frame["state"]
    session["status"] = frame["status"]
    session["move"] = len(history)
    session["last_action"] = history[-1] if history else None
    session["last_search"] = frame["last_search"]
    session["busy"] = False
    session["error"] = None
    session["revision"] += 1
    return undone


def _persist_interactive_session(
    server: DashboardServer,
    session: dict[str, Any],
) -> None:
    state = session.get("state")
    if not isinstance(state, GaiaState):
        raise ValueError("no interactive game has been started")
    session["duration_seconds"] = max(
        0.0,
        perf_counter() - float(session["started_clock"]),
    )
    state_snapshot = state.snapshot()
    config = dict(session["config"])
    config["random_setup"] = _resolved_random_setup(state)
    scores = list(state.final_scores()) if state.is_terminal else None
    record = {
        "run_id": session["session_id"],
        "source": "local",
        "started_at": session["created_at"],
        "updated_at": datetime.now(UTC).isoformat(),
        "completed_at": session.get("completed_at"),
        "status": session["status"],
        "ruleset": state_snapshot.get("ruleset"),
        "config": config,
        "roles": list(session["roles"]),
        "engine": session["engine"],
        "trace": {
            "run_id": session["session_id"],
            "iteration": 1,
            "game": 1,
            "started_at": session["created_at"],
            "completed_at": session.get("completed_at"),
            "summary": {
                "moves": session["move"],
                "positions": session["move"],
                "scores": scores,
                "returns": None,
                "duration_seconds": session["duration_seconds"],
            },
            "trace_complete": True,
            "captured_moves": session["move"],
            "steps": list(session["trace_steps"]),
        },
    }
    session["archive_path"] = str(write_local_game(server.history_path, record))


def _save_interactive_session(
    server: DashboardServer,
    session: dict[str, Any],
) -> None:
    try:
        _persist_interactive_session(server, session)
        session["archive_error"] = None
    except OSError as error:
        session["archive_error"] = f"{type(error).__name__}: {error}"


def _interactive_undo_count(session: dict[str, Any]) -> int:
    history = session.get("history", [])
    human_index = next(
        (index for index in range(len(history) - 1, -1, -1) if history[index]["role"] == "human"),
        None,
    )
    return 0 if human_index is None else len(history) - human_index


def _interactive_session_snapshot(session: dict[str, Any]) -> dict[str, Any]:
    if session.get("status") == "idle":
        return {"status": "idle"}
    state = session["state"]
    state_snapshot = state.snapshot()
    current_player = None if state.is_terminal else state.current_player
    legal_actions = [] if state.is_terminal else [
        _interactive_action_snapshot(state, action)
        for action in state.legal_actions()
    ]
    config = dict(session["config"])
    config["random_setup"] = _resolved_random_setup(state)
    undo_count = _interactive_undo_count(session)
    return {
        "status": session["status"],
        "session_id": session["session_id"],
        "move": session["move"],
        "revision": session["revision"],
        "busy": bool(session.get("busy")),
        "error": session.get("error"),
        "roles": list(session["roles"]),
        "current_role": None if current_player is None else session["roles"][current_player],
        "ai_engine": session["engine"],
        "booster_action_offset": PASS_BOOSTER_OFFSET,
        "config": config,
        "state": state_snapshot,
        "legal_actions": legal_actions,
        "last_action": session.get("last_action"),
        "last_search": session.get("last_search"),
        "history": list(session["history"]),
        "archive_path": session.get("archive_path"),
        "archive_error": session.get("archive_error"),
        "can_undo": undo_count > 0 and not session.get("busy"),
        "undo_count": undo_count,
    }


def _run_single_simulation(
    server: DashboardServer,
    initial: GaiaState,
    config: dict[str, Any],
    run_id: str,
) -> None:
    telemetry = JsonlTelemetry(server.metrics_path, run_id=run_id)
    started = perf_counter()
    state = initial
    moves = 0
    telemetry.emit(
        "run_started",
        config={"mode": "single-simulation", **config, "iterations": 1},
        device="heuristic-pimcts",
        observation_size=initial.observation_size,
        action_size=initial.action_size,
        state=initial.snapshot(),
    )
    telemetry.emit(
        "self_play_started",
        iteration=1,
        game_in_iteration=1,
        games_per_iteration=1,
        total_games=0,
        state=initial.snapshot(),
    )
    try:
        evaluator = GaiaHeuristicEvaluator()
        searches = [
            PUCTSearch(
                evaluator,
                SearchConfig(
                    simulations=config["simulations"],
                    c_puct=1.5,
                    root_noise_fraction=0.0,
                    seed=config["seed"] + player,
                ),
            )
            for player in range(config["players"])
        ]
        while not state.is_terminal:
            if moves >= 512:
                raise RuntimeError("single simulation exceeded 512 moves")
            before = state
            result = searches[before.current_player].run(
                before,
                add_root_noise=False,
                temperature=0.0,
            )
            action = int(np.argmax(result.policy))
            state = before.apply(action)
            moves += 1
            top_actions = np.argsort(result.policy)[-3:][::-1]
            telemetry.emit(
                "self_play_step",
                iteration=1,
                game_in_iteration=1,
                move=moves,
                player=before.current_player,
                action=action,
                action_label=before.describe_action(action),
                legal_actions=len(before.legal_actions()),
                search_sampled=True,
                root_value=result.root_value,
                candidates=[
                    {
                        "action": int(candidate),
                        "label": before.describe_action(int(candidate)),
                        "probability": float(result.policy[candidate]),
                        "visits": int(result.visits[candidate]),
                    }
                    for candidate in top_actions
                    if result.policy[candidate] > 0
                ],
                state=state.snapshot(),
            )
            with server.simulation_lock:
                server.simulation.update(
                    move=moves,
                    current_player=None if state.is_terminal else state.current_player,
                    last_action=before.describe_action(action),
                )
        duration = perf_counter() - started
        scores = state.final_scores()
        telemetry.emit(
            "self_play_completed",
            iteration=1,
            game_in_iteration=1,
            games_per_iteration=1,
            total_games=1,
            moves=moves,
            positions=moves,
            replay_positions=0,
            duration_seconds=duration,
            scores=scores,
            returns=state.returns(),
            state=state.snapshot(),
        )
        telemetry.emit(
            "iteration_completed",
            iteration=1,
            new_positions=moves,
            replay_positions=0,
            duration_seconds=duration,
        )
        telemetry.emit(
            "run_completed",
            iterations=1,
            total_games=1,
            replay_positions=0,
            duration_seconds=duration,
        )
        with server.simulation_lock:
            server.simulation.update(
                status="complete",
                move=moves,
                scores=list(scores),
                duration_seconds=duration,
            )
    except Exception as error:
        duration = perf_counter() - started
        telemetry.emit(
            "run_failed",
            error_type=type(error).__name__,
            message=str(error),
            duration_seconds=duration,
            state=state.snapshot(),
        )
        with server.simulation_lock:
            server.simulation.update(
                status="failed",
                move=moves,
                error=f"{type(error).__name__}: {error}",
                duration_seconds=duration,
            )


def serve_dashboard(
    metrics_path: str | Path,
    host: str = "127.0.0.1",
    port: int = 8765,
    history_path: str | Path | None = None,
) -> None:
    server = create_dashboard_server(
        metrics_path,
        host,
        port,
        history_path=history_path,
    )
    print(f"GaiaZero dashboard: http://{host}:{server.server_port}")
    print(f"Metrics source: {server.metrics_path}")
    print(f"Local history: {server.history_path}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
