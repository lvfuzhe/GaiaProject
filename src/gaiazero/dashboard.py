from __future__ import annotations

import json
import threading
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from time import perf_counter
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np

from gaiazero.game import GaiaHeuristicEvaluator, GaiaState
from gaiazero.mcts import PUCTSearch, SearchConfig
from gaiazero.telemetry import (
    JsonlTelemetry,
    build_history_index,
    read_events,
    read_game_trace,
)

WEB_ROOT = Path(__file__).with_name("web")
ASSETS = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
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
    name = f"faction-{number:02d}.jpg"
    ASSETS[f"/assets/factions/{name}"] = (f"assets/factions/{name}", "image/jpeg")
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
        quiet: bool = False,
    ) -> None:
        self.metrics_path = Path(metrics_path).resolve()
        self.quiet = quiet
        self.simulation_lock = threading.Lock()
        self.simulation: dict[str, Any] = {"status": "idle"}
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
        if request.path == "/api/game":
            self._serve_game(parse_qs(request.query))
            return
        if request.path == "/api/simulation":
            self._send_json(self._simulation_status())
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
            config["random_setup"] = resolved_random_setup
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            self._send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
            return

        if request.path == "/api/setup/preview":
            self._send_json({"config": config, "state": initial.snapshot()})
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
        self._send_json({**build_history_index(self.server.metrics_path), "source": str(self.server.metrics_path)})

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
    quiet: bool = False,
) -> DashboardServer:
    return DashboardServer((host, port), metrics_path, quiet=quiet)


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
    allowed_fields = {*array_fields, "terraforming_federation_tile"}
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
    if "terraforming_federation_tile" in value:
        normalized["terraforming_federation_tile"] = int(
            value["terraforming_federation_tile"]
        )
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
    return GaiaState.initial(
        config["players"],
        config["seed"],
        faction_indices=tuple(config["factions"]),
        first_player=config["first_player"],
        **overrides,
    )


def _resolved_random_setup(state: GaiaState) -> dict[str, Any]:
    turn_order = [
        (state.first_player + offset) % state.num_players
        for offset in range(state.num_players)
    ]
    assigned_boosters = [
        next(
            booster
            for booster, owner in enumerate(state.booster_owner)
            if owner == player
        )
        for player in reversed(turn_order)
    ]
    public_boosters = [
        booster
        for booster, owner in enumerate(state.booster_owner)
        if owner == -1
    ]
    return {
        "sector_tiles": list(state.sector_tiles),
        "sector_rotations": list(state.sector_rotations),
        "booster_tiles": assigned_boosters + public_boosters,
        "round_scoring_tiles": list(state.round_scoring_tiles),
        "final_scoring_tiles": list(state.final_scoring_tiles),
        "standard_tech_tiles": list(state.standard_tech_tiles),
        "advanced_tech_tiles": list(state.advanced_tech_tiles),
        "terraforming_federation_tile": state.terraforming_federation_tile,
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
) -> None:
    server = create_dashboard_server(metrics_path, host, port)
    print(f"GaiaZero dashboard: http://{host}:{server.server_port}")
    print(f"Metrics source: {server.metrics_path}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
