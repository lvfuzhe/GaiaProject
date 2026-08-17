from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from gaiazero.arena import evaluate_against, play_arena_game
from gaiazero.dashboard import serve_dashboard
from gaiazero.game import (
    GaiaHeuristicEvaluator,
    GaiaState,
    MiniGaiaHeuristicEvaluator,
    MiniGaiaState,
)
from gaiazero.mcts import SearchConfig
from gaiazero.model import (
    NetworkConfig,
    NetworkEvaluator,
    PolicyValueNetwork,
    architecture_for_players,
    load_checkpoint,
    resolve_device,
    save_checkpoint,
)
from gaiazero.replay import ReplayBuffer
from gaiazero.selfplay import SelfPlayConfig, play_self_game
from gaiazero.telemetry import JsonlTelemetry
from gaiazero.training import AlphaZeroTrainer, TrainerConfig


def _search_config(args: argparse.Namespace, seed: int | None = None) -> SearchConfig:
    return SearchConfig(
        simulations=args.simulations,
        c_puct=args.c_puct,
        dirichlet_alpha=args.dirichlet_alpha,
        root_noise_fraction=args.root_noise_fraction,
        seed=args.seed if seed is None else seed,
    )


def _game_components(ruleset: str):
    if ruleset == "mini":
        return MiniGaiaState, MiniGaiaHeuristicEvaluator()
    return GaiaState, GaiaHeuristicEvaluator()


def command_demo(args: argparse.Namespace) -> None:
    state_type, evaluator = _game_components(args.ruleset)
    initial = state_type.initial(args.players, args.seed)
    result = play_arena_game(initial, [evaluator] * args.players, _search_config(args))
    if args.show_actions:
        state = initial
        for turn, action in enumerate(result.actions, start=1):
            print(f"{turn:03d} P{state.current_player}: {state.describe_action(action)}")
            state = state.apply(action)
    print(result.final_state.render())
    print(f"Returns: {result.final_state.returns().tolist()}")


def command_train(args: argparse.Namespace) -> None:
    if args.metrics_move_interval < 1:
        raise ValueError("metrics-move-interval must be positive")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    state_type, baseline = _game_components(args.ruleset)
    template = state_type.initial(args.players, args.seed)
    network_config = NetworkConfig(
        observation_size=template.observation_size,
        action_size=template.action_size,
        num_players=args.players,
        hidden_size=args.hidden_size,
        residual_blocks=args.residual_blocks,
        architecture=architecture_for_players(args.players),
    )
    model = PolicyValueNetwork(network_config)
    evaluator = NetworkEvaluator(model, device)
    trainer = AlphaZeroTrainer(
        model,
        TrainerConfig(
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            device=str(device),
        ),
    )
    replay = ReplayBuffer(args.replay_capacity, args.seed)
    total_games = 0
    telemetry = JsonlTelemetry(args.metrics)
    run_started = perf_counter()
    configuration = {
        key: value
        for key, value in vars(args).items()
        if key not in {"handler", "command"}
    }
    configuration["architecture"] = model.architecture
    print(
        f"training players={args.players} architecture={model.architecture} "
        f"device={device}"
    )
    telemetry.emit(
        "run_started",
        config=configuration,
        device=str(device),
        architecture=model.architecture,
        model_parameters=sum(parameter.numel() for parameter in model.parameters()),
        observation_size=template.observation_size,
        action_size=template.action_size,
        state=template.snapshot(),
    )

    try:
        for iteration in range(1, args.iterations + 1):
            positions = 0
            iteration_started = perf_counter()
            for game in range(args.games_per_iteration):
                game_seed = args.seed + total_games
                initial = state_type.initial(args.players, game_seed)
                game_started = perf_counter()
                telemetry.emit(
                    "self_play_started",
                    iteration=iteration,
                    game_in_iteration=game + 1,
                    games_per_iteration=args.games_per_iteration,
                    total_games=total_games,
                    state=initial.snapshot(),
                )

                def observe_move(move: int, before, action: int, after, search_result) -> None:
                    search_sampled = move == 1 or move % args.metrics_move_interval == 0
                    top_actions = np.argsort(search_result.policy)[-3:][::-1] if search_sampled else ()
                    telemetry.emit(
                        "self_play_step",
                        iteration=iteration,
                        game_in_iteration=game + 1,
                        move=move,
                        player=before.current_player,
                        action=action,
                        action_label=before.describe_action(action),
                        legal_actions=len(before.legal_actions()),
                        search_sampled=search_sampled,
                        root_value=search_result.root_value if search_sampled else None,
                        candidates=[
                            {
                                "action": int(candidate),
                                "label": before.describe_action(int(candidate)),
                                "probability": float(search_result.policy[candidate]),
                                "visits": int(search_result.visits[candidate]),
                            }
                            for candidate in top_actions
                            if search_sampled and search_result.policy[candidate] > 0
                        ],
                        state=after.snapshot(),
                    )

                result = play_self_game(
                    initial,
                    evaluator,
                    _search_config(args, game_seed),
                    SelfPlayConfig(
                        temperature_moves=args.temperature_moves,
                        seed=game_seed,
                    ),
                    observer=observe_move,
                )
                replay.extend(result.examples)
                positions += len(result.examples)
                total_games += 1
                telemetry.emit(
                    "self_play_completed",
                    iteration=iteration,
                    game_in_iteration=game + 1,
                    games_per_iteration=args.games_per_iteration,
                    total_games=total_games,
                    moves=len(result.actions),
                    positions=len(result.examples),
                    replay_positions=len(replay),
                    duration_seconds=perf_counter() - game_started,
                    scores=result.final_state.final_scores(),
                    returns=result.final_state.returns(),
                    state=result.final_state.snapshot(),
                )

            telemetry.emit(
                "training_started",
                iteration=iteration,
                updates=args.updates_per_iteration,
                replay_positions=len(replay),
            )

            def observe_update(update: int, update_metrics) -> None:
                telemetry.emit(
                    "training_update",
                    iteration=iteration,
                    update=update,
                    updates=args.updates_per_iteration,
                    replay_positions=len(replay),
                    loss=update_metrics.loss,
                    policy_loss=update_metrics.policy_loss,
                    value_loss=update_metrics.value_loss,
                    policy_entropy=update_metrics.policy_entropy,
                )

            metrics = trainer.train_updates(
                replay,
                args.updates_per_iteration,
                observer=observe_update,
            )
            print(
                f"iteration={iteration} games={total_games} new_positions={positions} "
                f"replay={len(replay)} loss={metrics.loss:.4f} "
                f"policy={metrics.policy_loss:.4f} value={metrics.value_loss:.4f}"
            )

            if args.eval_games > 0:
                telemetry.emit("arena_started", iteration=iteration, games=args.eval_games)
                arena_started = perf_counter()
                summary = evaluate_against(
                    lambda seed: state_type.initial(args.players, args.seed + 100_000 + seed),
                    evaluator,
                    baseline,
                    num_players=args.players,
                    games=args.eval_games,
                    search_config=_search_config(args, args.seed + iteration),
                )
                telemetry.emit(
                    "arena_completed",
                    iteration=iteration,
                    games=summary.games,
                    mean_value=summary.mean_value,
                    first_places=summary.first_places,
                    draws=summary.draws,
                    mean_score=summary.mean_score,
                    duration_seconds=perf_counter() - arena_started,
                )
                print(
                    f"arena_games={summary.games} mean_value={summary.mean_value:.3f} "
                    f"first={summary.first_places} draws={summary.draws} "
                    f"mean_score={summary.mean_score:.2f}"
                )

            save_checkpoint(
                args.output,
                model,
                optimizer=trainer.optimizer,
                metadata={
                    "iteration": iteration,
                    "self_play_games": total_games,
                    "replay_positions": len(replay),
                    "player_count": args.players,
                    "architecture": model.architecture,
                    "model_id": (
                        f"{args.ruleset}-{args.players}p-{model.architecture}"
                    ),
                    "ruleset": template.snapshot()["ruleset"],
                },
            )
            telemetry.emit(
                "iteration_completed",
                iteration=iteration,
                iterations=args.iterations,
                total_games=total_games,
                replay_positions=len(replay),
                new_positions=positions,
                loss=metrics.loss,
                policy_loss=metrics.policy_loss,
                value_loss=metrics.value_loss,
                policy_entropy=metrics.policy_entropy,
                duration_seconds=perf_counter() - iteration_started,
                checkpoint=str(Path(args.output).resolve()),
            )
        telemetry.emit(
            "run_completed",
            iterations=args.iterations,
            total_games=total_games,
            replay_positions=len(replay),
            duration_seconds=perf_counter() - run_started,
            checkpoint=str(Path(args.output).resolve()),
        )
    except Exception as error:
        telemetry.emit(
            "run_failed",
            error_type=type(error).__name__,
            message=str(error),
            duration_seconds=perf_counter() - run_started,
        )
        raise
    print(f"checkpoint={Path(args.output).resolve()}")
    print(f"metrics={Path(args.metrics).resolve()}")


def command_train_all(args: argparse.Namespace) -> None:
    """Train one independent checkpoint for each requested player count."""
    output_dir = Path(args.output_dir)
    metrics_dir = Path(args.metrics_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    player_counts = tuple(sorted(set(args.player_counts)))
    if player_counts != (2, 3, 4):
        raise ValueError("train-all requires player counts 2, 3 and 4")

    for index, players in enumerate(player_counts):
        child = argparse.Namespace(**vars(args))
        child.command = "train"
        child.handler = command_train
        child.players = players
        child.seed = args.seed + index * 1_000_003
        architecture = architecture_for_players(players)
        child.output = str(
            output_dir / f"gaia-{args.ruleset}-{players}p-{architecture}.pt"
        )
        child.metrics = str(
            metrics_dir
            / f"metrics-{args.ruleset}-{players}p-{architecture}.jsonl"
        )
        print(
            f"[train-all] players={players} "
            f"architecture={architecture} output={child.output} "
            f"metrics={child.metrics} seed={child.seed}"
        )
        command_train(child)


def command_dashboard(args: argparse.Namespace) -> None:
    serve_dashboard(args.metrics, args.host, args.port)


def command_evaluate(args: argparse.Namespace) -> None:
    model, metadata = load_checkpoint(args.checkpoint, args.device)
    state_type, baseline = _game_components(args.ruleset)
    state = state_type.initial(args.players, args.seed)
    config = model.config
    expected = (state.observation_size, state.action_size, state.num_players)
    actual = (config.observation_size, config.action_size, config.num_players)
    if actual != expected:
        raise ValueError(f"checkpoint dimensions {actual} do not match game {expected}")
    expected_architecture = architecture_for_players(args.players)
    if config.architecture != expected_architecture:
        raise ValueError(
            f"checkpoint architecture {config.architecture} does not match "
            f"{args.players}-player {expected_architecture} architecture"
        )
    challenger = NetworkEvaluator(model, args.device)
    summary = evaluate_against(
        lambda seed: state_type.initial(args.players, args.seed + seed),
        challenger,
        baseline,
        num_players=args.players,
        games=args.games,
        search_config=_search_config(args),
    )
    print(f"metadata={metadata}")
    print(
        f"games={summary.games} mean_value={summary.mean_value:.3f} "
        f"first={summary.first_places} draws={summary.draws} mean_score={summary.mean_score:.2f}"
    )


def _add_search_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--simulations", type=int, default=64)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--dirichlet-alpha", type=float, default=0.3)
    parser.add_argument("--root-noise-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)


def _add_ruleset_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ruleset", choices=("standard", "mini"), default="standard")


def _add_training_arguments(
    parser: argparse.ArgumentParser,
    *,
    include_players: bool,
    include_output_paths: bool,
) -> None:
    if include_players:
        parser.add_argument("--players", type=int, choices=(2, 3, 4), default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--games-per-iteration", type=int, default=8)
    parser.add_argument("--updates-per-iteration", type=int, default=50)
    parser.add_argument("--eval-games", type=int, default=2)
    parser.add_argument("--temperature-moves", type=int, default=24)
    parser.add_argument("--replay-capacity", type=int, default=200_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--residual-blocks", type=int, default=4)
    parser.add_argument("--device", default="auto")
    if include_output_paths:
        parser.add_argument("--output", default="runs/gaia-standard.pt")
        parser.add_argument("--metrics", default="runs/metrics.jsonl")
    parser.add_argument(
        "--metrics-move-interval",
        type=int,
        default=4,
        help="sample search candidates every N moves; rule states are recorded every move",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gaiazero", description="AlphaZero + PIMCTS for Gaia")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="play a heuristic PIMCTS demonstration")
    demo.add_argument("--players", type=int, choices=(2, 3, 4), default=2)
    demo.add_argument("--show-actions", action="store_true")
    _add_ruleset_argument(demo)
    _add_search_arguments(demo)
    demo.set_defaults(handler=command_demo)

    train = subparsers.add_parser("train", help="run neural self-play training")
    _add_training_arguments(train, include_players=True, include_output_paths=True)
    _add_ruleset_argument(train)
    _add_search_arguments(train)
    train.set_defaults(handler=command_train)

    train_all = subparsers.add_parser(
        "train-all",
        help="train independent neural models for 2, 3 and 4 players",
    )
    train_all.add_argument("--output-dir", default="runs/models")
    train_all.add_argument("--metrics-dir", default="runs/metrics-by-players")
    _add_training_arguments(train_all, include_players=False, include_output_paths=False)
    _add_ruleset_argument(train_all)
    _add_search_arguments(train_all)
    train_all.set_defaults(handler=command_train_all, player_counts=(2, 3, 4))

    evaluate = subparsers.add_parser("evaluate", help="evaluate a checkpoint against heuristic PIMCTS")
    evaluate.add_argument("checkpoint")
    evaluate.add_argument("--players", type=int, choices=(2, 3, 4), default=2)
    evaluate.add_argument("--games", type=int, default=10)
    evaluate.add_argument("--device", default="auto")
    _add_ruleset_argument(evaluate)
    _add_search_arguments(evaluate)
    evaluate.set_defaults(handler=command_evaluate)

    dashboard = subparsers.add_parser("dashboard", help="serve the live training dashboard")
    dashboard.add_argument("--metrics", default="runs/metrics.jsonl")
    dashboard.add_argument("--host", default="127.0.0.1")
    dashboard.add_argument("--port", type=int, default=8765)
    dashboard.set_defaults(handler=command_dashboard)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.handler(args)
