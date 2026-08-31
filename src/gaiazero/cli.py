from __future__ import annotations

import argparse
from pathlib import Path

from gaiazero.arena import evaluate_against, play_arena_game
from gaiazero.dashboard import serve_dashboard
from gaiazero.config import load_training_config
from gaiazero.distributed import PipelineConfig, run_pipeline
from gaiazero.npz_history import convert_npz_directory, convert_npz_to_history, delete_training_history
from gaiazero.game import (
    GaiaHeuristicEvaluator,
    GaiaState,
)
from gaiazero.mcts import SearchConfig
from gaiazero.model import (
    NetworkEvaluator,
    architecture_for_players,
    load_checkpoint,
)


def _search_config(args: argparse.Namespace, seed: int | None = None) -> SearchConfig:
    return SearchConfig(
        simulations=args.simulations,
        c_puct=args.c_puct,
        dirichlet_alpha=args.dirichlet_alpha,
        root_noise_fraction=args.root_noise_fraction,
        seed=args.seed if seed is None else seed,
    )


def _game_components():
    return GaiaState, GaiaHeuristicEvaluator()


def command_demo(args: argparse.Namespace) -> None:
    state_type, evaluator = _game_components()
    initial = state_type.initial(args.players, args.seed)
    result = play_arena_game(initial, [evaluator] * args.players, _search_config(args))
    if args.show_actions:
        state = initial
        for turn, action in enumerate(result.actions, start=1):
            print(f"{turn:03d} P{state.current_player}: {state.describe_action(action)}")
            state = state.apply(action)
    print(result.final_state.render())
    print(f"Returns: {result.final_state.returns().tolist()}")


def command_dashboard(args: argparse.Namespace) -> None:
    serve_dashboard(
        args.storage_dir,
        args.host,
        args.port,
        args.history_dir,
        args.pipeline_root,
    )


def command_pipeline(args: argparse.Namespace) -> None:
    """Run the five asynchronous GaiaZero multiplayer workers."""
    if args.training_config is not None:
        config = load_training_config(args.training_config).pipeline_config(args.players)
        run_pipeline(config)
        return
    run_pipeline(
        PipelineConfig(
            root=Path(args.root),
            players=args.players,
            seed=args.seed,
            simulations=args.simulations,
            c_puct=args.c_puct,
            dirichlet_alpha=args.dirichlet_alpha,
            root_noise_fraction=args.root_noise_fraction,
            add_root_noise=args.add_root_noise,
            temperature_moves=args.temperature_moves,
            max_moves=args.max_moves,
            poll_seconds=args.poll_seconds,
            games_per_cycle=args.games_per_cycle,
            shuffle_pack_size=args.shuffle_pack_size,
            replay_capacity=args.replay_capacity,
            batch_size=args.batch_size,
            updates_per_cycle=args.updates_per_cycle,
            min_replay=args.min_replay,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            hidden_size=args.hidden_size,
            residual_blocks=args.residual_blocks,
            device=args.device,
            gate_games=args.gate_games,
            gate_threshold=args.gate_threshold,
        )
    )


def command_npz_to_history(args: argparse.Namespace) -> None:
    source = Path(args.source)
    if source.is_dir():
        outputs = convert_npz_directory(source, args.history_dir)
    else:
        outputs = [convert_npz_to_history(source, args.history_dir, run_id=args.run_id)]
    for output in outputs:
        print(output)


def command_delete_training_history(args: argparse.Namespace) -> None:
    if not delete_training_history(args.history_dir, args.run_id):
        raise ValueError(f"training history not found: {args.run_id}")
    print(f"deleted {args.run_id}")


def command_evaluate(args: argparse.Namespace) -> None:
    model, metadata = load_checkpoint(args.checkpoint, args.device)
    state_type, baseline = _game_components()
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gaiazero", description="AlphaZero + PIMCTS for Gaia")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="play a heuristic PIMCTS demonstration")
    demo.add_argument("--players", type=int, choices=(2, 3, 4), default=2)
    demo.add_argument("--show-actions", action="store_true")
    _add_search_arguments(demo)
    demo.set_defaults(handler=command_demo)

    evaluate = subparsers.add_parser("evaluate", help="evaluate a checkpoint against heuristic PIMCTS")
    evaluate.add_argument("checkpoint")
    evaluate.add_argument("--players", type=int, choices=(2, 3, 4), default=2)
    evaluate.add_argument("--games", type=int, default=10)
    evaluate.add_argument("--device", default="auto")
    _add_search_arguments(evaluate)
    evaluate.set_defaults(handler=command_evaluate)

    dashboard = subparsers.add_parser("dashboard", help="serve the live training dashboard")
    dashboard.add_argument(
        "--storage-dir",
        default="runs",
        help="dashboard history and pipeline parent directory",
    )
    dashboard.add_argument(
        "--pipeline-root",
        default="runs/multiplayer-pipeline",
        help="asynchronous NPZ pipeline directory",
    )
    dashboard.add_argument(
        "--history-dir",
        default=None,
        help="local interactive-game archive (default: <storage-dir>/history)",
    )
    dashboard.add_argument("--host", default="127.0.0.1")
    dashboard.add_argument("--port", type=int, default=8765)
    dashboard.set_defaults(handler=command_dashboard)

    pipeline = subparsers.add_parser(
        "pipeline",
        help="run asynchronous multiplayer self-play, shuffle, train, export and gatekeeper workers",
    )
    pipeline.add_argument("--root", default="runs/multiplayer-pipeline")
    pipeline.add_argument("--players", type=int, choices=(2, 3, 4), default=4)
    pipeline.add_argument("--seed", type=int, default=0)
    pipeline.add_argument("--simulations", type=int, default=64)
    pipeline.add_argument("--c-puct", type=float, default=1.5)
    pipeline.add_argument("--dirichlet-alpha", type=float, default=0.3)
    pipeline.add_argument("--root-noise-fraction", type=float, default=0.25)
    pipeline.add_argument(
        "--add-root-noise",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    pipeline.add_argument("--temperature-moves", type=int, default=24)
    pipeline.add_argument("--max-moves", type=int, default=512)
    pipeline.add_argument("--poll-seconds", type=float, default=2.0)
    pipeline.add_argument("--games-per-cycle", type=int, default=1)
    pipeline.add_argument("--shuffle-pack-size", type=int, default=4096)
    pipeline.add_argument("--replay-capacity", type=int, default=200_000)
    pipeline.add_argument("--batch-size", type=int, default=256)
    pipeline.add_argument("--updates-per-cycle", type=int, default=32)
    pipeline.add_argument("--min-replay", type=int, default=256)
    pipeline.add_argument("--learning-rate", type=float, default=1e-3)
    pipeline.add_argument("--weight-decay", type=float, default=1e-4)
    pipeline.add_argument("--hidden-size", type=int, default=256)
    pipeline.add_argument("--residual-blocks", type=int, default=4)
    pipeline.add_argument("--device", default="auto")
    pipeline.add_argument("--gate-games", type=int, default=20)
    pipeline.add_argument("--gate-threshold", type=float, default=0.55)
    pipeline.add_argument(
        "--training-config",
        type=Path,
        default=None,
        help="load runtime, seed-stream and network settings from gaia-training.json",
    )
    pipeline.set_defaults(handler=command_pipeline)

    npz_history = subparsers.add_parser(
        "npz-to-history",
        help="explicitly convert NPZ self-play samples to dashboard replay JSON",
    )
    npz_history.add_argument("source", help="an NPZ file or directory")
    npz_history.add_argument("--history-dir", default="runs/history")
    npz_history.add_argument("--run-id", default=None)
    npz_history.set_defaults(handler=command_npz_to_history)

    delete_training = subparsers.add_parser(
        "delete-training-history",
        help="delete one replay previously converted from NPZ training data",
    )
    delete_training.add_argument("run_id")
    delete_training.add_argument("--history-dir", default="runs/history")
    delete_training.set_defaults(handler=command_delete_training_history)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.handler(args)
