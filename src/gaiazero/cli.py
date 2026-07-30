from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from gaiazero.arena import evaluate_against, play_arena_game
from gaiazero.game import MiniGaiaHeuristicEvaluator, MiniGaiaState
from gaiazero.mcts import SearchConfig
from gaiazero.model import (
    NetworkConfig,
    NetworkEvaluator,
    PolicyValueNetwork,
    load_checkpoint,
    resolve_device,
    save_checkpoint,
)
from gaiazero.replay import ReplayBuffer
from gaiazero.selfplay import SelfPlayConfig, play_self_game
from gaiazero.training import AlphaZeroTrainer, TrainerConfig


def _search_config(args: argparse.Namespace, seed: int | None = None) -> SearchConfig:
    return SearchConfig(
        simulations=args.simulations,
        c_puct=args.c_puct,
        dirichlet_alpha=args.dirichlet_alpha,
        root_noise_fraction=args.root_noise_fraction,
        seed=args.seed if seed is None else seed,
    )


def command_demo(args: argparse.Namespace) -> None:
    evaluator = MiniGaiaHeuristicEvaluator()
    initial = MiniGaiaState.initial(args.players, args.seed)
    result = play_arena_game(initial, [evaluator] * args.players, _search_config(args))
    if args.show_actions:
        state = initial
        for turn, action in enumerate(result.actions, start=1):
            print(f"{turn:03d} P{state.current_player}: {state.describe_action(action)}")
            state = state.apply(action)
    print(result.final_state.render())
    print(f"Returns: {result.final_state.returns().tolist()}")


def command_train(args: argparse.Namespace) -> None:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    template = MiniGaiaState.initial(args.players, args.seed)
    network_config = NetworkConfig(
        observation_size=template.observation_size,
        action_size=template.action_size,
        num_players=args.players,
        hidden_size=args.hidden_size,
        residual_blocks=args.residual_blocks,
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
    baseline = MiniGaiaHeuristicEvaluator()
    total_games = 0

    for iteration in range(1, args.iterations + 1):
        positions = 0
        for game in range(args.games_per_iteration):
            game_seed = args.seed + total_games
            result = play_self_game(
                MiniGaiaState.initial(args.players, game_seed),
                evaluator,
                _search_config(args, game_seed),
                SelfPlayConfig(
                    temperature_moves=args.temperature_moves,
                    seed=game_seed,
                ),
            )
            replay.extend(result.examples)
            positions += len(result.examples)
            total_games += 1
        metrics = trainer.train_updates(replay, args.updates_per_iteration)
        print(
            f"iteration={iteration} games={total_games} new_positions={positions} "
            f"replay={len(replay)} loss={metrics.loss:.4f} "
            f"policy={metrics.policy_loss:.4f} value={metrics.value_loss:.4f}"
        )

        if args.eval_games > 0:
            summary = evaluate_against(
                lambda seed: MiniGaiaState.initial(args.players, args.seed + 100_000 + seed),
                evaluator,
                baseline,
                num_players=args.players,
                games=args.eval_games,
                search_config=_search_config(args, args.seed + iteration),
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
                "ruleset": "mini-gaia-v1",
            },
        )
    print(f"checkpoint={Path(args.output).resolve()}")


def command_evaluate(args: argparse.Namespace) -> None:
    model, metadata = load_checkpoint(args.checkpoint, args.device)
    state = MiniGaiaState.initial(args.players, args.seed)
    config = model.config
    expected = (state.observation_size, state.action_size, state.num_players)
    actual = (config.observation_size, config.action_size, config.num_players)
    if actual != expected:
        raise ValueError(f"checkpoint dimensions {actual} do not match game {expected}")
    challenger = NetworkEvaluator(model, args.device)
    summary = evaluate_against(
        lambda seed: MiniGaiaState.initial(args.players, args.seed + seed),
        challenger,
        MiniGaiaHeuristicEvaluator(),
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

    train = subparsers.add_parser("train", help="run neural self-play training")
    train.add_argument("--players", type=int, choices=(2, 3, 4), default=2)
    train.add_argument("--iterations", type=int, default=5)
    train.add_argument("--games-per-iteration", type=int, default=8)
    train.add_argument("--updates-per-iteration", type=int, default=50)
    train.add_argument("--eval-games", type=int, default=2)
    train.add_argument("--temperature-moves", type=int, default=24)
    train.add_argument("--replay-capacity", type=int, default=200_000)
    train.add_argument("--batch-size", type=int, default=256)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--hidden-size", type=int, default=256)
    train.add_argument("--residual-blocks", type=int, default=4)
    train.add_argument("--device", default="auto")
    train.add_argument("--output", default="runs/mini-gaia.pt")
    _add_search_arguments(train)
    train.set_defaults(handler=command_train)

    evaluate = subparsers.add_parser("evaluate", help="evaluate a checkpoint against heuristic PIMCTS")
    evaluate.add_argument("checkpoint")
    evaluate.add_argument("--players", type=int, choices=(2, 3, 4), default=2)
    evaluate.add_argument("--games", type=int, default=10)
    evaluate.add_argument("--device", default="auto")
    _add_search_arguments(evaluate)
    evaluate.set_defaults(handler=command_evaluate)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.handler(args)

