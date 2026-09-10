"""Measure the CPU reference baseline used by step-0 contract acceptance.

This deliberately measures the Python reference path only. GPU/TensorRT
measurements belong to the later performance gate and are not inferred from
CPU timings.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import psutil
import torch

from gaiazero.config import GaiaTrainingConfig
from gaiazero.game import GaiaHeuristicEvaluator, GaiaState
from gaiazero.gnn import GraphHybridNetwork, graph_inputs_from_state
from gaiazero.mcts import PUCTSearch, SearchConfig
from gaiazero.selfplay import SelfPlayConfig, play_self_game


def _measure(callable_):
    process = psutil.Process()
    before = process.memory_info().rss
    started = time.perf_counter()
    result = callable_()
    elapsed = time.perf_counter() - started
    after = process.memory_info().rss
    return result, elapsed, max(0, after - before)


def main() -> None:
    players = 3
    seed = 20260828
    state = GaiaState.initial(players, seed)
    evaluator = GaiaHeuristicEvaluator()
    _, mcts_seconds, mcts_rss_delta = _measure(
        lambda: PUCTSearch(evaluator, SearchConfig(simulations=8, seed=seed)).run(state)
    )
    game, selfplay_seconds, selfplay_rss_delta = _measure(
        lambda: play_self_game(
            state,
            evaluator,
            SearchConfig(simulations=1, seed=seed),
            SelfPlayConfig(
                temperature_moves=0,
                max_moves=512,
                add_root_noise=False,
                seed=seed,
            ),
        )
    )
    config = GaiaTrainingConfig.load().graph_network_config(players)
    model = GraphHybridNetwork(config).eval()
    inputs = graph_inputs_from_state(state, config)
    with torch.inference_mode():
        _, gnn_batch1_seconds, gnn_batch1_rss_delta = _measure(lambda: model(*inputs))
        batch_inputs = tuple(
            value.repeat((4,) + (1,) * (value.ndim - 1)) for value in inputs
        )
        _, gnn_batch4_seconds, gnn_batch4_rss_delta = _measure(
            lambda: model(*batch_inputs)
        )
    payload = {
        "schema_version": "step0-cpu-baseline-v1",
        "measured_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": {
            "python": sys.version.split()[0],
            "pytorch": torch.__version__,
            "platform": platform.platform(),
            "device": "cpu",
            "gpu_memory_bytes": None,
        },
        "workload": {
            "player_count": players,
            "seed": seed,
            "mcts_simulations": 8,
            "selfplay_simulations": 1,
            "selfplay_positions": len(game.examples),
            "gnn_network_config_id": config.network_config_id,
        },
        "measurements": {
            "mcts8_seconds": mcts_seconds,
            "mcts8_runs_per_second": 1.0 / mcts_seconds,
            "selfplay_one_game_seconds": selfplay_seconds,
            "selfplay_positions_per_second": len(game.examples) / selfplay_seconds,
            "gnn_batch1_seconds": gnn_batch1_seconds,
            "gnn_batch1_inferences_per_second": 1.0 / gnn_batch1_seconds,
            "gnn_batch4_seconds": gnn_batch4_seconds,
            "gnn_batch4_inferences_per_second": 4.0 / gnn_batch4_seconds,
        },
        "rss_delta_bytes": {
            "mcts8": mcts_rss_delta,
            "selfplay_one_game": selfplay_rss_delta,
            "gnn_batch1": gnn_batch1_rss_delta,
            "gnn_batch4": gnn_batch4_rss_delta,
        },
        "limitations": [
            "CPU reference measurements only; no GPU/TensorRT throughput is inferred.",
            "RSS deltas are process-level observations, not allocator or VRAM usage.",
            "Use the same workload and script after C++/TensorRT integration for comparison.",
        ],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
