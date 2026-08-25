from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from gaiazero.bga import BGA_NOTIFICATION_FUNCTIONS


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit a locally imported BGA replay")
    parser.add_argument("replay", type=Path, help="Path to bga-<table>.json")
    args = parser.parse_args()

    record = json.loads(args.replay.read_text(encoding="utf-8"))
    steps = record.get("trace", {}).get("steps", [])
    unknown = sorted(
        {
            str(notice.get("type") or "")
            for packet in record.get("bga", {}).get("log_packets", [])
            for notice in packet.get("data", [])
            if isinstance(notice, dict)
            and notice.get("type") not in BGA_NOTIFICATION_FUNCTIONS
        }
    )
    failed_moves = [
        step.get("move")
        for step in steps[1:]
        if step.get("record", {}).get("vp", {}).get("audit", {}).get("matches_state")
        is not True
    ]
    reasons: Counter[str] = Counter()
    player_deltas = [0] * len(steps[-1]["state"]["scores"])
    for step in steps[1:]:
        for event in step.get("record", {}).get("vp", {}).get("events", []):
            delta = int(event.get("delta", 0))
            reasons[str(event.get("reason") or "unknown")] += delta
            player_deltas[int(event["player"])] += delta

    initial = [int(value) for value in steps[0]["state"]["scores"]]
    final = [int(value) for value in steps[-1]["state"]["scores"]]
    expected = [base + delta for base, delta in zip(initial, player_deltas, strict=True)]
    report = {
        "moves": len(steps) - 1,
        "initial_scores": initial,
        "event_deltas": player_deltas,
        "expected_scores": expected,
        "final_scores": final,
        "vp_moves_ok": not failed_moves,
        "failed_vp_moves": failed_moves,
        "unknown_notifications": unknown,
        "vp_by_reason": dict(sorted(reasons.items())),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not failed_moves and not unknown and expected == final else 1


if __name__ == "__main__":
    raise SystemExit(main())
