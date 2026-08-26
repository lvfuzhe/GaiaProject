from gaiazero.npz_history import delete_training_history

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Delete a history replay created from NPZ training data")
    parser.add_argument("run_id")
    parser.add_argument("--history-dir", type=Path, required=True)
    args = parser.parse_args()
    deleted = delete_training_history(args.history_dir, args.run_id)
    if not deleted:
        raise SystemExit(f"training history not found: {args.run_id}")
    print(f"deleted {args.run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
