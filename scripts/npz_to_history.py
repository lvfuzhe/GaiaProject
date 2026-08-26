from gaiazero.npz_history import convert_npz_directory, convert_npz_to_history

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert GaiaZero NPZ samples to dashboard history JSON")
    parser.add_argument("source", type=Path, help="NPZ file or directory")
    parser.add_argument("--history-dir", type=Path, required=True)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()
    if args.source.is_dir():
        outputs = convert_npz_directory(args.source, args.history_dir)
    else:
        outputs = [convert_npz_to_history(args.source, args.history_dir, run_id=args.run_id)]
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
