from gaiazero.distributed import main


if __name__ == "__main__":
    raise SystemExit(main(["selfplay", *__import__("sys").argv[1:]]))
