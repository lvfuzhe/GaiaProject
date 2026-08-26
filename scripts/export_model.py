from gaiazero.distributed import main


if __name__ == "__main__":
    raise SystemExit(main(["export", *__import__("sys").argv[1:]]))
