"""Download and validate the BGA map-piece sprite sheets used by the star map."""

from __future__ import annotations

import argparse
from hashlib import sha256
from pathlib import Path
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = ROOT / "runs" / "bga-pieces"
OUTPUT_DIR = ROOT / "src" / "gaiazero" / "web" / "assets" / "map-pieces"
SOURCE_ROOT = (
    "https://x.boardgamearena.net/data/themereleases/260812-1015/"
    "games/gaiaproject/260630-1810/img"
)
SOURCES = {
    "blankHex.png": "EC7E0DBE14B0D18E4CA8A5A4F9006FB2E1A592066CAA1FB3FB46F40CD6EE4CAE",
    "structures.png": "6C4D82B722452CA172337A93B9CEE9F8314790FC4AC0BFC63B0C8F7A02581A11",
    "planets.png": "59CB838D21B3788C42AFC65B86C94C20CDB3BE7A2AA532CF2897E62E7333E9DF",
    "icons.png": "A426F26D8269E074E417E7038F318452E71C4FF8063F249D6E1EB146296849E3",
}


def _download(name: str, source_dir: Path) -> Path:
    request = Request(
        f"{SOURCE_ROOT}/{name}",
        headers={"User-Agent": "GaiaProject asset updater"},
    )
    with urlopen(request, timeout=60) as response:
        content = response.read()
    path = source_dir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_bytes(content)
    temporary.replace(path)
    return path


def _validate(name: str, source_dir: Path, download: bool) -> Path:
    path = source_dir / name
    if download:
        path = _download(name, source_dir)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing BGA source sheet: {path}. Rerun with --download."
        )
    digest = sha256(path.read_bytes()).hexdigest().upper()
    if digest != SOURCES[name]:
        raise ValueError(f"BGA sprite sheet failed validation: {name} ({digest})")
    return path


def build(source_dir: Path, download: bool = False) -> list[Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = {name: _validate(name, source_dir, download) for name in SOURCES}
    outputs = []
    for name, source in paths.items():
        output = OUTPUT_DIR / name
        output.write_bytes(source.read_bytes())
        outputs.append(output)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    outputs = build(args.source_dir.resolve(), args.download)
    print(f"Updated {len(outputs)} BGA map-piece sprite sheets in {OUTPUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
