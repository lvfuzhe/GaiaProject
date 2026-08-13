"""Download the verified high-resolution research board used by BGA."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "src" / "gaiazero" / "web" / "assets" / "boards" / "research-board.png"
SOURCE = (
    "https://x.boardgamearena.net/data/themereleases/260812-1015/"
    "games/gaiaproject/260630-1810/img/gameBoard.png"
)
EXPECTED_SHA256 = "6A9CB95AFD5410927303E56F671206821188FD309FC55E2B268116A67DE44418"


def build() -> Path:
    request = Request(SOURCE, headers={"User-Agent": "GaiaProject asset updater"})
    with urlopen(request, timeout=60) as response:
        content = response.read()
    digest = sha256(content).hexdigest().upper()
    if not content.startswith(b"\x89PNG\r\n\x1a\n") or digest != EXPECTED_SHA256:
        raise ValueError(f"BGA research board failed validation: {digest}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".png.tmp")
    temporary.write_bytes(content)
    temporary.replace(OUTPUT)
    return OUTPUT


if __name__ == "__main__":
    print(build().relative_to(ROOT))
