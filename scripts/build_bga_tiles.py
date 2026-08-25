"""Build the dashboard tile assets from verified BGA sprite sheets.

By default this script only reads sprite sheets already stored in
``runs/bga-tiles``. Pass ``--download`` explicitly to refresh those sources.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from urllib.request import Request, urlopen

from PIL import Image, ImageStat


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = ROOT / "runs" / "bga-tiles"
OUTPUT_DIR = ROOT / "src" / "gaiazero" / "web" / "assets" / "tiles"
SOURCE_ROOT = (
    "https://x.boardgamearena.net/data/themereleases/260812-1015/"
    "games/gaiaproject/260630-1810/img"
)


@dataclass(frozen=True)
class SpriteSheet:
    filename: str
    sha256: str


SHEETS = (
    SpriteSheet(
        "techs.png",
        "47329C264E74DB4F00E975BA01A843186AC060D8BBC4AE58DEEABFF08F3E5C19",
    ),
    SpriteSheet(
        "boosterTile.png",
        "31ADB8A5F818A5A892E9C1467BD73DE7A3003F1A980DBF5647DDA3FEA49B6B27",
    ),
    SpriteSheet(
        "roundBonus.png",
        "6A8518A86A27DA7EE30883CE45F68659C5E9F060B4FC31D734C418C59E021BB4",
    ),
    SpriteSheet(
        "endGameBonus.png",
        "3494E636C3CDF4D6DDC9F6D247EF1360285F8B9D6CAB10D1BCA10CC181D082CF",
    ),
    SpriteSheet(
        "federationTokens.png",
        "64B3A410C6D01B3BCCCDB3663A7257C826FE04CB1F70F3D6880CFB26BC393A5F",
    ),
)


def _download(sheet: SpriteSheet, source_dir: Path) -> Path:
    request = Request(
        f"{SOURCE_ROOT}/{sheet.filename}",
        headers={"User-Agent": "GaiaProject asset updater"},
    )
    with urlopen(request, timeout=60) as response:
        content = response.read()
    path = source_dir / sheet.filename
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_bytes(content)
    temporary.replace(path)
    return path


def _validate_source(sheet: SpriteSheet, source_dir: Path, download: bool) -> Path:
    path = source_dir / sheet.filename
    if download:
        path = _download(sheet, source_dir)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing local sprite sheet: {path}. "
            "Place the BGA source there or rerun with --download."
        )
    digest = sha256(path.read_bytes()).hexdigest().upper()
    if digest != sheet.sha256:
        raise ValueError(f"BGA sprite sheet failed validation: {path.name} ({digest})")
    return path


def _crop_columns(
    source: Path,
    *,
    first_column: int,
    count: int,
    source_size: tuple[int, int],
    output_size: tuple[int, int],
    filename_pattern: str,
    output_format: str,
) -> list[Path]:
    tile_width, tile_height = source_size
    with Image.open(source) as sprite:
        sprite.load()
        outputs = []
        for output_index, column in enumerate(
            range(first_column, first_column + count),
            start=1,
        ):
            left = column * tile_width
            image = sprite.crop((left, 0, left + tile_width, tile_height))
            image = image.resize(output_size, Image.Resampling.LANCZOS)
            if max(ImageStat.Stat(image.convert("RGB")).stddev) < 1:
                raise ValueError(f"BGA tile crop is blank: {source.name} column {column}")

            output = OUTPUT_DIR / filename_pattern.format(output_index)
            if output_format == "PNG":
                image.convert("RGBA").save(output, format="PNG", optimize=True)
            elif output_format == "GIF":
                image.convert("RGBA").save(output, format="GIF", optimize=True)
            else:
                background = Image.new("RGB", image.size, "white")
                if image.mode == "RGBA":
                    background.paste(image, mask=image.getchannel("A"))
                else:
                    background.paste(image.convert("RGB"))
                background.save(output, format="JPEG", quality=95, optimize=True)
            outputs.append(output)
    return outputs


def build(source_dir: Path, download: bool = False) -> list[Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sources = {
        sheet.filename: _validate_source(sheet, source_dir, download)
        for sheet in SHEETS
    }
    outputs = []
    outputs.extend(_crop_columns(
        sources["techs.png"],
        first_column=1,
        count=9,
        source_size=(150, 116),
        output_size=(130, 97),
        filename_pattern="tech-standard-{:02d}.jpg",
        output_format="JPEG",
    ))
    outputs.extend(_crop_columns(
        sources["techs.png"],
        first_column=10,
        count=15,
        source_size=(150, 116),
        output_size=(104, 80),
        filename_pattern="tech-advanced-{:02d}.jpg",
        output_format="JPEG",
    ))
    outputs.extend(_crop_columns(
        sources["boosterTile.png"],
        first_column=1,
        count=10,
        source_size=(116, 353),
        output_size=(78, 249),
        filename_pattern="booster-{:02d}.jpg",
        output_format="JPEG",
    ))
    outputs.extend(_crop_columns(
        sources["roundBonus.png"],
        first_column=1,
        count=10,
        source_size=(182, 211),
        output_size=(128, 147),
        filename_pattern="round-scoring-{:02d}.gif",
        output_format="GIF",
    ))
    outputs.extend(_crop_columns(
        sources["endGameBonus.png"],
        first_column=1,
        count=6,
        source_size=(199, 128),
        output_size=(141, 87),
        filename_pattern="final-scoring-{:02d}.jpg",
        output_format="JPEG",
    ))
    outputs.extend(_crop_columns(
        sources["federationTokens.png"],
        first_column=1,
        count=7,
        source_size=(96, 119),
        output_size=(96, 119),
        filename_pattern="federation-{:02d}.png",
        output_format="PNG",
    ))
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument(
        "--download",
        action="store_true",
        help="Explicitly download and verify the BGA source sprite sheets.",
    )
    args = parser.parse_args()
    outputs = build(args.source_dir.resolve(), args.download)
    print(f"Updated {len(outputs)} tile assets in {OUTPUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
