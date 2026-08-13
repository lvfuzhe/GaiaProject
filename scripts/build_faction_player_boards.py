"""Build the 14 complete player-board previews from the shared board and faction strips."""

from __future__ import annotations

from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "src" / "gaiazero" / "web" / "assets" / "factions"
TEMPLATE = ASSET_DIR / "player-board-template.jpg"
TEMPLATE_CROP = (10, 58, 990, 699)
FACTION_PANEL = (520, 78, 980, 274)
COMMON_BOARD_START_Y = 276

FACTION_COLORS = (
    "#4d78b8",  # Terrans
    "#4d78b8",  # Lantids
    "#d7a93e",  # Xenos
    "#d7a93e",  # Gleens
    "#967248",  # Taklons
    "#967248",  # Ambas
    "#ba5640",  # Hadsch Hallas
    "#ba5640",  # Ivits
    "#d86b39",  # Geodens
    "#d86b39",  # Bal T'aks
    "#8b8f96",  # Firaks
    "#8b8f96",  # Bescods
    "#b7d8e7",  # Nevlas
    "#b7d8e7",  # Itars
)


def _tint_common_board(image: Image.Image, color: str) -> Image.Image:
    """Retint only the printed player-color regions while preserving icon colors."""
    hsv = image.convert("HSV")
    pixels = hsv.load()
    target_h, target_s, _target_v = Image.new("RGB", (1, 1), color).convert("HSV").getpixel((0, 0))

    for y in range(COMMON_BOARD_START_Y, hsv.height):
        for x in range(hsv.width):
            hue, saturation, value = pixels[x, y]
            is_red_print = saturation >= 52 and (hue <= 18 or hue >= 238)
            if not is_red_print:
                continue
            if target_s < 45:
                saturation = max(10, int(saturation * 0.18))
            else:
                saturation = max(38, min(255, int(saturation * 0.72 + target_s * 0.28)))
            pixels[x, y] = (target_h, saturation, value)
    return hsv.convert("RGB")


def build() -> list[Path]:
    if not TEMPLATE.is_file():
        raise FileNotFoundError(f"missing shared player-board template: {TEMPLATE}")

    template = Image.open(TEMPLATE).convert("RGB").crop(TEMPLATE_CROP)
    panel_width = FACTION_PANEL[2] - FACTION_PANEL[0]
    panel_height = FACTION_PANEL[3] - FACTION_PANEL[1]
    outputs: list[Path] = []

    for number, color in enumerate(FACTION_COLORS, start=1):
        strip_path = ASSET_DIR / f"faction-{number:02d}.jpg"
        if not strip_path.is_file():
            raise FileNotFoundError(f"missing faction strip: {strip_path}")
        strip = Image.open(strip_path).convert("RGB").resize(
            (panel_width, panel_height),
            Image.Resampling.LANCZOS,
        )
        board = _tint_common_board(template.copy(), color)
        board.paste(strip, FACTION_PANEL[:2])
        output = ASSET_DIR / f"player-board-{number:02d}.jpg"
        board.save(output, "JPEG", quality=94, subsampling=1, optimize=True, progressive=True)
        outputs.append(output)

    return outputs


if __name__ == "__main__":
    for output_path in build():
        print(output_path.relative_to(ROOT))
