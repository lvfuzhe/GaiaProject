"""Download the 14 verified player boards used by Board Game Arena."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "src" / "gaiazero" / "web" / "assets" / "factions"
BGA_ASSET_ROOT = (
    "https://x.boardgamearena.net/data/themereleases/260812-1015/"
    "games/gaiaproject/260630-1810/img"
)
PLAYER_BOARDS = (
    ("Terrans", "B8C804C4CD83E4CA52183B771EB221D9761592169E62B6FF15D906522F1E1CE6"),
    ("Lantids", "EFFE0E9DF5A5611CD325381D2332B10E6C719D4E0A91D0DBF70050EDD83C7691"),
    ("Xenos", "9615BC9FD6CDD9D882CFFAFF969F42807B358E9027272F008F011FB676FB7D90"),
    ("Gleens", "5925510A5C0D64EF1FA070DFF4B01F7D75789971C3C41ACDB2A57FB2A5842C52"),
    ("Taklons", "C2A729135041A8D9E6D57A44995917856299B8333279AC19EA27F7B6FFEAAF43"),
    ("Ambas", "FE9FFE039C2DFB1A11488A5E42451560894D90391879FF2119AA6B76DD096869"),
    ("Hadsch Hallas", "AA267C6020E1CF80C3ACC1F451EE30FB4DC600B8AAF3D79013C37603CB0F6676"),
    ("Ivits", "141EBCEFE4A26D9FA0F96DE7A1080B1E994145FCC04EA435B661D860784A0385"),
    ("Geodens", "AC61AA9C4F7A5E7076A2ABD02811A5E13F9E675CCE704C9F79E8C7BE921A8B52"),
    ("Bal T'aks", "FC106493FADEFC60F19B6583DDA59BB6F6017E6F7D27AE18D86D5695B6D4197D"),
    ("Firaks", "4D96FF535EE50B529ABF662AB847F3F9BB6938CFDC8AEBFFFB4FD855757E5C36"),
    ("Bescods", "ED328DFD860A9D0428100D299E8A9BB66FD88E789A0D6D883216CFF457C50CC1"),
    ("Nevlas", "A8AD6A1F676C5712F2650979260A7333507EC4367AC8E2C070CE6B9AC42F58A1"),
    ("Itars", "38A05D18A67CDBD9FD11E5BA8F13398AF0B4DA294449359B5760DC7D782B33CF"),
)


def build() -> list[Path]:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    for number, (faction, expected_hash) in enumerate(PLAYER_BOARDS, start=1):
        request = Request(
            f"{BGA_ASSET_ROOT}/race{number}.jpg",
            headers={"User-Agent": "GaiaProject asset updater"},
        )
        with urlopen(request, timeout=60) as response:
            content = response.read()
        digest = sha256(content).hexdigest().upper()
        if not content.startswith(b"\xff\xd8\xff") or digest != expected_hash:
            raise ValueError(
                f"BGA race{number} ({faction}) failed validation: {digest}"
            )
        output = ASSET_DIR / f"player-board-{number:02d}.jpg"
        temporary = output.with_suffix(".jpg.tmp")
        temporary.write_bytes(content)
        temporary.replace(output)
        outputs.append(output)

    return outputs


if __name__ == "__main__":
    for output_path in build():
        print(output_path.relative_to(ROOT))
