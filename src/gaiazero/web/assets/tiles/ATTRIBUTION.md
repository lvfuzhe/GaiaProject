# Gaia Project board tile artwork

The technology, round-scoring, end-game-scoring, and booster images in this
directory are cropped from the Board Game Arena Gaia Project sprite sheets:

- `techs.png`
- `boosterTile.png`
- `roundBonus.png`
- `endGameBonus.png`

`scripts/build_bga_tiles.py` validates the source files by SHA-256 and produces
the runtime assets locally. The script reads `runs/bga-tiles` by default; it
does not require an HTTP request during normal application startup or builds.

The original Gaia Project artwork and component designs remain the property of
their respective copyright holders, including Feuerland Spiele and Z-Man
Games. These files are included to reproduce the physical board setup in the
local monitoring interface. Board Game Arena also retains rights in its site
assets. Review the applicable rights before redistributing the artwork outside
this project.
