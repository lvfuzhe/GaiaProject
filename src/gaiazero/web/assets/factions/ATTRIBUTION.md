The `player-board-01.jpg` through `player-board-14.jpg` files are the
actual full faction-board backgrounds published by Board Game Arena for Gaia
Project. They were downloaded on 2026-08-13 from BGA game release
`260630-1810` under theme release `260812-1015`:

`https://x.boardgamearena.net/data/themereleases/260812-1015/games/gaiaproject/260630-1810/img/raceN.jpg`

The BGA race order matches the local faction ids exactly:

| Local file | BGA file | Faction |
| --- | --- | --- |
| `player-board-01.jpg` | `race1.jpg` | Terrans |
| `player-board-02.jpg` | `race2.jpg` | Lantids |
| `player-board-03.jpg` | `race3.jpg` | Xenos |
| `player-board-04.jpg` | `race4.jpg` | Gleens |
| `player-board-05.jpg` | `race5.jpg` | Taklons |
| `player-board-06.jpg` | `race6.jpg` | Ambas |
| `player-board-07.jpg` | `race7.jpg` | Hadsch Hallas |
| `player-board-08.jpg` | `race8.jpg` | Ivits |
| `player-board-09.jpg` | `race9.jpg` | Geodens |
| `player-board-10.jpg` | `race10.jpg` | Bal T'aks |
| `player-board-11.jpg` | `race11.jpg` | Firaks |
| `player-board-12.jpg` | `race12.jpg` | Bescods |
| `player-board-13.jpg` | `race13.jpg` | Nevlas |
| `player-board-14.jpg` | `race14.jpg` | Itars |

`scripts/build_faction_player_boards.py` downloads these files and verifies
their SHA-256 digests before replacing local assets. Gaia Project, Board Game
Arena, and all game artwork remain the property of their respective copyright
holders. These files are used only for the local game interface and are not
represented as original project artwork.
