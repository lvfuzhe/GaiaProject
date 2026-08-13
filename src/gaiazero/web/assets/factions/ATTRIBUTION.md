# Faction and player-board artwork

The 14 faction board scans in this directory were downloaded from the
[BoardGameHelpers Gaia Project setup tool](https://www.boardgamehelpers.com/GaiaProject/LoadGame.aspx).
They are used only to visualize the corresponding physical player-board faces.

Source-to-local mapping:

| Local file | Source face | Faction |
| --- | --- | --- |
| `faction-01.jpg` | `Faction_TL_A.jpg` | Terrans |
| `faction-02.jpg` | `Faction_TL_B.jpg` | Lantids |
| `faction-03.jpg` | `Faction_XG_A.jpg` | Xenos |
| `faction-04.jpg` | `Faction_XG_B.jpg` | Gleens |
| `faction-05.jpg` | `Faction_TA_A.jpg` | Taklons |
| `faction-06.jpg` | `Faction_TA_B.jpg` | Ambas |
| `faction-07.jpg` | `Faction_HI_A.jpg` | Hadsch Hallas |
| `faction-08.jpg` | `Faction_HI_B.jpg` | Ivits |
| `faction-09.jpg` | `Faction_GB_A.jpg` | Geodens |
| `faction-10.jpg` | `Faction_GB_B.jpg` | Bal T'aks |
| `faction-11.jpg` | `Faction_FB_A.jpg` | Firaks |
| `faction-12.jpg` | `Faction_FB_B.jpg` | Bescods |
| `faction-13.jpg` | `Faction_IN_B.jpg` | Nevlas |
| `faction-14.jpg` | `Faction_IN_A.jpg` | Itars |

The complete `player-board-01.jpg` through `player-board-14.jpg` files are the
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
