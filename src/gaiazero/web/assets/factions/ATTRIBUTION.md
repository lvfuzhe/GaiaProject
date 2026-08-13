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

Gaia Project and its artwork are the property of their respective copyright
holders. These files are not represented as original project artwork.

The shared player-board reference in `player-board-template.jpg` comes from
the [Ivits player-board image on Ludopedia](https://ludopedia.com.br/jogo/gaia-project/imagens/131851).
The `player-board-01.jpg` through `player-board-14.jpg` previews are generated
locally by `scripts/build_faction_player_boards.py`: each combines that shared
board with the matching faction strip above and retints only the common printed
player-color regions. They are setup/reference previews; live resources and
structure inventory remain rendered from game state by the dashboard.
