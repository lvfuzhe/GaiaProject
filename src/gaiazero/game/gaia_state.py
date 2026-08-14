from __future__ import annotations

from dataclasses import dataclass, replace
from enum import IntEnum
from itertools import combinations
from math import exp

import numpy as np

from gaiazero.core import BoolArray, FloatArray
from gaiazero.game.gaia_setup import (
    BOOSTER_COUNT,
    MAX_PLANETS,
    generate_setup,
    hex_distance,
)


class Terrain(IntEnum):
    TERRA = 0
    DESERT = 1
    SWAMP = 2
    VOLCANIC = 3
    OXIDE = 4
    TITANIUM = 5
    ICE = 6
    TRANSDIM = 7
    GAIA = 8


class Building(IntEnum):
    EMPTY = 0
    MINE = 1
    TRADING_STATION = 2
    RESEARCH_LAB = 3
    PLANETARY_INSTITUTE = 4
    ACADEMY = 5


class Track(IntEnum):
    TERRAFORMING = 0
    NAVIGATION = 1
    ARTIFICIAL_INTELLIGENCE = 2
    GAIA_PROJECT = 3
    ECONOMY = 4
    SCIENCE = 5


class PowerAction(IntEnum):
    KNOWLEDGE_THREE = 0
    TERRAFORM_TWO = 1
    ORE_TWO = 2
    CREDITS_SEVEN = 3
    KNOWLEDGE_TWO = 4
    ORE_ONE = 5
    POWER_TOKENS_TWO = 6


class QicAction(IntEnum):
    TECH = 0
    FEDERATION_REWARD = 1
    PLANET_TYPES = 2


@dataclass(frozen=True, slots=True)
class FactionSpec:
    name: str
    home: Terrain
    start_track: Track
    power: tuple[int, int, int]
    board: int
    ability: str
    starting_qic: int = 1
    starting_structures: int = 2
    starts_with_pi: bool = False
    places_last: bool = False
    federation_threshold: int = 7
    gaia_to_bowl_two: bool = False
    passive_power_token: bool = False
    knowledge_for_new_type: bool = False
    starting_credits: int = 15
    starting_ore: int = 4
    starting_knowledge: int = 3
    income_credits: int = 0
    income_ore: int = 0
    income_knowledge: int = 0
    income_qic: int = 0
    income_power_tokens: int = 0


FACTIONS: tuple[FactionSpec, ...] = (
    FactionSpec("Terrans", Terrain.TERRA, Track.GAIA_PROJECT, (4, 4, 0), 0, "Gaia power returns to bowl II", gaia_to_bowl_two=True),
    FactionSpec("Lantids", Terrain.TERRA, Track.SCIENCE, (4, 0, 0), 0, "May coexist on colonized planets", starting_credits=13),
    FactionSpec("Xenos", Terrain.DESERT, Track.ARTIFICIAL_INTELLIGENCE, (2, 4, 0), 1, "Starts with a third mine; its PI lowers federation power to 6", starting_structures=3),
    FactionSpec("Gleens", Terrain.DESERT, Track.NAVIGATION, (2, 4, 0), 1, "Ore replaces Q.I.C. for Gaia colonization", starting_qic=0),
    FactionSpec("Taklons", Terrain.SWAMP, Track.ECONOMY, (2, 4, 0), 2, "Brainstone strengthens the power cycle", passive_power_token=True),
    FactionSpec("Ambas", Terrain.SWAMP, Track.NAVIGATION, (4, 4, 0), 2, "Planetary institute can swap with a mine", income_ore=1),
    FactionSpec("Hadsch Hallas", Terrain.OXIDE, Track.ECONOMY, (2, 4, 0), 3, "Credits unlock expanded free actions", income_credits=3),
    FactionSpec("Ivits", Terrain.OXIDE, Track.NAVIGATION, (4, 4, 0), 3, "Places its starting planetary institute after all starting mines", starting_structures=1, starts_with_pi=True, places_last=True, income_qic=1),
    FactionSpec("Geodens", Terrain.VOLCANIC, Track.TERRAFORMING, (2, 4, 0), 4, "Knowledge for newly colonized planet types", knowledge_for_new_type=True),
    FactionSpec("Bal T'aks", Terrain.VOLCANIC, Track.GAIA_PROJECT, (2, 2, 0), 4, "Gaiaformers can be converted to Q.I.C.", starting_qic=0),
    FactionSpec("Firaks", Terrain.TITANIUM, Track.SCIENCE, (2, 4, 0), 5, "May downgrade a research lab to research", starting_ore=3, starting_knowledge=2, income_knowledge=1),
    FactionSpec("Bescods", Terrain.TITANIUM, Track.ECONOMY, (2, 4, 0), 5, "Lowest research areas advance together", starting_knowledge=1, income_knowledge=-1),
    FactionSpec("Nevlas", Terrain.ICE, Track.SCIENCE, (2, 4, 0), 6, "Bowl III power counts double for free actions", starting_knowledge=2),
    FactionSpec("Itars", Terrain.ICE, Track.GAIA_PROJECT, (4, 4, 0), 6, "Gaia power can buy technology", starting_qic=1, starting_ore=5, income_power_tokens=1),
)

FACTION_BOARDS: tuple[tuple[int, int], ...] = (
    (0, 1),
    (2, 3),
    (4, 5),
    (6, 7),
    (8, 9),
    (10, 11),
    (12, 13),
)

# Side labels follow the printed player boards. The Ice board is ordered
# Nevlas/Itars internally, while its physical A/B faces are Itars/Nevlas.
FACTION_BOARD_SIDES: tuple[str, ...] = (
    "A",
    "B",
    "A",
    "B",
    "A",
    "B",
    "A",
    "B",
    "A",
    "B",
    "A",
    "B",
    "B",
    "A",
)

MAX_ROUNDS = 6
TRACK_COUNT = len(Track)
STANDARD_TECH_COUNT = 9
TECH_STANDARD_ACTION_COUNT = STANDARD_TECH_COUNT
TECH_ADVANCED_ACTION_COUNT = TRACK_COUNT
TECH_COUNT = TECH_STANDARD_ACTION_COUNT + TECH_ADVANCED_ACTION_COUNT
POWER_ACTION_COUNT = len(PowerAction)
MAX_BUILDINGS = {
    Building.MINE: 8,
    Building.TRADING_STATION: 4,
    Building.RESEARCH_LAB: 3,
    Building.PLANETARY_INSTITUTE: 1,
    Building.ACADEMY: 2,
}
STRUCTURE_POWER = {
    Building.MINE: 1,
    Building.TRADING_STATION: 2,
    Building.RESEARCH_LAB: 2,
    Building.PLANETARY_INSTITUTE: 3,
    Building.ACADEMY: 3,
}
ECONOMY_TRACK_INCOME: tuple[tuple[int, int, int], ...] = (
    (0, 0, 0),
    (2, 0, 1),
    (2, 1, 2),
    (3, 1, 3),
    (4, 2, 4),
    (0, 0, 0),
)
SCIENCE_TRACK_INCOME: tuple[int, ...] = (0, 1, 2, 3, 4, 0)


@dataclass(frozen=True, slots=True)
class TileSpec:
    key: str
    label: str
    kind: str = ""
    points: int = 0


ROUND_SCORING_TILES: tuple[TileSpec, ...] = (
    TileSpec("terraform-2", "Terraforming steps", "terraform", 2),
    TileSpec("research-2", "Advance research", "research", 2),
    TileSpec("mine-2", "Build mines", "mine", 2),
    TileSpec("federation-5", "Gain federation tokens", "federation", 5),
    TileSpec("trading-3", "Build trading stations", "trading", 3),
    TileSpec("trading-4", "Build trading stations", "trading", 4),
    TileSpec("gaia-mine-3", "Build mines on Gaia planets", "gaia", 3),
    TileSpec("gaia-mine-4", "Build mines on Gaia planets", "gaia", 4),
    TileSpec("big-5a", "Build PI or academy", "big", 5),
    TileSpec("big-5b", "Build PI or academy", "big", 5),
)

FINAL_SCORING_TILES: tuple[TileSpec, ...] = (
    TileSpec("federation-structures", "Structures in federations"),
    TileSpec("structures", "Total structures"),
    TileSpec("planet-types", "Colonized planet types"),
    TileSpec("gaia-planets", "Colonized Gaia planets"),
    TileSpec("sectors", "Colonized sectors"),
    TileSpec("satellites", "Placed satellites and space stations"),
)
FINAL_SCORING_NEUTRAL: tuple[int, ...] = (10, 11, 5, 4, 6, 8)

STANDARD_TECH_TILES: tuple[TileSpec, ...] = (
    TileSpec("ore-qic", "Immediately gain 1 ore and 1 Q.I.C."),
    TileSpec("planet-type-knowledge", "1 knowledge per colonized planet type"),
    TileSpec("vp-7", "Immediately gain 7 VP"),
    TileSpec("gaia-mine-vp", "3 VP when building a mine on a Gaia planet"),
    TileSpec("structure-power", "PI and academy power value becomes 4"),
    TileSpec("ore-power-income", "Income: 1 ore and charge 1 power"),
    TileSpec("knowledge-credit-income", "Income: 1 knowledge and 1 credit"),
    TileSpec("credits-income", "Income: 4 credits"),
    TileSpec("power-action", "Action: charge 4 power"),
)

ADVANCED_TECH_TILES: tuple[TileSpec, ...] = tuple(
    TileSpec(f"advanced-{index + 1:02d}", label)
    for index, label in enumerate(
        (
            "Action: gain 1 Q.I.C. and 5 credits",
            "Action: gain 3 ore",
            "Action: gain 3 knowledge",
            "2 VP per mine",
            "1 ore per colonized sector",
            "2 VP per colonized sector",
            "2 VP per Gaia planet",
            "5 VP per federation token",
            "4 VP per trading station",
            "Pass: 3 VP per federation token",
            "Pass: 3 VP per research lab",
            "Pass: 1 VP per planet type",
            "2 VP per research advance",
            "3 VP per mine built",
            "3 VP per trading station built",
        )
    )
)

FEDERATION_TILES: tuple[TileSpec, ...] = (
    TileSpec("vp-knowledge", "6 VP + 2 knowledge"),
    TileSpec("vp-ore", "7 VP + 2 ore"),
    TileSpec("vp-qic", "8 VP + 1 Q.I.C."),
    TileSpec("vp-power", "7 VP + power"),
    TileSpec("credits", "Credits + VP"),
    TileSpec("twelve-vp", "12 VP"),
)

BOOSTER_LABELS: tuple[str, ...] = (
    "2 credits; action: build a mine with 1 free terraforming step",
    "Charge 2 power; action: range +3 for one build or Gaia Project",
    "1 ore and 1 knowledge",
    "1 ore and charge 2 power",
    "2 credits and 1 Q.I.C.",
    "1 ore; pass: 1 VP per mine",
    "1 ore; pass: 2 VP per trading station",
    "1 knowledge; pass: 3 VP per research lab",
    "Charge 4 power; pass: 4 VP per PI or academy",
    "4 credits; pass: 1 VP per colonized Gaia planet",
)

N = MAX_PLANETS
BUILD_OFFSET = 0
GAIA_OFFSET = BUILD_OFFSET + N
UPGRADE_TRADING_OFFSET = GAIA_OFFSET + N
UPGRADE_LAB_OFFSET = UPGRADE_TRADING_OFFSET + N
UPGRADE_PI_OFFSET = UPGRADE_LAB_OFFSET + N
UPGRADE_ACADEMY_OFFSET = UPGRADE_PI_OFFSET + N
UPGRADE_QIC_ACADEMY_OFFSET = UPGRADE_ACADEMY_OFFSET + N
RESEARCH_OFFSET = UPGRADE_QIC_ACADEMY_OFFSET + N
POWER_OFFSET = RESEARCH_OFFSET + TRACK_COUNT
TECH_OFFSET = POWER_OFFSET + POWER_ACTION_COUNT
FEDERATION_OFFSET = TECH_OFFSET + TECH_COUNT
FEDERATION_ACTION = FEDERATION_OFFSET
FEDERATION_ACTION_COUNT = 6
QIC_ACADEMY_ACTION = FEDERATION_OFFSET + FEDERATION_ACTION_COUNT
STANDARD_TECH_ACTION = QIC_ACADEMY_ACTION + 1
ADVANCED_TECH_ACTION_OFFSET = STANDARD_TECH_ACTION + 1
ADVANCED_TECH_SPECIAL_COUNT = 3
QIC_TECH_ACTION = ADVANCED_TECH_ACTION_OFFSET + ADVANCED_TECH_SPECIAL_COUNT
QIC_FEDERATION_ACTION_OFFSET = QIC_TECH_ACTION + 1
QIC_FEDERATION_ACTION_COUNT = len(FEDERATION_TILES) + 1
QIC_PLANET_TYPES_ACTION = QIC_FEDERATION_ACTION_OFFSET + QIC_FEDERATION_ACTION_COUNT
BOOSTER_TERRAFORM_ACTION = QIC_PLANET_TYPES_ACTION + 1
BOOSTER_RANGE_ACTION = BOOSTER_TERRAFORM_ACTION + 1
PASS_BOOSTER_OFFSET = BOOSTER_RANGE_ACTION + 1
PASS_FINAL_ACTION = PASS_BOOSTER_OFFSET + BOOSTER_COUNT
ACTION_SIZE = PASS_FINAL_ACTION + 1


@dataclass(frozen=True, slots=True)
class PlayerState:
    faction: int
    credits: int = 15
    ore: int = 4
    knowledge: int = 3
    qic: int = 1
    vp: int = 10
    bowl_one: int = 2
    bowl_two: int = 4
    bowl_three: int = 0
    gaia_power: int = 0
    gaiaformers: int = 0
    tracks: tuple[int, ...] = (0, 0, 0, 0, 0, 0)
    tech_tiles: int = 0
    advanced_tech_tiles: int = 0
    covered_tech_tiles: int = 0
    knowledge_academies: int = 0
    qic_academies: int = 0
    used_qic_academy_action: bool = False
    used_standard_tech_action: bool = False
    used_advanced_tech_actions: int = 0
    used_booster_action: bool = False
    federation_tokens: int = 0
    federation_keys: int = 0
    board_federations: int = 0
    federation_tile_counts: tuple[int, ...] = (0, 0, 0, 0, 0, 0)
    gleens_federation_tokens: int = 0
    satellites: int = 0
    colonized_types: int = 0
    passed: bool = False

    def spend(
        self,
        *,
        credits: int = 0,
        ore: int = 0,
        knowledge: int = 0,
        qic: int = 0,
    ) -> PlayerState:
        if self.credits < credits or self.ore < ore or self.knowledge < knowledge or self.qic < qic:
            raise ValueError("insufficient resources")
        return replace(
            self,
            credits=self.credits - credits,
            ore=self.ore - ore,
            knowledge=self.knowledge - knowledge,
            qic=self.qic - qic,
        )


@dataclass(frozen=True, slots=True)
class GaiaState:
    """Deterministic standard-rules core for neural perfect-information search.

    The state implements the shared Gaia Project rules. Optional out-of-turn
    charging is resolved by a deterministic accept-when-affordable policy, and
    federation building uses a canonical minimum-satellite plan so that both
    mechanics fit a fixed AlphaZero action space.
    """

    player_count: int
    setup_seed: int
    round_number: int
    player_to_move: int
    first_player: int
    next_first_player: int
    players: tuple[PlayerState, ...]
    starting_planets: tuple[tuple[int, ...], ...]
    placement_order: tuple[int, ...]
    placement_step: int
    active_planets: tuple[bool, ...]
    planet_q: tuple[int, ...]
    planet_r: tuple[int, ...]
    planet_source_q: tuple[int, ...]
    planet_source_r: tuple[int, ...]
    planet_source_ids: tuple[int, ...]
    planet_source_catalog: tuple[tuple[int, int, int, int, int], ...]
    planet_sectors: tuple[int, ...]
    sector_tiles: tuple[int, ...]
    sector_rotations: tuple[int, ...]
    sector_centers: tuple[tuple[int, int], ...]
    map_mode: str
    owners: tuple[int, ...]
    buildings: tuple[int, ...]
    terrains: tuple[int, ...]
    gaiaformer_owner: tuple[int, ...]
    federated: tuple[bool, ...]
    booster_owner: tuple[int, ...]
    booster_selection_order: tuple[int, ...]
    booster_selection_step: int
    round_scoring_tiles: tuple[int, ...]
    final_scoring_tiles: tuple[int, ...]
    standard_tech_tiles: tuple[int, ...]
    advanced_tech_tiles: tuple[int, ...]
    terraforming_federation_tile: int
    federation_tile_supply: tuple[int, ...]
    used_power_actions: int = 0
    used_qic_actions: int = 0
    pending_tech_player: int = -1
    pending_advanced_tech: int = -1
    pending_research_player: int = -1
    pending_power_terraform_player: int = -1
    pending_booster_terraform_player: int = -1
    pending_booster_range_player: int = -1

    @classmethod
    def initial(
        cls,
        num_players: int = 2,
        seed: int = 0,
        *,
        faction_indices: tuple[int, ...] | None = None,
        first_player: int | None = None,
        sector_tiles: tuple[int, ...] | None = None,
        sector_rotations: tuple[int, ...] | None = None,
        planet_positions: tuple[tuple[int, int, int], ...] | None = None,
        planet_layout: tuple[tuple[int, int, int, int], ...] | None = None,
        booster_tiles: tuple[int, ...] | None = None,
        round_scoring_tiles: tuple[int, ...] | None = None,
        final_scoring_tiles: tuple[int, ...] | None = None,
        standard_tech_tiles: tuple[int, ...] | None = None,
        advanced_tech_tiles: tuple[int, ...] | None = None,
        terraforming_federation_tile: int | None = None,
        map_mode: str = "bga-random",
    ) -> GaiaState:
        if not 2 <= num_players <= 4:
            raise ValueError("GaiaState supports two to four players")
        setup = generate_setup(
            num_players,
            seed,
            faction_boards=FACTION_BOARDS,
            faction_homes=tuple(int(faction.home) for faction in FACTIONS),
            faction_starting_structures=tuple(
                faction.starting_structures for faction in FACTIONS
            ),
            faction_places_last=tuple(faction.places_last for faction in FACTIONS),
            faction_indices=faction_indices,
            first_player=first_player,
            sector_tiles=sector_tiles,
            sector_rotations=sector_rotations,
            planet_positions=planet_positions,
            planet_layout=planet_layout,
            booster_tiles=booster_tiles,
            round_scoring_tiles=round_scoring_tiles,
            final_scoring_tiles=final_scoring_tiles,
            standard_tech_tiles=standard_tech_tiles,
            advanced_tech_tiles=advanced_tech_tiles,
            terraforming_federation_tile=terraforming_federation_tile,
            map_mode=map_mode,
        )
        owners = [-1] * N
        buildings = [Building.EMPTY] * N
        players: list[PlayerState] = []
        for player in range(num_players):
            faction_index = setup.faction_indices[player]
            players.append(cls._base_player_state(faction_index))

        first = setup.first_player
        state = cls(
            player_count=num_players,
            setup_seed=seed,
            round_number=0,
            player_to_move=setup.placement_order[0],
            first_player=first,
            next_first_player=-1,
            players=tuple(players),
            starting_planets=setup.starting_planets,
            placement_order=setup.placement_order,
            placement_step=0,
            active_planets=setup.active_planets,
            planet_q=setup.planet_q,
            planet_r=setup.planet_r,
            planet_source_q=setup.planet_source_q,
            planet_source_r=setup.planet_source_r,
            planet_source_ids=setup.planet_source_ids,
            planet_source_catalog=setup.planet_source_catalog,
            planet_sectors=setup.planet_sectors,
            sector_tiles=setup.sector_tiles,
            sector_rotations=setup.sector_rotations,
            sector_centers=setup.sector_centers,
            map_mode=setup.map_mode,
            owners=tuple(owners),
            buildings=tuple(int(value) for value in buildings),
            terrains=setup.terrains,
            gaiaformer_owner=tuple([-1] * N),
            federated=tuple([False] * N),
            booster_owner=setup.booster_owner,
            booster_selection_order=tuple(
                (first - offset - 1) % num_players for offset in range(num_players)
            ),
            booster_selection_step=0,
            round_scoring_tiles=setup.round_scoring_tiles,
            final_scoring_tiles=setup.final_scoring_tiles,
            standard_tech_tiles=setup.standard_tech_tiles,
            advanced_tech_tiles=setup.advanced_tech_tiles,
            terraforming_federation_tile=setup.terraforming_federation_tile,
            federation_tile_supply=tuple(
                2 if tile == setup.terraforming_federation_tile else 3
                for tile in range(len(FEDERATION_TILES))
            ),
        )
        initialized_players = tuple(
            state._advance_research(
                player,
                info,
                FACTIONS[info.faction].start_track,
                score_round=False,
            )
            for player, info in enumerate(state.players)
        )
        return replace(state, players=initialized_players)

    @staticmethod
    def _base_player_state(faction_index: int) -> PlayerState:
        faction = FACTIONS[faction_index]
        return PlayerState(
            faction=faction_index,
            credits=faction.starting_credits,
            ore=faction.starting_ore,
            knowledge=faction.starting_knowledge,
            qic=faction.starting_qic,
            bowl_one=faction.power[0],
            bowl_two=faction.power[1],
            bowl_three=faction.power[2],
            tracks=(0,) * TRACK_COUNT,
            gaiaformers=0,
            colonized_types=0,
        )

    @property
    def num_players(self) -> int:
        return self.player_count

    @property
    def current_player(self) -> int:
        return self.player_to_move

    @property
    def action_size(self) -> int:
        return ACTION_SIZE

    @property
    def observation_size(self) -> int:
        return int(self.observation().shape[0])

    @property
    def is_terminal(self) -> bool:
        return self.round_number > MAX_ROUNDS

    @property
    def is_starting_placement(self) -> bool:
        return self.round_number == 0 and self.placement_step < len(self.placement_order)

    @property
    def is_booster_selection(self) -> bool:
        return (
            self.round_number == 0
            and self.placement_step >= len(self.placement_order)
            and self.booster_selection_step < len(self.booster_selection_order)
        )

    @property
    def pass_action(self) -> int:
        return PASS_FINAL_ACTION if self.round_number == MAX_ROUNDS else PASS_BOOSTER_OFFSET

    @staticmethod
    def build_action(planet: int) -> int:
        return BUILD_OFFSET + planet

    @staticmethod
    def gaia_action(planet: int) -> int:
        return GAIA_OFFSET + planet

    @staticmethod
    def upgrade_trading_action(planet: int) -> int:
        return UPGRADE_TRADING_OFFSET + planet

    @staticmethod
    def upgrade_lab_action(planet: int) -> int:
        return UPGRADE_LAB_OFFSET + planet

    @staticmethod
    def upgrade_pi_action(planet: int) -> int:
        return UPGRADE_PI_OFFSET + planet

    @staticmethod
    def upgrade_academy_action(planet: int) -> int:
        return UPGRADE_ACADEMY_OFFSET + planet

    @staticmethod
    def upgrade_qic_academy_action(planet: int) -> int:
        return UPGRADE_QIC_ACADEMY_OFFSET + planet

    @staticmethod
    def research_action(track: Track | int) -> int:
        return RESEARCH_OFFSET + int(track)

    @staticmethod
    def power_action(power_action: PowerAction | int) -> int:
        return POWER_OFFSET + int(power_action)

    @staticmethod
    def tech_action(track: Track | int) -> int:
        return TECH_OFFSET + int(track)

    @staticmethod
    def federation_action(tile: int) -> int:
        return FEDERATION_OFFSET + int(tile)

    @staticmethod
    def pass_booster_action(booster: int) -> int:
        return PASS_BOOSTER_OFFSET + booster

    def legal_actions(self) -> tuple[int, ...]:
        if self.is_terminal:
            return ()
        player = self.player_to_move
        info = self.players[player]
        if self.is_starting_placement:
            home = FACTIONS[info.faction].home
            return tuple(
                self.build_action(planet)
                for planet in range(N)
                if self.active_planets[planet]
                and self.owners[planet] == -1
                and Terrain(self.terrains[planet]) == home
            )
        if self.is_booster_selection:
            return tuple(
                self.pass_booster_action(booster)
                for booster, owner in enumerate(self.booster_owner)
                if owner == -1
            )
        if self.pending_advanced_tech >= 0:
            return tuple(
                self.tech_action(space)
                for space, tile in enumerate(self.standard_tech_tiles)
                if info.tech_tiles & (1 << tile)
                and not info.covered_tech_tiles & (1 << tile)
            )
        if self.pending_research_player >= 0:
            return tuple(
                self.research_action(track)
                for track in Track
                if self._can_player_advance(player, track)
            )
        if self.pending_power_terraform_player >= 0:
            return tuple(
                self.build_action(planet)
                for planet in range(N)
                if self._can_build_mine(player, planet, free_steps=2)
            )
        if self.pending_booster_terraform_player >= 0:
            return tuple(
                self.build_action(planet)
                for planet in range(N)
                if self._can_build_mine(player, planet, free_steps=1)
            )
        if self.pending_booster_range_player >= 0:
            actions: list[int] = [
                self.build_action(planet)
                for planet in range(N)
                if self._can_build_mine(player, planet, range_bonus=3)
            ]
            actions.extend(
                self.gaia_action(planet)
                for planet in range(N)
                if self.active_planets[planet]
                and Terrain(self.terrains[planet]) == Terrain.TRANSDIM
                and self.owners[planet] == -1
                and self.gaiaformer_owner[planet] == -1
                and info.gaiaformers > 0
                and self._cycle_power(info) >= self._gaia_cost(info)
                and self._is_reachable(player, planet, range_bonus=3)
            )
            return tuple(actions)
        if self.pending_tech_player >= 0:
            return self._legal_technology_actions(player)

        actions: list[int] = []
        has_tech_choice = self._has_tech_choice(player)
        for planet in range(N):
            if not self.active_planets[planet]:
                continue
            terrain = Terrain(self.terrains[planet])
            if self.owners[planet] == -1 and terrain != Terrain.TRANSDIM:
                credits, ore, qic = self._build_cost(player, planet)
                if (
                    self._building_count(player, Building.MINE) < MAX_BUILDINGS[Building.MINE]
                    and info.credits >= credits
                    and info.ore >= ore
                    and info.qic >= qic
                    and self._can_colonize(player, planet)
                ):
                    actions.append(self.build_action(planet))
            if (
                terrain == Terrain.TRANSDIM
                and self.owners[planet] == -1
                and self.gaiaformer_owner[planet] == -1
                and info.gaiaformers > 0
                and self._cycle_power(info) >= self._gaia_cost(info)
                and self._is_reachable(player, planet)
            ):
                actions.append(self.gaia_action(planet))
            if self.owners[planet] != player:
                continue
            level = Building(self.buildings[planet])
            if level == Building.MINE and self._building_count(player, Building.TRADING_STATION) < 4:
                credits = 3 if self._has_nearby_opponent(player, planet) else 6
                if info.credits >= credits and info.ore >= 2:
                    actions.append(self.upgrade_trading_action(planet))
            elif level == Building.TRADING_STATION:
                if (
                    self._building_count(player, Building.RESEARCH_LAB) < 3
                    and info.credits >= 5
                    and info.ore >= 3
                    and has_tech_choice
                ):
                    actions.append(self.upgrade_lab_action(planet))
                if self._building_count(player, Building.PLANETARY_INSTITUTE) < 1 and info.credits >= 6 and info.ore >= 4:
                    actions.append(self.upgrade_pi_action(planet))
            elif level == Building.RESEARCH_LAB:
                if (
                    self._building_count(player, Building.ACADEMY) < 2
                    and info.credits >= 6
                    and info.ore >= 6
                    and has_tech_choice
                ):
                    if info.knowledge_academies < 1:
                        actions.append(self.upgrade_academy_action(planet))
                    if info.qic_academies < 1:
                        actions.append(self.upgrade_qic_academy_action(planet))

        for track in Track:
            if info.knowledge >= 4 and self._can_player_advance(player, track):
                actions.append(self.research_action(track))
        for power_action in PowerAction:
            cost = self._power_action_cost(player, power_action)
            has_target = (
                power_action != PowerAction.TERRAFORM_TWO
                or any(
                    self._can_build_mine(player, planet, free_steps=2)
                    for planet in range(N)
                )
            )
            if (
                info.bowl_three >= cost
                and not self.used_power_actions & (1 << power_action)
                and has_target
            ):
                actions.append(self.power_action(power_action))
        if self._federation_plan(player) is not None:
            actions.extend(
                self.federation_action(tile)
                for tile, count in enumerate(self.federation_tile_supply)
                if count > 0
            )
        if info.qic_academies and not info.used_qic_academy_action:
            actions.append(QIC_ACADEMY_ACTION)
        if self._has_active_standard_tech(info, 8) and not info.used_standard_tech_action:
            actions.append(STANDARD_TECH_ACTION)
        actions.extend(
            ADVANCED_TECH_ACTION_OFFSET + tile
            for tile in range(ADVANCED_TECH_SPECIAL_COUNT)
            if info.advanced_tech_tiles & (1 << tile)
            and not info.used_advanced_tech_actions & (1 << tile)
        )
        if (
            not self.used_qic_actions & (1 << QicAction.TECH)
            and info.qic >= 4
            and has_tech_choice
        ):
            actions.append(QIC_TECH_ACTION)
        if (
            not self.used_qic_actions & (1 << QicAction.FEDERATION_REWARD)
            and info.qic >= 3
        ):
            actions.extend(
                QIC_FEDERATION_ACTION_OFFSET + tile
                for tile, count in enumerate(info.federation_tile_counts)
                if count > 0
            )
            if info.gleens_federation_tokens:
                actions.append(
                    QIC_FEDERATION_ACTION_OFFSET + len(FEDERATION_TILES)
                )
        if (
            not self.used_qic_actions & (1 << QicAction.PLANET_TYPES)
            and info.qic >= 2
        ):
            actions.append(QIC_PLANET_TYPES_ACTION)
        booster = self._player_booster(player)
        if not info.used_booster_action:
            if booster == 0 and any(
                self._can_build_mine(player, planet, free_steps=1)
                for planet in range(N)
            ):
                actions.append(BOOSTER_TERRAFORM_ACTION)
            elif booster == 1 and any(
                self._can_build_mine(player, planet, range_bonus=3)
                or (
                    self.active_planets[planet]
                    and Terrain(self.terrains[planet]) == Terrain.TRANSDIM
                    and self.owners[planet] == -1
                    and self.gaiaformer_owner[planet] == -1
                    and info.gaiaformers > 0
                    and self._cycle_power(info) >= self._gaia_cost(info)
                    and self._is_reachable(player, planet, range_bonus=3)
                )
                for planet in range(N)
            ):
                actions.append(BOOSTER_RANGE_ACTION)
        if self.round_number == MAX_ROUNDS:
            actions.append(PASS_FINAL_ACTION)
        else:
            actions.extend(
                self.pass_booster_action(booster)
                for booster, owner in enumerate(self.booster_owner)
                if owner == -1
            )
        return tuple(actions)

    def legal_action_mask(self) -> BoolArray:
        mask = np.zeros(ACTION_SIZE, dtype=np.bool_)
        if not self.is_terminal:
            mask[list(self.legal_actions())] = True
        return mask

    def apply(self, action: int) -> GaiaState:
        if self.is_terminal:
            raise ValueError("cannot act in a terminal state")
        if action not in self.legal_actions():
            raise ValueError(f"illegal action {action}: {self.describe_action(action)}")
        if self.pending_advanced_tech >= 0:
            return self._apply_advanced_tech_cover(action - TECH_OFFSET)._advance_turn()
        if self.pending_research_player >= 0:
            return self._apply_free_research(action - RESEARCH_OFFSET)._advance_turn()
        if self.pending_power_terraform_player >= 0:
            return replace(
                self._apply_build(action - BUILD_OFFSET, free_steps=2),
                pending_power_terraform_player=-1,
            )._advance_turn()
        if self.pending_booster_terraform_player >= 0:
            return replace(
                self._apply_build(action - BUILD_OFFSET, free_steps=1),
                pending_booster_terraform_player=-1,
            )._advance_turn()
        if self.pending_booster_range_player >= 0:
            if BUILD_OFFSET <= action < GAIA_OFFSET:
                state = self._apply_build(action - BUILD_OFFSET)
            else:
                state = self._apply_gaia(action - GAIA_OFFSET)
            return replace(state, pending_booster_range_player=-1)._advance_turn()
        if self.pending_tech_player >= 0:
            return self._apply_tech(action - TECH_OFFSET)._advance_turn()
        if self.is_starting_placement and BUILD_OFFSET <= action < GAIA_OFFSET:
            return self._apply_starting_placement(action - BUILD_OFFSET)
        if self.is_booster_selection and PASS_BOOSTER_OFFSET <= action < PASS_FINAL_ACTION:
            return self._apply_initial_booster(action - PASS_BOOSTER_OFFSET)
        if action == PASS_FINAL_ACTION or PASS_BOOSTER_OFFSET <= action < PASS_BOOSTER_OFFSET + BOOSTER_COUNT:
            booster = -1 if action == PASS_FINAL_ACTION else action - PASS_BOOSTER_OFFSET
            return self._apply_pass(booster)
        if BUILD_OFFSET <= action < GAIA_OFFSET:
            state = self._apply_build(action - BUILD_OFFSET)
        elif GAIA_OFFSET <= action < UPGRADE_TRADING_OFFSET:
            state = self._apply_gaia(action - GAIA_OFFSET)
        elif UPGRADE_TRADING_OFFSET <= action < UPGRADE_LAB_OFFSET:
            state = self._apply_upgrade(action - UPGRADE_TRADING_OFFSET, Building.TRADING_STATION)
        elif UPGRADE_LAB_OFFSET <= action < UPGRADE_PI_OFFSET:
            return self._apply_upgrade(action - UPGRADE_LAB_OFFSET, Building.RESEARCH_LAB)
        elif UPGRADE_PI_OFFSET <= action < UPGRADE_ACADEMY_OFFSET:
            state = self._apply_upgrade(action - UPGRADE_PI_OFFSET, Building.PLANETARY_INSTITUTE)
        elif UPGRADE_ACADEMY_OFFSET <= action < RESEARCH_OFFSET:
            if action < UPGRADE_QIC_ACADEMY_OFFSET:
                return self._apply_upgrade(
                    action - UPGRADE_ACADEMY_OFFSET,
                    Building.ACADEMY,
                    academy_kind="knowledge",
                )
            return self._apply_upgrade(
                action - UPGRADE_QIC_ACADEMY_OFFSET,
                Building.ACADEMY,
                academy_kind="qic",
            )
        elif RESEARCH_OFFSET <= action < POWER_OFFSET:
            state = self._apply_research(action - RESEARCH_OFFSET)
        elif POWER_OFFSET <= action < TECH_OFFSET:
            state = self._apply_power_action(action - POWER_OFFSET)
        elif FEDERATION_OFFSET <= action < PASS_BOOSTER_OFFSET:
            if action == QIC_ACADEMY_ACTION:
                state = self._apply_qic_academy_action()
            elif action == STANDARD_TECH_ACTION:
                state = self._apply_standard_tech_action()
            elif ADVANCED_TECH_ACTION_OFFSET <= action < QIC_TECH_ACTION:
                state = self._apply_advanced_tech_action(
                    action - ADVANCED_TECH_ACTION_OFFSET
                )
            elif action == QIC_TECH_ACTION:
                return self._apply_qic_tech_action()._advance_turn()
            elif QIC_FEDERATION_ACTION_OFFSET <= action < QIC_PLANET_TYPES_ACTION:
                state = self._apply_qic_federation_action(
                    action - QIC_FEDERATION_ACTION_OFFSET
                )
            elif action == QIC_PLANET_TYPES_ACTION:
                state = self._apply_qic_planet_types_action()
            elif action == BOOSTER_TERRAFORM_ACTION:
                state = self._apply_booster_terraform_action()
            elif action == BOOSTER_RANGE_ACTION:
                state = self._apply_booster_range_action()
            else:
                state = self._apply_federation(action - FEDERATION_OFFSET)
        else:
            raise ValueError(f"unknown action {action}")
        return state._advance_turn()

    def _apply_starting_placement(self, planet: int) -> GaiaState:
        player = self.player_to_move
        info = self.players[player]
        home = FACTIONS[info.faction].home
        if not self.active_planets[planet] or self.owners[planet] != -1:
            raise ValueError("starting planet is unavailable")
        if Terrain(self.terrains[planet]) != home:
            raise ValueError("starting planet must match the faction home terrain")

        owners = list(self.owners)
        buildings = list(self.buildings)
        owners[planet] = player
        buildings[planet] = (
            Building.PLANETARY_INSTITUTE
            if FACTIONS[info.faction].starts_with_pi
            else Building.MINE
        )
        starting_planets = list(self.starting_planets)
        starting_planets[player] = (*starting_planets[player], planet)
        players = self._replace_player(
            player,
            replace(info, colonized_types=info.colonized_types | (1 << int(home))),
        )
        next_step = self.placement_step + 1
        state = replace(
            self,
            players=players,
            owners=tuple(owners),
            buildings=tuple(int(value) for value in buildings),
            starting_planets=tuple(starting_planets),
            placement_step=next_step,
        )
        if next_step < len(self.placement_order):
            return replace(state, player_to_move=self.placement_order[next_step])
        return replace(
            state,
            player_to_move=self.booster_selection_order[0],
        )

    def _apply_initial_booster(self, booster: int) -> GaiaState:
        player = self.player_to_move
        boosters = list(self.booster_owner)
        if boosters[booster] != -1:
            raise ValueError("starting booster is unavailable")
        boosters[booster] = player
        next_step = self.booster_selection_step + 1
        state = replace(
            self,
            booster_owner=tuple(boosters),
            booster_selection_step=next_step,
        )
        if next_step < len(self.booster_selection_order):
            return replace(state, player_to_move=self.booster_selection_order[next_step])
        return replace(
            state,
            round_number=1,
            player_to_move=self.first_player,
        )._grant_income()

    def _apply_build(self, planet: int, *, free_steps: int = 0) -> GaiaState:
        player = self.player_to_move
        terrain = Terrain(self.terrains[planet])
        credits, ore, qic = self._build_cost(player, planet, free_steps=free_steps)
        info = self.players[player].spend(credits=credits, ore=ore, qic=qic)
        home = FACTIONS[info.faction].home
        steps = 0 if terrain == Terrain.GAIA else self._terrain_steps(home, terrain)
        info = self._score(info, "mine")
        if steps:
            info = self._score(info, "terraform", steps)
        if terrain == Terrain.GAIA:
            info = self._score(info, "gaia")
            if self._has_active_standard_tech(info, 3):
                info = replace(info, vp=info.vp + 3)
            if FACTIONS[info.faction].name == "Gleens":
                info = replace(info, vp=info.vp + 2)
        if info.advanced_tech_tiles & (1 << 13):
            info = replace(info, vp=info.vp + 3)
        new_type = terrain not in (Terrain.TRANSDIM,) and not info.colonized_types & (1 << int(terrain))
        info = replace(info, colonized_types=info.colonized_types | (1 << int(terrain)))
        if new_type and FACTIONS[info.faction].knowledge_for_new_type and self._has_pi(player):
            info = replace(info, knowledge=min(15, info.knowledge + 3))

        owners = list(self.owners)
        buildings = list(self.buildings)
        gaiaformers = list(self.gaiaformer_owner)
        owners[planet] = player
        buildings[planet] = Building.MINE
        if gaiaformers[planet] == player:
            gaiaformers[planet] = -1
            info = replace(info, gaiaformers=info.gaiaformers + 1)
        state = replace(
            self,
            players=self._replace_player(player, info),
            owners=tuple(owners),
            buildings=tuple(int(value) for value in buildings),
            gaiaformer_owner=tuple(gaiaformers),
        )
        return state._trigger_passive_charge(
            player,
            planet,
            state._structure_power(player, Building.MINE, planet),
        )

    def _apply_gaia(self, planet: int) -> GaiaState:
        player = self.player_to_move
        info = self.players[player]
        cost = self._gaia_cost(info)
        info = self._move_power_to_gaia(info, cost)
        info = replace(info, gaiaformers=info.gaiaformers - 1)
        gaiaformers = list(self.gaiaformer_owner)
        gaiaformers[planet] = player
        return replace(
            self,
            players=self._replace_player(player, info),
            gaiaformer_owner=tuple(gaiaformers),
        )

    def _apply_upgrade(
        self,
        planet: int,
        target: Building,
        *,
        academy_kind: str | None = None,
    ) -> GaiaState:
        player = self.player_to_move
        info = self.players[player]
        if target == Building.TRADING_STATION:
            credits = 3 if self._has_nearby_opponent(player, planet) else 6
            info = info.spend(credits=credits, ore=2)
            info = self._score(info, "trading")
        elif target == Building.RESEARCH_LAB:
            info = info.spend(credits=5, ore=3)
        elif target == Building.PLANETARY_INSTITUTE:
            info = info.spend(credits=6, ore=4)
            info = self._score(info, "big")
            if FACTIONS[info.faction].name == "Gleens":
                info = self._gain_gleens_federation_reward(info)
                info = replace(
                    info,
                    federation_tokens=info.federation_tokens + 1,
                    federation_keys=info.federation_keys + 1,
                    gleens_federation_tokens=info.gleens_federation_tokens + 1,
                )
                info = self._score(info, "federation")
        else:
            info = info.spend(credits=6, ore=6)
            info = self._score(info, "big")
            if academy_kind == "knowledge":
                info = replace(info, knowledge_academies=info.knowledge_academies + 1)
            elif academy_kind == "qic":
                info = replace(info, qic_academies=info.qic_academies + 1)
            else:
                raise ValueError("academy type is required")
        if target == Building.TRADING_STATION and info.advanced_tech_tiles & (1 << 14):
            info = replace(info, vp=info.vp + 3)
        buildings = list(self.buildings)
        buildings[planet] = target
        state = replace(
            self,
            players=self._replace_player(player, info),
            buildings=tuple(int(value) for value in buildings),
            pending_tech_player=player if target in (Building.RESEARCH_LAB, Building.ACADEMY) else -1,
        )
        state = state._trigger_passive_charge(
            player,
            planet,
            state._structure_power(player, target, planet),
        )
        if target in (Building.RESEARCH_LAB, Building.ACADEMY):
            return state
        return state

    def _apply_research(self, track: int) -> GaiaState:
        player = self.player_to_move
        info = self.players[player].spend(knowledge=4)
        info = self._advance_research(player, info, Track(track))
        return replace(self, players=self._replace_player(player, info))

    def _apply_tech(self, space: int) -> GaiaState:
        player = self.player_to_move
        info = self.players[player]
        if space >= STANDARD_TECH_COUNT:
            track = space - STANDARD_TECH_COUNT
            return replace(
                self,
                pending_tech_player=-1,
                pending_advanced_tech=self.advanced_tech_tiles[track],
            )

        tile = self.standard_tech_tiles[space]
        info = self._gain_standard_tech(info, tile)
        state = replace(
            self,
            players=self._replace_player(player, info),
            pending_tech_player=-1,
        )
        if space < TRACK_COUNT and state._can_player_advance(player, space):
            info = state._advance_research(player, info, Track(space))
            return replace(state, players=state._replace_player(player, info))
        if space >= TRACK_COUNT and state._has_research_choice(player):
            return replace(state, pending_research_player=player)
        return state

    def _gain_standard_tech(self, info: PlayerState, tile: int) -> PlayerState:
        info = replace(info, tech_tiles=info.tech_tiles | (1 << tile))
        if tile == 0:
            info = replace(info, ore=min(15, info.ore + 1))
            info = self._gain_qic(info, 1)
        elif tile == 1:
            info = replace(
                info,
                knowledge=min(15, info.knowledge + info.colonized_types.bit_count()),
            )
        elif tile == 2:
            info = replace(info, vp=info.vp + 7)
        return info

    def _apply_advanced_tech_cover(self, space: int) -> GaiaState:
        player = self.player_to_move
        info = self.players[player]
        standard_tile = self.standard_tech_tiles[space]
        advanced_tile = self.pending_advanced_tech
        if not info.tech_tiles & (1 << standard_tile):
            raise ValueError("advanced technology must cover an owned standard tile")
        if info.covered_tech_tiles & (1 << standard_tile):
            raise ValueError("standard technology is already covered")
        info = replace(
            info,
            covered_tech_tiles=info.covered_tech_tiles | (1 << standard_tile),
            advanced_tech_tiles=info.advanced_tech_tiles | (1 << advanced_tile),
            federation_keys=info.federation_keys - 1,
        )
        info = self._gain_advanced_tech_reward(player, info, advanced_tile)
        state = replace(
            self,
            players=self._replace_player(player, info),
            pending_advanced_tech=-1,
        )
        if state._has_research_choice(player):
            return replace(state, pending_research_player=player)
        return state

    def _gain_advanced_tech_reward(
        self,
        player: int,
        info: PlayerState,
        tile: int,
    ) -> PlayerState:
        if tile == 3:
            return replace(info, vp=info.vp + 2 * self._building_count(player, Building.MINE))
        if tile == 4:
            sectors = len({
                self.planet_sectors[planet]
                for planet, owner in enumerate(self.owners)
                if owner == player
            })
            return replace(info, ore=min(15, info.ore + sectors))
        if tile == 5:
            sectors = len({
                self.planet_sectors[planet]
                for planet, owner in enumerate(self.owners)
                if owner == player
            })
            return replace(info, vp=info.vp + 2 * sectors)
        if tile == 6:
            gaia_planets = sum(
                owner == player and terrain == Terrain.GAIA
                for owner, terrain in zip(self.owners, self.terrains, strict=True)
            )
            return replace(info, vp=info.vp + 2 * gaia_planets)
        if tile == 7:
            return replace(info, vp=info.vp + 5 * info.federation_tokens)
        if tile == 8:
            return replace(
                info,
                vp=info.vp + 4 * self._building_count(player, Building.TRADING_STATION),
            )
        return info

    def _apply_free_research(self, track: int) -> GaiaState:
        player = self.player_to_move
        info = self._advance_research(player, self.players[player], Track(track))
        return replace(
            self,
            players=self._replace_player(player, info),
            pending_research_player=-1,
        )

    def _apply_power_action(self, power_action: int) -> GaiaState:
        player = self.player_to_move
        info = self.players[player]
        cost = self._power_action_cost(player, power_action)
        info = self._spend_power(info, cost)
        pending_power_terraform_player = -1
        if power_action == PowerAction.KNOWLEDGE_THREE:
            info = replace(info, knowledge=min(15, info.knowledge + 3))
        elif power_action == PowerAction.TERRAFORM_TWO:
            pending_power_terraform_player = player
        elif power_action == PowerAction.ORE_TWO:
            info = replace(info, ore=min(15, info.ore + 2))
        elif power_action == PowerAction.CREDITS_SEVEN:
            info = replace(info, credits=min(30, info.credits + 7))
        elif power_action == PowerAction.KNOWLEDGE_TWO:
            info = replace(info, knowledge=min(15, info.knowledge + 2))
        elif power_action == PowerAction.ORE_ONE:
            info = replace(info, ore=min(15, info.ore + 1))
        else:
            info = replace(info, bowl_one=info.bowl_one + 2)
        return replace(
            self,
            players=self._replace_player(player, info),
            used_power_actions=self.used_power_actions | (1 << power_action),
            pending_power_terraform_player=pending_power_terraform_player,
        )

    def _apply_qic_academy_action(self) -> GaiaState:
        player = self.player_to_move
        info = self.players[player]
        if not info.qic_academies or info.used_qic_academy_action:
            raise ValueError("Q.I.C. academy action is unavailable")
        info = self._gain_qic(info, 1)
        info = replace(info, used_qic_academy_action=True)
        return replace(self, players=self._replace_player(player, info))

    def _apply_standard_tech_action(self) -> GaiaState:
        player = self.player_to_move
        info = self.players[player]
        if (
            not self._has_active_standard_tech(info, 8)
            or info.used_standard_tech_action
        ):
            raise ValueError("standard technology action is unavailable")
        info, _ = self._charge_power(info, 4)
        info = replace(info, used_standard_tech_action=True)
        return replace(self, players=self._replace_player(player, info))

    def _apply_advanced_tech_action(self, tile: int) -> GaiaState:
        player = self.player_to_move
        info = self.players[player]
        mask = 1 << tile
        if (
            tile >= ADVANCED_TECH_SPECIAL_COUNT
            or not info.advanced_tech_tiles & mask
            or info.used_advanced_tech_actions & mask
        ):
            raise ValueError("advanced technology action is unavailable")
        if tile == 0:
            info = replace(info, credits=min(30, info.credits + 5))
            info = self._gain_qic(info, 1)
        elif tile == 1:
            info = replace(info, ore=min(15, info.ore + 3))
        else:
            info = replace(info, knowledge=min(15, info.knowledge + 3))
        info = replace(
            info,
            used_advanced_tech_actions=info.used_advanced_tech_actions | mask,
        )
        return replace(self, players=self._replace_player(player, info))

    def _apply_qic_tech_action(self) -> GaiaState:
        player = self.player_to_move
        info = self.players[player].spend(qic=4)
        return replace(
            self,
            players=self._replace_player(player, info),
            used_qic_actions=self.used_qic_actions | (1 << QicAction.TECH),
            pending_tech_player=player,
        )

    def _apply_qic_federation_action(self, tile: int) -> GaiaState:
        player = self.player_to_move
        info = self.players[player].spend(qic=3)
        if tile < len(FEDERATION_TILES):
            if not info.federation_tile_counts[tile]:
                raise ValueError("player does not own that federation tile")
            info = self._gain_federation_reward(info, tile)
        else:
            if not info.gleens_federation_tokens:
                raise ValueError("player does not own the Gleens federation tile")
            info = self._gain_gleens_federation_reward(info)
        return replace(
            self,
            players=self._replace_player(player, info),
            used_qic_actions=self.used_qic_actions
            | (1 << QicAction.FEDERATION_REWARD),
        )

    def _apply_qic_planet_types_action(self) -> GaiaState:
        player = self.player_to_move
        info = self.players[player].spend(qic=2)
        info = replace(
            info,
            vp=info.vp + 3 + info.colonized_types.bit_count(),
        )
        return replace(
            self,
            players=self._replace_player(player, info),
            used_qic_actions=self.used_qic_actions
            | (1 << QicAction.PLANET_TYPES),
        )

    def _apply_booster_terraform_action(self) -> GaiaState:
        player = self.player_to_move
        info = self.players[player]
        if self._player_booster(player) != 0 or info.used_booster_action:
            raise ValueError("terraforming booster action is unavailable")
        return replace(
            self,
            players=self._replace_player(
                player,
                replace(info, used_booster_action=True),
            ),
            pending_booster_terraform_player=player,
        )

    def _apply_booster_range_action(self) -> GaiaState:
        player = self.player_to_move
        info = self.players[player]
        if self._player_booster(player) != 1 or info.used_booster_action:
            raise ValueError("range booster action is unavailable")
        return replace(
            self,
            players=self._replace_player(
                player,
                replace(info, used_booster_action=True),
            ),
            pending_booster_range_player=player,
        )

    def _apply_federation(self, reward: int = 0) -> GaiaState:
        player = self.player_to_move
        plan = self._federation_plan(player)
        if plan is None:
            raise ValueError("no legal federation plan")
        planets, satellites = plan
        original = self.players[player]
        if FACTIONS[original.faction].name == "Ivits":
            info = original.spend(qic=satellites)
        else:
            info = self._discard_power(original, satellites)
        info = self._gain_federation_reward(info, reward)
        counts = list(info.federation_tile_counts)
        counts[reward] += 1
        info = replace(
            info,
            federation_tokens=info.federation_tokens + 1,
            federation_keys=info.federation_keys + (reward != 5),
            board_federations=info.board_federations + 1,
            federation_tile_counts=tuple(counts),
            satellites=info.satellites + satellites,
        )
        info = self._score(info, "federation")
        federated = list(self.federated)
        for planet in planets:
            federated[planet] = True
        supply = list(self.federation_tile_supply)
        supply[reward] -= 1
        return replace(
            self,
            players=self._replace_player(player, info),
            federated=tuple(federated),
            federation_tile_supply=tuple(supply),
        )

    def _apply_pass(self, booster: int) -> GaiaState:
        player = self.player_to_move
        info = self.players[player]
        current_booster = self._player_booster(player)
        info = replace(info, vp=info.vp + self._booster_pass_points(player, current_booster), passed=True)
        players = self._replace_player(player, info)
        boosters = list(self.booster_owner)
        if self.round_number < MAX_ROUNDS:
            if current_booster >= 0:
                boosters[current_booster] = -1
            boosters[booster] = player
        next_first = player if self.next_first_player == -1 else self.next_first_player
        state = replace(
            self,
            players=players,
            booster_owner=tuple(boosters),
            next_first_player=next_first,
        )
        if not all(candidate.passed for candidate in players):
            return state._advance_turn()
        if self.round_number == MAX_ROUNDS:
            return replace(state, round_number=MAX_ROUNDS + 1, player_to_move=next_first)
        reset_players = tuple(
            replace(
                candidate,
                passed=False,
                used_qic_academy_action=False,
                used_standard_tech_action=False,
                used_advanced_tech_actions=0,
                used_booster_action=False,
            )
            for candidate in players
        )
        return replace(
            state,
            round_number=self.round_number + 1,
            player_to_move=next_first,
            first_player=next_first,
            next_first_player=-1,
            players=reset_players,
            used_power_actions=0,
            used_qic_actions=0,
        )._grant_income()._gaia_phase()

    def _advance_turn(self) -> GaiaState:
        if (
            self.pending_tech_player >= 0
            or self.pending_advanced_tech >= 0
            or self.pending_research_player >= 0
            or self.pending_power_terraform_player >= 0
            or self.pending_booster_terraform_player >= 0
            or self.pending_booster_range_player >= 0
        ):
            return self
        for offset in range(1, self.num_players + 1):
            candidate = (self.player_to_move + offset) % self.num_players
            if not self.players[candidate].passed:
                return replace(self, player_to_move=candidate)
        raise RuntimeError("no active player after turn")

    def _grant_income(self) -> GaiaState:
        updated: list[PlayerState] = []
        for player, info in enumerate(self.players):
            faction = FACTIONS[info.faction]
            mines = self._building_count(player, Building.MINE)
            trading = self._building_count(player, Building.TRADING_STATION)
            labs = self._building_count(player, Building.RESEARCH_LAB)
            institutes = self._building_count(player, Building.PLANETARY_INSTITUTE)
            economy_credits, economy_ore, economy_charge = ECONOMY_TRACK_INCOME[
                info.tracks[Track.ECONOMY]
            ]
            science_knowledge = SCIENCE_TRACK_INCOME[info.tracks[Track.SCIENCE]]
            booster = self._player_booster(player)
            (
                booster_credits,
                booster_ore,
                booster_knowledge,
                booster_qic,
                booster_charge,
            ) = self._booster_income(booster)
            trading_credits = (0, 3, 7, 11, 16)[trading]
            lab_credits = (0, 3, 7, 12)[labs]
            is_bescods = faction.name == "Bescods"
            credits = min(30, info.credits + (
                (lab_credits if is_bescods else trading_credits)
                + economy_credits
                + booster_credits
                + faction.income_credits
            ))
            mine_ore = 1 + mines - int(mines >= 4)
            ore = min(
                15,
                info.ore + mine_ore + economy_ore + booster_ore + faction.income_ore,
            )
            board_knowledge = trading if is_bescods else labs
            academy_knowledge = info.knowledge_academies * (
                3 if faction.name == "Itars" else 2
            )
            knowledge = min(
                15,
                info.knowledge
                + 1
                + board_knowledge
                + academy_knowledge
                + science_knowledge
                + booster_knowledge
                + faction.income_knowledge,
            )
            info = replace(
                info,
                credits=credits,
                ore=ore,
                knowledge=knowledge,
            )
            info = self._gain_qic(info, booster_qic + faction.income_qic)
            power_tokens = faction.income_power_tokens
            if institutes:
                if faction.name == "Xenos":
                    info = self._gain_qic(info, institutes)
                elif faction.name == "Gleens":
                    info = replace(info, ore=min(15, info.ore + institutes))
                elif faction.name == "Lantids":
                    pass
                elif faction.name in ("Ambas", "Bescods"):
                    power_tokens += 2 * institutes
                else:
                    power_tokens += institutes
            if power_tokens:
                info = replace(info, bowl_one=info.bowl_one + power_tokens)
            charge = institutes * 4 + economy_charge + booster_charge
            if self._has_active_standard_tech(info, 5):
                info = replace(info, ore=min(15, info.ore + 1))
                charge += 1
            if self._has_active_standard_tech(info, 6):
                info = replace(
                    info,
                    credits=min(30, info.credits + 1),
                    knowledge=min(15, info.knowledge + 1),
                )
            if self._has_active_standard_tech(info, 7):
                info = replace(info, credits=min(30, info.credits + 4))
            info, _ = self._charge_power(info, charge)
            updated.append(info)
        return replace(self, players=tuple(updated))

    def _gaia_phase(self) -> GaiaState:
        players: list[PlayerState] = []
        for info in self.players:
            faction = FACTIONS[info.faction]
            if faction.gaia_to_bowl_two:
                info = replace(info, bowl_two=info.bowl_two + info.gaia_power, gaia_power=0)
            else:
                info = replace(info, bowl_one=info.bowl_one + info.gaia_power, gaia_power=0)
            players.append(info)
        terrains = list(self.terrains)
        for planet, owner in enumerate(self.gaiaformer_owner):
            if owner >= 0 and Terrain(terrains[planet]) == Terrain.TRANSDIM:
                terrains[planet] = Terrain.GAIA
        return replace(self, players=tuple(players), terrains=tuple(int(value) for value in terrains))

    def _advance_research(
        self,
        player: int,
        info: PlayerState,
        track: Track,
        *,
        score_round: bool = True,
    ) -> PlayerState:
        if not self._can_player_advance(player, track, info):
            raise ValueError(f"cannot advance {track.name}")
        levels = list(info.tracks)
        old_level = levels[track]
        levels[track] += 1
        info = replace(info, tracks=tuple(levels))
        if old_level == 4:
            info = replace(info, federation_keys=info.federation_keys - 1)
        if old_level == 2:
            info, _ = self._charge_power(info, 3)
        new_level = levels[track]
        if track == Track.TERRAFORMING:
            if new_level in (1, 4):
                info = replace(info, ore=min(15, info.ore + 2))
            elif new_level == 5:
                info = self._gain_federation_reward(
                    info,
                    self.terraforming_federation_tile,
                )
                info = replace(
                    info,
                    federation_tokens=info.federation_tokens + 1,
                    federation_keys=info.federation_keys
                    + (self.terraforming_federation_tile != 5),
                    federation_tile_counts=tuple(
                        count + (index == self.terraforming_federation_tile)
                        for index, count in enumerate(info.federation_tile_counts)
                    ),
                )
                info = self._score(info, "federation")
        elif track == Track.NAVIGATION and new_level in (1, 3):
            info = self._gain_qic(info, 1)
        elif track == Track.ARTIFICIAL_INTELLIGENCE:
            info = self._gain_qic(info, (1, 1, 2, 2, 4)[new_level - 1])
        elif track == Track.GAIA_PROJECT:
            if new_level in (1, 3, 4):
                info = replace(info, gaiaformers=info.gaiaformers + 1)
            elif new_level == 2:
                info = replace(info, bowl_one=info.bowl_one + 3)
            elif new_level == 5:
                gaia_planets = sum(
                    owner == player and terrain == Terrain.GAIA
                    for owner, terrain in zip(self.owners, self.terrains, strict=True)
                )
                info = replace(info, vp=info.vp + 4 + gaia_planets)
        elif track == Track.ECONOMY and new_level == 5:
            info = replace(
                info,
                credits=min(30, info.credits + 6),
                ore=min(15, info.ore + 3),
            )
            info, _ = self._charge_power(info, 6)
        elif track == Track.SCIENCE and new_level == 5:
            info = replace(info, knowledge=min(15, info.knowledge + 9))
        if info.advanced_tech_tiles & (1 << 12):
            info = replace(info, vp=info.vp + 2)
        if score_round:
            info = self._score(info, "research")
        return info

    @staticmethod
    def _gain_qic(info: PlayerState, amount: int) -> PlayerState:
        if amount <= 0:
            return info
        if FACTIONS[info.faction].name == "Gleens" and not info.qic_academies:
            return replace(info, ore=min(15, info.ore + amount))
        return replace(info, qic=info.qic + amount)

    def _gain_federation_reward(self, info: PlayerState, tile: int) -> PlayerState:
        victory_points = (6, 7, 8, 7, 7, 12)[tile]
        info = replace(info, vp=info.vp + victory_points)
        if tile == 0:
            return replace(info, knowledge=min(15, info.knowledge + 2))
        if tile == 1:
            return replace(info, ore=min(15, info.ore + 2))
        if tile == 2:
            return self._gain_qic(info, 1)
        if tile == 3:
            return replace(info, bowl_one=info.bowl_one + 2)
        if tile == 4:
            return replace(info, credits=min(30, info.credits + 6))
        return info

    @staticmethod
    def _gain_gleens_federation_reward(info: PlayerState) -> PlayerState:
        return replace(
            info,
            credits=min(30, info.credits + 2),
            ore=min(15, info.ore + 1),
            knowledge=min(15, info.knowledge + 1),
        )

    @staticmethod
    def _can_advance(info: PlayerState, track: Track | int) -> bool:
        level = info.tracks[int(track)]
        return level < 5 and (level < 4 or info.federation_keys > 0)

    def _can_player_advance(
        self,
        player: int,
        track: Track | int,
        info: PlayerState | None = None,
    ) -> bool:
        info = self.players[player] if info is None else info
        track = Track(track)
        if not self._can_advance(info, track):
            return False
        if (
            FACTIONS[info.faction].name == "Bal T'aks"
            and track == Track.NAVIGATION
            and not self._has_pi(player)
        ):
            return False
        return info.tracks[track] != 4 or not any(
            opponent != player and other.tracks[track] == 5
            for opponent, other in enumerate(self.players)
        )

    @staticmethod
    def _has_active_standard_tech(info: PlayerState, tile: int) -> bool:
        mask = 1 << tile
        return bool(info.tech_tiles & mask and not info.covered_tech_tiles & mask)

    def _has_research_choice(self, player: int) -> bool:
        return any(self._can_player_advance(player, track) for track in Track)

    def _legal_technology_actions(self, player: int) -> tuple[int, ...]:
        info = self.players[player]
        actions = [
            self.tech_action(space)
            for space, tile in enumerate(self.standard_tech_tiles)
            if not info.tech_tiles & (1 << tile)
        ]
        has_cover_tile = any(
            info.tech_tiles & (1 << tile)
            and not info.covered_tech_tiles & (1 << tile)
            for tile in self.standard_tech_tiles
        )
        if info.federation_keys > 0 and has_cover_tile:
            actions.extend(
                self.tech_action(STANDARD_TECH_COUNT + track)
                for track, tile in enumerate(self.advanced_tech_tiles)
                if info.tracks[track] >= 4
                and not any(
                    candidate.advanced_tech_tiles & (1 << tile)
                    for candidate in self.players
                )
            )
        return tuple(actions)

    def _has_tech_choice(self, player: int) -> bool:
        return bool(self._legal_technology_actions(player))

    def _power_action_cost(self, player: int, power_action: PowerAction | int) -> int:
        cost = (7, 5, 4, 4, 4, 3, 3)[int(power_action)]
        info = self.players[player]
        if FACTIONS[info.faction].name == "Nevlas" and self._has_pi(player):
            return (cost + 1) // 2
        return cost

    @staticmethod
    def _charge_power(info: PlayerState, amount: int) -> tuple[PlayerState, int]:
        one, two, three = info.bowl_one, info.bowl_two, info.bowl_three
        charged = 0
        for _ in range(amount):
            if one > 0:
                one -= 1
                two += 1
            elif two > 0:
                two -= 1
                three += 1
            else:
                break
            charged += 1
        return replace(info, bowl_one=one, bowl_two=two, bowl_three=three), charged

    @staticmethod
    def _spend_power(info: PlayerState, amount: int) -> PlayerState:
        if info.bowl_three < amount:
            raise ValueError("insufficient charged power")
        return replace(info, bowl_one=info.bowl_one + amount, bowl_three=info.bowl_three - amount)

    @staticmethod
    def _discard_power(info: PlayerState, amount: int) -> PlayerState:
        if info.bowl_one + info.bowl_two + info.bowl_three < amount:
            raise ValueError("insufficient power tokens")
        one, two, three = info.bowl_one, info.bowl_two, info.bowl_three
        take = min(one, amount)
        one -= take
        amount -= take
        take = min(two, amount)
        two -= take
        amount -= take
        three -= amount
        return replace(info, bowl_one=one, bowl_two=two, bowl_three=three)

    @staticmethod
    def _move_power_to_gaia(info: PlayerState, amount: int) -> PlayerState:
        moved = GaiaState._discard_power(info, amount)
        return replace(moved, gaia_power=info.gaia_power + amount)

    def _trigger_passive_charge(self, acting: int, planet: int, amount: int) -> GaiaState:
        players = list(self.players)
        for opponent, info in enumerate(players):
            if opponent == acting:
                continue
            adjacent = any(
                owner == opponent and self._distance(planet, other) <= 2
                for other, owner in enumerate(self.owners)
            )
            if not adjacent:
                continue
            affordable = min(amount, info.vp + 1)
            charged_info, charged = self._charge_power(info, affordable)
            if charged == 0:
                continue
            charged_info = replace(charged_info, vp=charged_info.vp - max(0, charged - 1))
            faction = FACTIONS[charged_info.faction]
            if faction.passive_power_token and self._has_pi(opponent):
                charged_info = replace(charged_info, bowl_one=charged_info.bowl_one + 1)
            players[opponent] = charged_info
        return replace(self, players=tuple(players))

    def _federation_plan(self, player: int) -> tuple[tuple[int, ...], int] | None:
        info = self.players[player]
        faction = FACTIONS[info.faction]
        is_ivits = faction.name == "Ivits"
        existing = tuple(
            planet
            for planet, owner in enumerate(self.owners)
            if owner == player and self.federated[planet]
        ) if is_ivits else ()
        candidates = [
            planet
            for planet, owner in enumerate(self.owners)
            if owner == player and not self.federated[planet]
        ]
        threshold = (
            7 * (info.board_federations + 1)
            if is_ivits
            else 6 if faction.name == "Xenos" and self._has_pi(player) else 7
        )
        available_satellites = info.qic if faction.name == "Ivits" else self._cycle_power(info)
        best: tuple[tuple[int, int, int], tuple[int, ...], int] | None = None
        minimum_size = 0 if existing else 1
        for size in range(minimum_size, len(candidates) + 1):
            for subset in combinations(candidates, size):
                power = sum(
                    self._structure_power(
                        player,
                        Building(self.buildings[planet]),
                        planet,
                    )
                    for planet in subset
                )
                if existing:
                    power += sum(
                        self._structure_power(
                            player,
                            Building(self.buildings[planet]),
                            planet,
                        )
                        for planet in existing
                    )
                if power < threshold:
                    continue
                satellites = (
                    self._minimum_extension_satellites(existing, subset)
                    if existing
                    else self._minimum_satellites(subset)
                )
                if satellites > available_satellites or info.satellites + satellites > 25:
                    continue
                key = (satellites, power - threshold, size)
                if best is None or key < best[0]:
                    best = (key, (*existing, *subset), satellites)
            if best is not None and best[0][0] == 0:
                break
        return None if best is None else (best[1], best[2])

    def _minimum_extension_satellites(
        self,
        existing: tuple[int, ...],
        additions: tuple[int, ...],
    ) -> int:
        connected = set(existing)
        remaining = set(additions)
        cost = 0
        while remaining:
            distance, target = min(
                (self._distance(source, candidate), candidate)
                for source in connected
                for candidate in remaining
            )
            cost += max(0, distance - 1)
            connected.add(target)
            remaining.remove(target)
        return cost

    def _minimum_satellites(self, planets: tuple[int, ...]) -> int:
        if len(planets) < 2:
            return 0
        connected = {planets[0]}
        remaining = set(planets[1:])
        cost = 0
        while remaining:
            distance, target = min(
                (self._distance(source, candidate), candidate)
                for source in connected
                for candidate in remaining
            )
            cost += max(0, distance - 1)
            connected.add(target)
            remaining.remove(target)
        return cost

    def _can_colonize(
        self,
        player: int,
        planet: int,
        *,
        range_bonus: int = 0,
    ) -> bool:
        if not self._is_reachable(player, planet, range_bonus=range_bonus):
            return False
        reserved = self.gaiaformer_owner[planet]
        return reserved in (-1, player)

    def _is_reachable(
        self,
        player: int,
        destination: int,
        *,
        range_bonus: int = 0,
    ) -> bool:
        reach = (
            (1, 1, 2, 2, 3, 4)[self.players[player].tracks[Track.NAVIGATION]]
            + range_bonus
        )
        return any(
            owner == player and self._distance(source, destination) <= reach
            for source, owner in enumerate(self.owners)
        )

    def _distance(self, source: int, destination: int) -> int:
        return hex_distance(
            self.planet_q[source],
            self.planet_r[source],
            self.planet_q[destination],
            self.planet_r[destination],
        )

    def _build_cost(
        self,
        player: int,
        planet: int,
        *,
        free_steps: int = 0,
    ) -> tuple[int, int, int]:
        info = self.players[player]
        terrain = Terrain(self.terrains[planet])
        if terrain == Terrain.GAIA:
            qic = 0 if self.gaiaformer_owner[planet] == player else 1
            if FACTIONS[info.faction].name == "Gleens" and qic:
                return 2, 2, 0
            return 2, 1, qic
        steps = max(
            0,
            self._terrain_steps(FACTIONS[info.faction].home, terrain) - free_steps,
        )
        ore_per_step = (3, 3, 2, 1, 1, 1)[info.tracks[Track.TERRAFORMING]]
        return 2, 1 + steps * ore_per_step, 0

    def _can_build_mine(
        self,
        player: int,
        planet: int,
        *,
        free_steps: int = 0,
        range_bonus: int = 0,
    ) -> bool:
        if (
            not self.active_planets[planet]
            or self.owners[planet] != -1
            or Terrain(self.terrains[planet]) == Terrain.TRANSDIM
            or self._building_count(player, Building.MINE) >= MAX_BUILDINGS[Building.MINE]
        ):
            return False
        credits, ore, qic = self._build_cost(player, planet, free_steps=free_steps)
        info = self.players[player]
        return (
            info.credits >= credits
            and info.ore >= ore
            and info.qic >= qic
            and self._can_colonize(player, planet, range_bonus=range_bonus)
        )

    @staticmethod
    def _terrain_steps(home: Terrain, destination: Terrain) -> int:
        if int(destination) >= 7:
            return 0
        clockwise = (int(destination) - int(home)) % 7
        return min(clockwise, 7 - clockwise)

    @staticmethod
    def _gaia_cost(info: PlayerState) -> int:
        return (99, 6, 6, 4, 3, 3)[info.tracks[Track.GAIA_PROJECT]]

    @staticmethod
    def _cycle_power(info: PlayerState) -> int:
        return info.bowl_one + info.bowl_two + info.bowl_three

    def _building_count(self, player: int, level: Building) -> int:
        return sum(
            owner == player and building == level
            for owner, building in zip(self.owners, self.buildings, strict=True)
        )

    def _structure_power(
        self,
        player: int,
        building: Building,
        planet: int | None = None,
    ) -> int:
        power = STRUCTURE_POWER[building]
        if (
            building in (Building.PLANETARY_INSTITUTE, Building.ACADEMY)
            and self._has_active_standard_tech(self.players[player], 4)
        ):
            power = 4
        if (
            planet is not None
            and FACTIONS[self.players[player].faction].name == "Bescods"
            and self._has_pi(player)
            and Terrain(self.terrains[planet]) == Terrain.TITANIUM
        ):
            power += 1
        return power

    def _has_pi(self, player: int) -> bool:
        return self._building_count(player, Building.PLANETARY_INSTITUTE) > 0

    def _has_nearby_opponent(self, player: int, planet: int) -> bool:
        return any(
            owner not in (-1, player) and self._distance(planet, other) <= 2
            for other, owner in enumerate(self.owners)
        )

    def _replace_player(self, player: int, info: PlayerState) -> tuple[PlayerState, ...]:
        players = list(self.players)
        players[player] = info
        return tuple(players)

    def _score(self, info: PlayerState, kind: str, amount: int = 1) -> PlayerState:
        tile = ROUND_SCORING_TILES[self.round_scoring_tiles[self.round_number - 1]]
        if tile.kind != kind:
            return info
        return replace(info, vp=info.vp + tile.points * amount)

    def _player_booster(self, player: int) -> int:
        return next((index for index, owner in enumerate(self.booster_owner) if owner == player), -1)

    @staticmethod
    def _booster_income(booster: int) -> tuple[int, int, int, int, int]:
        incomes = (
            (2, 0, 0, 0, 0),
            (0, 0, 0, 0, 2),
            (0, 1, 1, 0, 0),
            (0, 1, 0, 0, 2),
            (2, 0, 0, 1, 0),
            (0, 1, 0, 0, 0),
            (0, 1, 0, 0, 0),
            (0, 0, 1, 0, 0),
            (0, 0, 0, 0, 4),
            (4, 0, 0, 0, 0),
        )
        return incomes[booster] if booster >= 0 else (0, 0, 0, 0, 0)

    def _booster_pass_points(self, player: int, booster: int) -> int:
        points = 0
        if booster == 5:
            points += self._building_count(player, Building.MINE)
        elif booster == 6:
            points += 2 * self._building_count(player, Building.TRADING_STATION)
        elif booster == 7:
            points += 3 * self._building_count(player, Building.RESEARCH_LAB)
        elif booster == 8:
            points += 4 * (
                self._building_count(player, Building.PLANETARY_INSTITUTE)
                + self._building_count(player, Building.ACADEMY)
            )
        elif booster == 9:
            points += sum(
                owner == player and terrain == Terrain.GAIA
                for owner, terrain in zip(self.owners, self.terrains, strict=True)
            )
        info = self.players[player]
        if info.advanced_tech_tiles & (1 << 9):
            points += 3 * info.federation_tokens
        if info.advanced_tech_tiles & (1 << 10):
            points += 3 * self._building_count(player, Building.RESEARCH_LAB)
        if info.advanced_tech_tiles & (1 << 11):
            points += info.colonized_types.bit_count()
        return points

    def final_scores(self) -> tuple[float, ...]:
        base: list[float] = []
        for player, info in enumerate(self.players):
            research_points = sum(max(0, level - 2) * 4 for level in info.tracks)
            resources = (info.credits + info.ore + info.knowledge) // 3
            base.append(float(info.vp + research_points + resources))
        awards = [0.0] * self.num_players
        for tile in self.final_scoring_tiles:
            ranking = self._ranking_awards(
                [self._final_scoring_metric(player, tile) for player in range(self.num_players)],
                FINAL_SCORING_NEUTRAL[tile] if self.num_players == 2 else None,
            )
            for player, points in enumerate(ranking):
                awards[player] += points
        return tuple(base[player] + awards[player] for player in range(self.num_players))

    def _final_scoring_metric(self, player: int, tile: int) -> float:
        if tile == 0:
            return float(sum(
                owner == player and federated
                for owner, federated in zip(self.owners, self.federated, strict=True)
            ))
        if tile == 1:
            return float(sum(owner == player for owner in self.owners))
        if tile == 2:
            return float(self.players[player].colonized_types.bit_count())
        if tile == 3:
            return float(sum(
                owner == player and terrain == Terrain.GAIA
                for owner, terrain in zip(self.owners, self.terrains, strict=True)
            ))
        if tile == 4:
            return float(len({
                self.planet_sectors[index]
                for index, owner in enumerate(self.owners)
                if owner == player
            }))
        return float(self.players[player].satellites)

    def _ranking_awards(
        self,
        values: list[float],
        neutral_value: float | None = None,
    ) -> list[float]:
        awards = (18.0, 12.0, 6.0, 0.0)
        result = [0.0] * self.num_players
        ranked_values = [*values]
        if neutral_value is not None:
            ranked_values.append(float(neutral_value))
        ordered = sorted(
            range(len(ranked_values)),
            key=lambda competitor: ranked_values[competitor],
            reverse=True,
        )
        place = 0
        while place < len(ordered):
            end = place + 1
            while (
                end < len(ordered)
                and ranked_values[ordered[end]] == ranked_values[ordered[place]]
            ):
                end += 1
            award = sum(awards[place:end]) / (end - place)
            for index in range(place, end):
                competitor = ordered[index]
                if competitor < self.num_players:
                    result[competitor] = award
            place = end
        return result

    def returns(self) -> FloatArray:
        if not self.is_terminal:
            raise ValueError("returns are only defined for terminal states")
        scores = self.final_scores()
        values = np.zeros(self.num_players, dtype=np.float32)
        for player in range(self.num_players):
            for opponent in range(self.num_players):
                if player == opponent:
                    continue
                values[player] += float(scores[player] > scores[opponent])
                values[player] -= float(scores[player] < scores[opponent])
            values[player] /= self.num_players - 1
        return values

    def heuristic_values(self) -> FloatArray:
        scores = np.asarray(self.final_scores(), dtype=np.float32)
        centered = scores - float(scores.mean())
        scale = max(18.0, float(np.max(np.abs(centered))))
        return np.tanh(centered / scale).astype(np.float32)

    def observation(self) -> FloatArray:
        values: list[float] = [
            self.round_number / MAX_ROUNDS,
            self.num_players / 4.0,
            float(self.is_starting_placement),
            float(self.is_booster_selection),
            self.booster_selection_step / max(1, self.num_players),
        ]
        values.extend(float(self.player_to_move == player) for player in range(self.num_players))
        values.extend(float(self.first_player == player) for player in range(self.num_players))
        values.extend(float(self.used_power_actions & (1 << action) != 0) for action in PowerAction)
        values.extend(float(self.used_qic_actions & (1 << action) != 0) for action in QicAction)
        for tile in self.round_scoring_tiles:
            values.extend(float(tile == candidate) for candidate in range(len(ROUND_SCORING_TILES)))
        for tile in self.final_scoring_tiles:
            values.extend(float(tile == candidate) for candidate in range(len(FINAL_SCORING_TILES)))
        for tile in self.standard_tech_tiles:
            values.extend(float(tile == candidate) for candidate in range(len(STANDARD_TECH_TILES)))
        for tile in self.advanced_tech_tiles:
            values.extend(float(tile == candidate) for candidate in range(len(ADVANCED_TECH_TILES)))
        values.extend(
            float(self.terraforming_federation_tile == candidate)
            for candidate in range(len(FEDERATION_TILES))
        )
        values.extend(count / 3.0 for count in self.federation_tile_supply)
        values.extend((
            float(self.pending_tech_player >= 0),
            float(self.pending_advanced_tech >= 0),
            float(self.pending_research_player >= 0),
            float(self.pending_power_terraform_player >= 0),
            float(self.pending_booster_terraform_player >= 0),
            float(self.pending_booster_range_player >= 0),
        ))
        values.extend(
            float(self.pending_advanced_tech == tile)
            for tile in range(len(ADVANCED_TECH_TILES))
        )
        for owner in self.booster_owner:
            values.append(float(owner == -2))
            values.append(float(owner == -1))
            values.extend(float(owner == player) for player in range(self.num_players))
        for position in range(10):
            present = position < len(self.sector_tiles)
            tile = self.sector_tiles[position] if present else -1
            rotation = self.sector_rotations[position] if present else -1
            values.append(float(present))
            values.extend(float(tile == candidate) for candidate in range(10))
            values.extend(float(rotation == candidate) for candidate in range(6))
        for player, info in enumerate(self.players):
            booster = self._player_booster(player)
            values.extend((
                info.credits / 30.0,
                info.ore / 15.0,
                info.knowledge / 15.0,
                info.qic / 10.0,
                info.vp / 150.0,
                info.bowl_one / 15.0,
                info.bowl_two / 15.0,
                info.bowl_three / 15.0,
                info.gaia_power / 15.0,
                info.gaiaformers / 3.0,
                info.federation_tokens / 6.0,
                info.federation_keys / 3.0,
                info.gleens_federation_tokens,
                info.board_federations / 6.0,
                info.knowledge_academies,
                info.qic_academies,
                float(info.used_qic_academy_action),
                float(info.used_standard_tech_action),
                float(info.used_booster_action),
                *(float(info.used_advanced_tech_actions & (1 << tile) != 0)
                  for tile in range(ADVANCED_TECH_SPECIAL_COUNT)),
                float(info.passed),
            ))
            values.extend(level / 5.0 for level in info.tracks)
            values.extend(float(info.faction == faction) for faction in range(len(FACTIONS)))
            values.extend(float(booster == candidate) for candidate in range(BOOSTER_COUNT))
            values.extend(
                float(info.tech_tiles & (1 << tech) != 0)
                for tech in range(STANDARD_TECH_COUNT)
            )
            values.extend(
                float(info.covered_tech_tiles & (1 << tech) != 0)
                for tech in range(STANDARD_TECH_COUNT)
            )
            values.extend(
                float(info.advanced_tech_tiles & (1 << tech) != 0)
                for tech in range(len(ADVANCED_TECH_TILES))
            )
            values.extend(count / 6.0 for count in info.federation_tile_counts)
        for planet in range(N):
            values.append(float(self.active_planets[planet]))
            values.extend(float(self.terrains[planet] == terrain) for terrain in range(len(Terrain)))
            values.append(float(self.owners[planet] == -1))
            values.extend(float(self.owners[planet] == player) for player in range(self.num_players))
            values.extend(float(self.buildings[planet] == building) for building in range(len(Building)))
            values.append(float(self.gaiaformer_owner[planet] == -1))
            values.extend(float(self.gaiaformer_owner[planet] == player) for player in range(self.num_players))
            values.append(float(self.federated[planet]))
        return np.asarray(values, dtype=np.float32)

    def describe_action(self, action: int) -> str:
        if BUILD_OFFSET <= action < GAIA_OFFSET:
            if self.is_starting_placement:
                structure = (
                    "planetary institute"
                    if FACTIONS[self.players[self.player_to_move].faction].starts_with_pi
                    else "mine"
                )
                return f"place starting {structure} at planet {action - BUILD_OFFSET}"
            return f"build mine at planet {action - BUILD_OFFSET}"
        if GAIA_OFFSET <= action < UPGRADE_TRADING_OFFSET:
            return f"start Gaia Project at planet {action - GAIA_OFFSET}"
        ranges = (
            (UPGRADE_TRADING_OFFSET, UPGRADE_LAB_OFFSET, "trading station"),
            (UPGRADE_LAB_OFFSET, UPGRADE_PI_OFFSET, "research lab"),
            (UPGRADE_PI_OFFSET, UPGRADE_ACADEMY_OFFSET, "planetary institute"),
            (UPGRADE_ACADEMY_OFFSET, UPGRADE_QIC_ACADEMY_OFFSET, "knowledge academy"),
            (UPGRADE_QIC_ACADEMY_OFFSET, RESEARCH_OFFSET, "Q.I.C. academy"),
        )
        for start, end, target in ranges:
            if start <= action < end:
                return f"upgrade planet {action - start} to {target}"
        if RESEARCH_OFFSET <= action < POWER_OFFSET:
            return f"research {Track(action - RESEARCH_OFFSET).name.lower()}"
        if POWER_OFFSET <= action < TECH_OFFSET:
            return f"power action {PowerAction(action - POWER_OFFSET).name.lower()}"
        if TECH_OFFSET <= action < FEDERATION_OFFSET:
            space = action - TECH_OFFSET
            if space < STANDARD_TECH_COUNT:
                tile = self.standard_tech_tiles[space]
                return f"take standard tech tile {tile} from space {space}"
            track = Track(space - STANDARD_TECH_COUNT)
            tile = self.advanced_tech_tiles[track]
            return f"take advanced tech tile {tile} from {track.name.lower()}"
        if FEDERATION_OFFSET <= action < PASS_BOOSTER_OFFSET:
            if action == QIC_ACADEMY_ACTION:
                return "use Q.I.C. academy action"
            if action == STANDARD_TECH_ACTION:
                return "use standard tech tile 8 action"
            if ADVANCED_TECH_ACTION_OFFSET <= action < QIC_TECH_ACTION:
                tile = action - ADVANCED_TECH_ACTION_OFFSET
                return f"use advanced tech tile {tile} action"
            if action == QIC_TECH_ACTION:
                return "Q.I.C. action: take a tech tile"
            if QIC_FEDERATION_ACTION_OFFSET <= action < QIC_PLANET_TYPES_ACTION:
                tile = action - QIC_FEDERATION_ACTION_OFFSET
                if tile == len(FEDERATION_TILES):
                    return "Q.I.C. action: repeat Gleens federation reward"
                return f"Q.I.C. action: repeat federation tile {tile}"
            if action == QIC_PLANET_TYPES_ACTION:
                return "Q.I.C. action: score planet types"
            if action == BOOSTER_TERRAFORM_ACTION:
                return "use booster action: build a mine with 1 free terraforming step"
            if action == BOOSTER_RANGE_ACTION:
                return "use booster action: build a mine or start Gaia Project with range +3"
            tile = action - FEDERATION_OFFSET
            return f"form federation and take tile {tile}"
        if PASS_BOOSTER_OFFSET <= action < PASS_FINAL_ACTION:
            if self.is_booster_selection:
                return f"take starting booster {action - PASS_BOOSTER_OFFSET}"
            return f"pass and take booster {action - PASS_BOOSTER_OFFSET}"
        if action == PASS_FINAL_ACTION:
            return "pass"
        return f"unknown action {action}"

    def render(self) -> str:
        if self.is_starting_placement:
            lines = [
                f"Starting placement {self.placement_step}/{len(self.placement_order)}"
            ]
            lines[0] += f" | player {self.player_to_move} to place"
        elif self.is_booster_selection:
            lines = [
                f"Starting booster selection {self.booster_selection_step}/{len(self.booster_selection_order)}"
            ]
            lines[0] += f" | player {self.player_to_move} to choose"
        else:
            lines = [f"Round {min(self.round_number, MAX_ROUNDS)}/{MAX_ROUNDS}"]
        if not self.is_terminal and not self.is_starting_placement and not self.is_booster_selection:
            scoring = ROUND_SCORING_TILES[
                self.round_scoring_tiles[self.round_number - 1]
            ]
            lines[0] += (
                f" | player {self.player_to_move} to move"
                f" | scoring {scoring.key}"
            )
        for player, info in enumerate(self.players):
            lines.append(
                f"P{player} {FACTIONS[info.faction].name} C{info.credits} O{info.ore} K{info.knowledge} "
                f"Q{info.qic} VP{info.vp} power={info.bowl_one}/{info.bowl_two}/{info.bowl_three} "
                f"tracks={info.tracks} fed={info.federation_tokens}"
            )
        if self.is_terminal:
            lines.append(f"Final scores: {self.final_scores()}")
        return "\n".join(lines)

    def snapshot(self) -> dict[str, object]:
        current_scoring = None
        if not self.is_terminal and not self.is_starting_placement and not self.is_booster_selection:
            current_scoring = ROUND_SCORING_TILES[
                self.round_scoring_tiles[self.round_number - 1]
            ].key
        return {
            "ruleset": "standard-v9",
            "round": max(0, min(self.round_number, MAX_ROUNDS)),
            "max_rounds": MAX_ROUNDS,
            "phase": (
                "starting_placement"
                if self.is_starting_placement
                else "booster_selection"
                if self.is_booster_selection
                else "terminal" if self.is_terminal else "round"
            ),
            "placement": {
                "active": self.is_starting_placement,
                "step": self.placement_step,
                "total": len(self.placement_order),
                "order": list(self.placement_order),
                "remaining": max(0, len(self.placement_order) - self.placement_step),
            },
            "booster_selection": {
                "active": self.is_booster_selection,
                "step": self.booster_selection_step,
                "total": len(self.booster_selection_order),
                "order": list(self.booster_selection_order),
                "remaining": max(
                    0,
                    len(self.booster_selection_order) - self.booster_selection_step,
                ),
            },
            "round_scoring": current_scoring,
            "current_player": None if self.is_terminal else self.player_to_move,
            "first_player": self.first_player,
            "terminal": self.is_terminal,
            "scores": list(self.final_scores()),
            "players": [
                {
                    "id": player,
                    "faction_id": info.faction,
                    "faction": FACTIONS[info.faction].name,
                    "home_terrain": int(FACTIONS[info.faction].home),
                    "faction_ability": FACTIONS[info.faction].ability,
                    "credits": info.credits,
                    "ore": info.ore,
                    "knowledge": info.knowledge,
                    "qic": info.qic,
                    "vp": info.vp,
                    "power": [info.bowl_one, info.bowl_two, info.bowl_three],
                    "gaia_power": info.gaia_power,
                    "gaiaformers": info.gaiaformers,
                    "gaiaformers_on_board": sum(
                        owner == player for owner in self.gaiaformer_owner
                    ),
                    "structures": {
                        building.name.lower(): {
                            "built": self._building_count(player, building),
                            "supply": maximum
                            - self._building_count(player, building),
                        }
                        for building, maximum in MAX_BUILDINGS.items()
                    },
                    "tracks": list(info.tracks),
                    "satellites": info.satellites,
                    "colonized_types": info.colonized_types,
                    "tech_tiles": [
                        tile
                        for tile in range(STANDARD_TECH_COUNT)
                        if info.tech_tiles & (1 << tile)
                    ],
                    "covered_tech_tiles": [
                        tile
                        for tile in range(STANDARD_TECH_COUNT)
                        if info.covered_tech_tiles & (1 << tile)
                    ],
                    "advanced_tech_tiles": [
                        tile
                        for tile in range(len(ADVANCED_TECH_TILES))
                        if info.advanced_tech_tiles & (1 << tile)
                    ],
                    "knowledge_academies": info.knowledge_academies,
                    "qic_academies": info.qic_academies,
                    "qic_academy_action_used": info.used_qic_academy_action,
                    "standard_tech_action_used": info.used_standard_tech_action,
                    "booster_action_used": info.used_booster_action,
                    "advanced_tech_actions_used": info.used_advanced_tech_actions,
                    "federations": info.federation_tokens,
                    "board_federations": info.board_federations,
                    "federation_keys": info.federation_keys,
                    "gleens_federation_tokens": info.gleens_federation_tokens,
                    "federation_tile_counts": list(info.federation_tile_counts),
                    "booster": self._player_booster(player),
                    "passed": info.passed,
                }
                for player, info in enumerate(self.players)
            ],
            "planets": [
                {
                    "id": index,
                    "q": self.planet_q[index],
                    "r": self.planet_r[index],
                    "source_q": self.planet_source_q[index],
                    "source_r": self.planet_source_r[index],
                    "source_id": self.planet_source_ids[index],
                    "sector": self.planet_sectors[index],
                    "terrain": self.terrains[index],
                    "owner": self.owners[index],
                    "building": Building(self.buildings[index]).name.lower(),
                    "gaiaformer": self.gaiaformer_owner[index],
                    "federated": self.federated[index],
                }
                for index in range(N)
                if self.active_planets[index]
            ],
            "setup": {
                "seed": self.setup_seed,
                "map": {
                    "method": self.map_mode,
                    "sector_count": len(self.sector_tiles),
                    "sectors": [
                        {
                            "position": position,
                            "tile": tile + 1,
                            "side": (
                                "outlined"
                                if self.num_players == 2 and tile + 1 in (5, 6, 7)
                                else "solid"
                            ),
                            "rotation": self.sector_rotations[position] * 60,
                            "q": self.sector_centers[position][0],
                            "r": self.sector_centers[position][1],
                        }
                        for position, tile in enumerate(self.sector_tiles)
                    ],
                    "planet_sources": [
                        {
                            "id": planet,
                            "q": q,
                            "r": r,
                            "terrain": terrain,
                            "sector": sector,
                        }
                        for planet, q, r, terrain, sector in self.planet_source_catalog
                    ],
                },
                "factions": [
                    {
                        "player": player,
                        "id": info.faction,
                        "board": FACTIONS[info.faction].board + 1,
                        "side": FACTION_BOARD_SIDES[info.faction],
                        "name": FACTIONS[info.faction].name,
                        "home_terrain": int(FACTIONS[info.faction].home),
                        "start_track": FACTIONS[info.faction].start_track.name.lower(),
                        "ability": FACTIONS[info.faction].ability,
                        "starting_structures": FACTIONS[info.faction].starting_structures,
                        "starts_with_pi": FACTIONS[info.faction].starts_with_pi,
                        "places_last": FACTIONS[info.faction].places_last,
                        "starting_planets": list(self.starting_planets[player]),
                    }
                    for player, info in enumerate(self.players)
                ],
                "faction_catalog": [
                    self._faction_catalog_entry(faction_id)
                    for faction_id in range(len(FACTIONS))
                ],
                "boosters": [
                    {
                        "id": booster,
                        "label": BOOSTER_LABELS[booster],
                        "owner": owner,
                    }
                    for booster, owner in enumerate(self.booster_owner)
                    if owner != -2
                ],
                "round_scoring": [
                    {
                        "round": round_index + 1,
                        "id": tile,
                        "key": ROUND_SCORING_TILES[tile].key,
                        "label": ROUND_SCORING_TILES[tile].label,
                        "points": ROUND_SCORING_TILES[tile].points,
                    }
                    for round_index, tile in enumerate(self.round_scoring_tiles)
                ],
                "final_scoring": [
                    {
                        "id": tile,
                        "key": FINAL_SCORING_TILES[tile].key,
                        "label": FINAL_SCORING_TILES[tile].label,
                    }
                    for tile in self.final_scoring_tiles
                ],
                "standard_tech": [
                    {
                        "space": position,
                        "track": Track(position).name.lower() if position < TRACK_COUNT else None,
                        "id": tile,
                        "key": STANDARD_TECH_TILES[tile].key,
                        "label": STANDARD_TECH_TILES[tile].label,
                    }
                    for position, tile in enumerate(self.standard_tech_tiles)
                ],
                "advanced_tech": [
                    {
                        "track": Track(position).name.lower(),
                        "id": tile,
                        "key": ADVANCED_TECH_TILES[tile].key,
                        "label": ADVANCED_TECH_TILES[tile].label,
                    }
                    for position, tile in enumerate(self.advanced_tech_tiles)
                ],
                "terraforming_federation": {
                    "id": self.terraforming_federation_tile,
                    "key": FEDERATION_TILES[self.terraforming_federation_tile].key,
                    "label": FEDERATION_TILES[self.terraforming_federation_tile].label,
                },
                "federation_supply": list(self.federation_tile_supply),
            },
        }

    def _faction_catalog_entry(self, faction_id: int) -> dict[str, object]:
        faction = FACTIONS[faction_id]
        info = self._advance_research(
            0,
            self._base_player_state(faction_id),
            faction.start_track,
            score_round=False,
        )
        return {
            "id": faction_id,
            "board": faction.board + 1,
            "side": FACTION_BOARD_SIDES[faction_id],
            "name": faction.name,
            "home_terrain": int(faction.home),
            "start_track": faction.start_track.name.lower(),
            "ability": faction.ability,
            "starting_power": list(faction.power),
            "starting_credits": info.credits,
            "starting_ore": info.ore,
            "starting_knowledge": info.knowledge,
            "starting_qic": info.qic,
            "starting_gaiaformers": info.gaiaformers,
            "starting_structures": faction.starting_structures,
            "starts_with_pi": faction.starts_with_pi,
            "places_last": faction.places_last,
            "federation_threshold": faction.federation_threshold,
        }


class GaiaHeuristicEvaluator:
    def evaluate(self, state: GaiaState) -> tuple[FloatArray, FloatArray]:
        priors = np.zeros(state.action_size, dtype=np.float32)
        legal = state.legal_actions()
        if not legal:
            return priors, state.returns()
        weights: list[float] = []
        for action in legal:
            score = 0.0
            if BUILD_OFFSET <= action < GAIA_OFFSET:
                score = 1.2
            elif GAIA_OFFSET <= action < UPGRADE_TRADING_OFFSET:
                score = 1.0
            elif UPGRADE_TRADING_OFFSET <= action < RESEARCH_OFFSET:
                score = 1.1
            elif RESEARCH_OFFSET <= action < POWER_OFFSET:
                score = 1.3
            elif POWER_OFFSET <= action < TECH_OFFSET:
                score = 0.9
            elif TECH_OFFSET <= action < FEDERATION_OFFSET:
                score = 1.5
            elif FEDERATION_OFFSET <= action < QIC_ACADEMY_ACTION:
                score = 2.0
            elif action == QIC_ACADEMY_ACTION:
                score = 1.0
            elif action == STANDARD_TECH_ACTION:
                score = 1.0
            elif ADVANCED_TECH_ACTION_OFFSET <= action < PASS_BOOSTER_OFFSET:
                score = 1.2
            elif len(legal) == 1:
                score = 0.5
            else:
                score = -1.5
            weights.append(exp(score))
        total = sum(weights)
        for action, weight in zip(legal, weights, strict=True):
            priors[action] = weight / total
        return priors, state.heuristic_values()
