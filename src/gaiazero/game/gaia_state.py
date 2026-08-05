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
    ORE = 0
    CREDITS = 1
    KNOWLEDGE = 2


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
    federation_threshold: int = 7
    gaia_to_bowl_two: bool = False
    passive_power_token: bool = False
    knowledge_for_new_type: bool = False


FACTIONS: tuple[FactionSpec, ...] = (
    FactionSpec("Terrans", Terrain.TERRA, Track.GAIA_PROJECT, (4, 4, 0), 0, "Gaia power returns to bowl II", gaia_to_bowl_two=True),
    FactionSpec("Lantids", Terrain.TERRA, Track.SCIENCE, (4, 4, 0), 0, "May coexist on colonized planets"),
    FactionSpec("Xenos", Terrain.DESERT, Track.ARTIFICIAL_INTELLIGENCE, (2, 4, 0), 1, "Starts with a third mine; federates at power 6", starting_structures=3, federation_threshold=6),
    FactionSpec("Gleens", Terrain.DESERT, Track.NAVIGATION, (2, 4, 0), 1, "Ore replaces Q.I.C. for Gaia colonization", starting_qic=0),
    FactionSpec("Taklons", Terrain.SWAMP, Track.ECONOMY, (2, 4, 0), 2, "Brainstone strengthens the power cycle", passive_power_token=True),
    FactionSpec("Ambas", Terrain.SWAMP, Track.NAVIGATION, (4, 4, 0), 2, "Planetary institute can swap with a mine"),
    FactionSpec("Hadsch Hallas", Terrain.OXIDE, Track.ECONOMY, (2, 4, 0), 3, "Credits unlock expanded free actions"),
    FactionSpec("Ivits", Terrain.OXIDE, Track.NAVIGATION, (4, 4, 0), 3, "Starts with its planetary institute", starting_structures=1, starts_with_pi=True),
    FactionSpec("Geodens", Terrain.VOLCANIC, Track.TERRAFORMING, (2, 4, 0), 4, "Knowledge for newly colonized planet types", knowledge_for_new_type=True),
    FactionSpec("Bal T'aks", Terrain.VOLCANIC, Track.GAIA_PROJECT, (4, 4, 0), 4, "Gaiaformers can be converted to Q.I.C."),
    FactionSpec("Firaks", Terrain.TITANIUM, Track.SCIENCE, (2, 4, 0), 5, "May downgrade a research lab to research"),
    FactionSpec("Bescods", Terrain.TITANIUM, Track.ECONOMY, (2, 4, 0), 5, "Lowest research areas advance together"),
    FactionSpec("Nevlas", Terrain.ICE, Track.SCIENCE, (2, 4, 0), 6, "Bowl III power counts double for free actions"),
    FactionSpec("Itars", Terrain.ICE, Track.GAIA_PROJECT, (4, 4, 0), 6, "Gaia power can buy technology", starting_qic=0),
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
TECH_COUNT = TRACK_COUNT
STANDARD_TECH_COUNT = 9
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
@dataclass(frozen=True, slots=True)
class TileSpec:
    key: str
    label: str
    kind: str = ""
    points: int = 0


ROUND_SCORING_TILES: tuple[TileSpec, ...] = (
    TileSpec("mine-2a", "Build mines", "mine", 2),
    TileSpec("mine-2b", "Build mines", "mine", 2),
    TileSpec("trading-3a", "Build trading stations", "trading", 3),
    TileSpec("trading-3b", "Build trading stations", "trading", 3),
    TileSpec("terraform-2a", "Terraforming steps", "terraform", 2),
    TileSpec("terraform-2b", "Terraforming steps", "terraform", 2),
    TileSpec("gaia-3a", "Colonize Gaia planets", "gaia", 3),
    TileSpec("gaia-3b", "Colonize Gaia planets", "gaia", 3),
    TileSpec("research-2", "Advance research", "research", 2),
    TileSpec("big-5", "Build PI or academy", "big", 5),
)

FINAL_SCORING_TILES: tuple[TileSpec, ...] = (
    TileSpec("federation-structures", "Structures in federations"),
    TileSpec("structures", "Total structures"),
    TileSpec("planet-types", "Colonized planet types"),
    TileSpec("gaia-planets", "Colonized Gaia planets"),
    TileSpec("sectors", "Colonized sectors"),
    TileSpec("satellites", "Placed satellites"),
)

STANDARD_TECH_TILES: tuple[TileSpec, ...] = (
    TileSpec("ore-income", "Ore income"),
    TileSpec("knowledge-income", "Knowledge income"),
    TileSpec("credits-income", "Credits income"),
    TileSpec("gaia-vp", "Gaia planet VP"),
    TileSpec("power-income", "Power income"),
    TileSpec("qic", "Immediate Q.I.C."),
    TileSpec("mine-vp", "Mine scoring"),
    TileSpec("federation-vp", "Federation scoring"),
    TileSpec("planet-type-vp", "Planet type scoring"),
)

ADVANCED_TECH_TILES: tuple[TileSpec, ...] = tuple(
    TileSpec(f"advanced-{index + 1:02d}", label)
    for index, label in enumerate(
        (
            "Federation VP",
            "Research VP",
            "Mine VP",
            "Trading station VP",
            "Planetary institute VP",
            "Academy VP",
            "Gaia planet VP",
            "Planet type VP",
            "Sector VP",
            "Satellite VP",
            "Power action",
            "Knowledge action",
            "Ore action",
            "Credits action",
            "Q.I.C. action",
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
    "Ore + mine scoring",
    "Knowledge + trading scoring",
    "Credits + power",
    "Credits",
    "Ore + power",
    "Knowledge + power",
    "Credits + power",
    "Navigation action",
    "Terraforming action",
    "Gaia action",
)

N = MAX_PLANETS
BUILD_OFFSET = 0
GAIA_OFFSET = BUILD_OFFSET + N
UPGRADE_TRADING_OFFSET = GAIA_OFFSET + N
UPGRADE_LAB_OFFSET = UPGRADE_TRADING_OFFSET + N
UPGRADE_PI_OFFSET = UPGRADE_LAB_OFFSET + N
UPGRADE_ACADEMY_OFFSET = UPGRADE_PI_OFFSET + N
RESEARCH_OFFSET = UPGRADE_ACADEMY_OFFSET + N
POWER_OFFSET = RESEARCH_OFFSET + TRACK_COUNT
TECH_OFFSET = POWER_OFFSET + POWER_ACTION_COUNT
FEDERATION_ACTION = TECH_OFFSET + TECH_COUNT
PASS_BOOSTER_OFFSET = FEDERATION_ACTION + 1
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
    federation_tokens: int = 0
    federation_keys: int = 0
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
    active_planets: tuple[bool, ...]
    planet_q: tuple[int, ...]
    planet_r: tuple[int, ...]
    planet_sectors: tuple[int, ...]
    sector_tiles: tuple[int, ...]
    sector_rotations: tuple[int, ...]
    sector_centers: tuple[tuple[int, int], ...]
    owners: tuple[int, ...]
    buildings: tuple[int, ...]
    terrains: tuple[int, ...]
    gaiaformer_owner: tuple[int, ...]
    federated: tuple[bool, ...]
    booster_owner: tuple[int, ...]
    round_scoring_tiles: tuple[int, ...]
    final_scoring_tiles: tuple[int, ...]
    standard_tech_tiles: tuple[int, ...]
    advanced_tech_tiles: tuple[int, ...]
    terraforming_federation_tile: int
    used_power_actions: int = 0
    pending_tech_player: int = -1

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
        booster_tiles: tuple[int, ...] | None = None,
        round_scoring_tiles: tuple[int, ...] | None = None,
        final_scoring_tiles: tuple[int, ...] | None = None,
        standard_tech_tiles: tuple[int, ...] | None = None,
        advanced_tech_tiles: tuple[int, ...] | None = None,
        terraforming_federation_tile: int | None = None,
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
            faction_indices=faction_indices,
            first_player=first_player,
            sector_tiles=sector_tiles,
            sector_rotations=sector_rotations,
            booster_tiles=booster_tiles,
            round_scoring_tiles=round_scoring_tiles,
            final_scoring_tiles=final_scoring_tiles,
            standard_tech_tiles=standard_tech_tiles,
            advanced_tech_tiles=advanced_tech_tiles,
            terraforming_federation_tile=terraforming_federation_tile,
        )
        owners = [-1] * N
        buildings = [Building.EMPTY] * N
        players: list[PlayerState] = []
        for player in range(num_players):
            faction_index = setup.faction_indices[player]
            faction = FACTIONS[faction_index]
            tracks = [0] * TRACK_COUNT
            tracks[faction.start_track] = 1
            gaiaformers = 1 if faction.start_track == Track.GAIA_PROJECT else 0
            info = PlayerState(
                faction=faction_index,
                qic=faction.starting_qic,
                bowl_one=faction.power[0],
                bowl_two=faction.power[1],
                bowl_three=faction.power[2],
                tracks=tuple(tracks),
                gaiaformers=gaiaformers,
                colonized_types=1 << int(faction.home),
            )
            players.append(info)
            for planet in setup.starting_planets[player]:
                owners[planet] = player
                buildings[planet] = (
                    Building.PLANETARY_INSTITUTE
                    if faction.starts_with_pi
                    else Building.MINE
                )

        first = setup.first_player
        state = cls(
            player_count=num_players,
            setup_seed=seed,
            round_number=1,
            player_to_move=first,
            first_player=first,
            next_first_player=-1,
            players=tuple(players),
            starting_planets=setup.starting_planets,
            active_planets=setup.active_planets,
            planet_q=setup.planet_q,
            planet_r=setup.planet_r,
            planet_sectors=setup.planet_sectors,
            sector_tiles=setup.sector_tiles,
            sector_rotations=setup.sector_rotations,
            sector_centers=setup.sector_centers,
            owners=tuple(owners),
            buildings=tuple(int(value) for value in buildings),
            terrains=setup.terrains,
            gaiaformer_owner=tuple([-1] * N),
            federated=tuple([False] * N),
            booster_owner=setup.booster_owner,
            round_scoring_tiles=setup.round_scoring_tiles,
            final_scoring_tiles=setup.final_scoring_tiles,
            standard_tech_tiles=setup.standard_tech_tiles,
            advanced_tech_tiles=setup.advanced_tech_tiles,
            terraforming_federation_tile=setup.terraforming_federation_tile,
        )
        return state._grant_income()

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
    def research_action(track: Track | int) -> int:
        return RESEARCH_OFFSET + int(track)

    @staticmethod
    def power_action(power_action: PowerAction | int) -> int:
        return POWER_OFFSET + int(power_action)

    @staticmethod
    def tech_action(track: Track | int) -> int:
        return TECH_OFFSET + int(track)

    @staticmethod
    def pass_booster_action(booster: int) -> int:
        return PASS_BOOSTER_OFFSET + booster

    def legal_actions(self) -> tuple[int, ...]:
        if self.is_terminal:
            return ()
        player = self.player_to_move
        info = self.players[player]
        if self.pending_tech_player >= 0:
            return tuple(
                self.tech_action(track)
                for track in Track
                if not info.tech_tiles & (1 << self.standard_tech_tiles[int(track)])
                and self._can_advance(info, track)
            )

        actions: list[int] = []
        has_tech_choice = self._has_tech_choice(info)
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
                    actions.append(self.upgrade_academy_action(planet))

        for track in Track:
            if info.knowledge >= 4 and self._can_advance(info, track):
                actions.append(self.research_action(track))
        for power_action, cost in enumerate((3, 4, 4)):
            if info.bowl_three >= cost and not self.used_power_actions & (1 << power_action):
                actions.append(self.power_action(power_action))
        if self._federation_plan(player) is not None:
            actions.append(FEDERATION_ACTION)
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
        if TECH_OFFSET <= action < TECH_OFFSET + TECH_COUNT:
            return self._apply_tech(action - TECH_OFFSET)._advance_turn()
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
            return self._apply_upgrade(action - UPGRADE_ACADEMY_OFFSET, Building.ACADEMY)
        elif RESEARCH_OFFSET <= action < POWER_OFFSET:
            state = self._apply_research(action - RESEARCH_OFFSET)
        elif POWER_OFFSET <= action < TECH_OFFSET:
            state = self._apply_power_action(action - POWER_OFFSET)
        elif action == FEDERATION_ACTION:
            state = self._apply_federation()
        else:
            raise ValueError(f"unknown action {action}")
        return state._advance_turn()

    def _apply_build(self, planet: int) -> GaiaState:
        player = self.player_to_move
        terrain = Terrain(self.terrains[planet])
        credits, ore, qic = self._build_cost(player, planet)
        info = self.players[player].spend(credits=credits, ore=ore, qic=qic)
        home = FACTIONS[info.faction].home
        steps = 0 if terrain == Terrain.GAIA else self._terrain_steps(home, terrain)
        info = self._score(info, "mine")
        if steps:
            info = self._score(info, "terraform", steps)
        if terrain == Terrain.GAIA:
            info = self._score(info, "gaia")
            if info.tech_tiles & (1 << 3):
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
        return state._trigger_passive_charge(player, planet, STRUCTURE_POWER[Building.MINE])

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

    def _apply_upgrade(self, planet: int, target: Building) -> GaiaState:
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
        else:
            info = info.spend(credits=6, ore=6)
            info = self._score(info, "big")
        buildings = list(self.buildings)
        buildings[planet] = target
        state = replace(
            self,
            players=self._replace_player(player, info),
            buildings=tuple(int(value) for value in buildings),
            pending_tech_player=player if target in (Building.RESEARCH_LAB, Building.ACADEMY) else -1,
        )
        state = state._trigger_passive_charge(player, planet, STRUCTURE_POWER[target])
        if target in (Building.RESEARCH_LAB, Building.ACADEMY):
            return state
        return state

    def _apply_research(self, track: int) -> GaiaState:
        player = self.player_to_move
        info = self.players[player].spend(knowledge=4)
        info = self._advance_research(info, Track(track))
        info = self._score(info, "research")
        return replace(self, players=self._replace_player(player, info))

    def _apply_tech(self, track: int) -> GaiaState:
        player = self.player_to_move
        info = self.players[player]
        tile = self.standard_tech_tiles[track]
        info = replace(info, tech_tiles=info.tech_tiles | (1 << tile))
        if tile == 5:
            info = replace(info, qic=info.qic + 1)
        info = self._advance_research(info, Track(track))
        return replace(
            self,
            players=self._replace_player(player, info),
            pending_tech_player=-1,
        )

    def _apply_power_action(self, power_action: int) -> GaiaState:
        player = self.player_to_move
        info = self.players[player]
        cost = (3, 4, 4)[power_action]
        info = self._spend_power(info, cost)
        if power_action == PowerAction.ORE:
            info = replace(info, ore=min(15, info.ore + 2))
        elif power_action == PowerAction.CREDITS:
            info = replace(info, credits=min(30, info.credits + 7))
        else:
            info = replace(info, knowledge=min(15, info.knowledge + 2))
        return replace(
            self,
            players=self._replace_player(player, info),
            used_power_actions=self.used_power_actions | (1 << power_action),
        )

    def _apply_federation(self) -> GaiaState:
        player = self.player_to_move
        plan = self._federation_plan(player)
        if plan is None:
            raise ValueError("no legal federation plan")
        planets, satellites = plan
        info = self._discard_power(self.players[player], satellites)
        reward = info.federation_tokens % 3
        if reward == 0:
            info = replace(info, vp=info.vp + 6, knowledge=min(15, info.knowledge + 2))
        elif reward == 1:
            info = replace(info, vp=info.vp + 7, ore=min(15, info.ore + 2))
        else:
            info = replace(info, vp=info.vp + 8, qic=info.qic + 1)
        info = replace(
            info,
            federation_tokens=info.federation_tokens + 1,
            federation_keys=info.federation_keys + 1,
            satellites=info.satellites + satellites,
        )
        federated = list(self.federated)
        for planet in planets:
            federated[planet] = True
        return replace(
            self,
            players=self._replace_player(player, info),
            federated=tuple(federated),
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
        reset_players = tuple(replace(candidate, passed=False) for candidate in players)
        return replace(
            state,
            round_number=self.round_number + 1,
            player_to_move=next_first,
            first_player=next_first,
            next_first_player=-1,
            players=reset_players,
            used_power_actions=0,
        )._grant_income()._gaia_phase()

    def _advance_turn(self) -> GaiaState:
        if self.pending_tech_player >= 0:
            return self
        for offset in range(1, self.num_players + 1):
            candidate = (self.player_to_move + offset) % self.num_players
            if not self.players[candidate].passed:
                return replace(self, player_to_move=candidate)
        raise RuntimeError("no active player after turn")

    def _grant_income(self) -> GaiaState:
        updated: list[PlayerState] = []
        for player, info in enumerate(self.players):
            mines = self._building_count(player, Building.MINE)
            trading = self._building_count(player, Building.TRADING_STATION)
            labs = self._building_count(player, Building.RESEARCH_LAB)
            academies = self._building_count(player, Building.ACADEMY)
            institutes = self._building_count(player, Building.PLANETARY_INSTITUTE)
            economy = info.tracks[Track.ECONOMY]
            science = info.tracks[Track.SCIENCE]
            booster = self._player_booster(player)
            booster_credits, booster_ore, booster_knowledge, booster_charge = self._booster_income(booster)
            credits = min(30, info.credits + 2 + 2 * trading + economy + booster_credits)
            ore = min(15, info.ore + 1 + mines + economy // 2 + booster_ore)
            knowledge = min(15, info.knowledge + 1 + labs + academies + science + booster_knowledge)
            info = replace(info, credits=credits, ore=ore, knowledge=knowledge)
            charge = institutes * 4 + economy + booster_charge
            if info.tech_tiles & (1 << 0):
                info = replace(info, ore=min(15, info.ore + 1))
            if info.tech_tiles & (1 << 1):
                info = replace(info, knowledge=min(15, info.knowledge + 1))
            if info.tech_tiles & (1 << 2):
                info = replace(info, credits=min(30, info.credits + 4))
            if info.tech_tiles & (1 << 4):
                charge += 1
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

    def _advance_research(self, info: PlayerState, track: Track) -> PlayerState:
        if not self._can_advance(info, track):
            raise ValueError(f"cannot advance {track.name}")
        levels = list(info.tracks)
        old_level = levels[track]
        levels[track] += 1
        info = replace(info, tracks=tuple(levels))
        if old_level == 4:
            info = replace(info, federation_keys=info.federation_keys - 1)
        new_level = levels[track]
        if track == Track.ARTIFICIAL_INTELLIGENCE:
            info = replace(info, qic=info.qic + (1, 1, 2, 2, 4)[new_level - 1])
        elif track == Track.GAIA_PROJECT and new_level in (1, 3, 4):
            info = replace(info, gaiaformers=info.gaiaformers + 1)
        elif track == Track.TERRAFORMING and new_level == 5:
            info = replace(info, federation_tokens=info.federation_tokens + 1, federation_keys=info.federation_keys + 1)
        return info

    @staticmethod
    def _can_advance(info: PlayerState, track: Track | int) -> bool:
        level = info.tracks[int(track)]
        return level < 5 and (level < 4 or info.federation_keys > 0)

    def _has_tech_choice(self, info: PlayerState) -> bool:
        return any(
            not info.tech_tiles & (1 << self.standard_tech_tiles[int(track)])
            and GaiaState._can_advance(info, track)
            for track in Track
        )

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
        candidates = [
            planet
            for planet, owner in enumerate(self.owners)
            if owner == player and not self.federated[planet]
        ]
        threshold = FACTIONS[self.players[player].faction].federation_threshold
        available_power = self._cycle_power(self.players[player])
        best: tuple[tuple[int, int, int], tuple[int, ...], int] | None = None
        for size in range(1, len(candidates) + 1):
            for subset in combinations(candidates, size):
                power = sum(STRUCTURE_POWER[Building(self.buildings[planet])] for planet in subset)
                if power < threshold:
                    continue
                satellites = self._minimum_satellites(subset)
                if satellites > available_power or self.players[player].satellites + satellites > 25:
                    continue
                key = (satellites, power - threshold, size)
                if best is None or key < best[0]:
                    best = (key, subset, satellites)
            if best is not None and best[0][0] == 0:
                break
        return None if best is None else (best[1], best[2])

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

    def _can_colonize(self, player: int, planet: int) -> bool:
        if not self._is_reachable(player, planet):
            return False
        reserved = self.gaiaformer_owner[planet]
        return reserved in (-1, player)

    def _is_reachable(self, player: int, destination: int) -> bool:
        reach = (1, 1, 2, 2, 3, 4)[self.players[player].tracks[Track.NAVIGATION]]
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

    def _build_cost(self, player: int, planet: int) -> tuple[int, int, int]:
        info = self.players[player]
        terrain = Terrain(self.terrains[planet])
        if terrain == Terrain.GAIA:
            qic = 0 if self.gaiaformer_owner[planet] == player else 1
            return 2, 1, qic
        steps = self._terrain_steps(FACTIONS[info.faction].home, terrain)
        ore_per_step = (3, 3, 2, 1, 1, 1)[info.tracks[Track.TERRAFORMING]]
        return 2, 1 + steps * ore_per_step, 0

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
    def _booster_income(booster: int) -> tuple[int, int, int, int]:
        incomes = (
            (0, 1, 0, 0),
            (0, 0, 1, 0),
            (2, 0, 0, 2),
            (4, 0, 0, 0),
            (0, 1, 0, 1),
            (0, 0, 1, 1),
            (2, 0, 0, 1),
            (0, 0, 0, 0),
            (0, 0, 0, 0),
            (0, 0, 0, 0),
        )
        return incomes[booster] if booster >= 0 else (0, 0, 0, 0)

    def _booster_pass_points(self, player: int, booster: int) -> int:
        if booster == 0:
            return self._building_count(player, Building.MINE)
        if booster == 1:
            return 2 * self._building_count(player, Building.TRADING_STATION)
        if booster == 4:
            return self.players[player].tracks[Track.NAVIGATION]
        if booster == 5:
            return self.players[player].tracks[Track.GAIA_PROJECT]
        return 0

    def final_scores(self) -> tuple[float, ...]:
        base: list[float] = []
        for player, info in enumerate(self.players):
            research_points = sum(max(0, level - 2) * 4 for level in info.tracks)
            resources = (info.credits + info.ore + info.knowledge + info.qic) // 3
            base.append(float(info.vp + research_points + resources))
        awards = [0.0] * self.num_players
        for tile in self.final_scoring_tiles:
            ranking = self._ranking_awards(
                [self._final_scoring_metric(player, tile) for player in range(self.num_players)]
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

    def _ranking_awards(self, values: list[float]) -> list[float]:
        awards = (18.0, 12.0, 6.0, 0.0)
        result = [0.0] * self.num_players
        ordered = sorted(range(self.num_players), key=lambda player: values[player], reverse=True)
        place = 0
        while place < len(ordered):
            end = place + 1
            while end < len(ordered) and values[ordered[end]] == values[ordered[place]]:
                end += 1
            award = sum(awards[place:end]) / (end - place)
            for index in range(place, end):
                result[ordered[index]] = award
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
        values: list[float] = [self.round_number / MAX_ROUNDS, self.num_players / 4.0]
        values.extend(float(self.player_to_move == player) for player in range(self.num_players))
        values.extend(float(self.first_player == player) for player in range(self.num_players))
        values.extend(float(self.used_power_actions & (1 << action) != 0) for action in PowerAction)
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
                float(info.passed),
            ))
            values.extend(level / 5.0 for level in info.tracks)
            values.extend(float(info.faction == faction) for faction in range(len(FACTIONS)))
            values.extend(float(booster == candidate) for candidate in range(BOOSTER_COUNT))
            values.extend(
                float(info.tech_tiles & (1 << tech) != 0)
                for tech in range(STANDARD_TECH_COUNT)
            )
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
            return f"build mine at planet {action - BUILD_OFFSET}"
        if GAIA_OFFSET <= action < UPGRADE_TRADING_OFFSET:
            return f"start Gaia Project at planet {action - GAIA_OFFSET}"
        ranges = (
            (UPGRADE_TRADING_OFFSET, UPGRADE_LAB_OFFSET, "trading station"),
            (UPGRADE_LAB_OFFSET, UPGRADE_PI_OFFSET, "research lab"),
            (UPGRADE_PI_OFFSET, UPGRADE_ACADEMY_OFFSET, "planetary institute"),
            (UPGRADE_ACADEMY_OFFSET, RESEARCH_OFFSET, "academy"),
        )
        for start, end, target in ranges:
            if start <= action < end:
                return f"upgrade planet {action - start} to {target}"
        if RESEARCH_OFFSET <= action < POWER_OFFSET:
            return f"research {Track(action - RESEARCH_OFFSET).name.lower()}"
        if POWER_OFFSET <= action < TECH_OFFSET:
            return f"power action {PowerAction(action - POWER_OFFSET).name.lower()}"
        if TECH_OFFSET <= action < FEDERATION_ACTION:
            return f"take {Track(action - TECH_OFFSET).name.lower()} tech tile"
        if action == FEDERATION_ACTION:
            return "form federation"
        if PASS_BOOSTER_OFFSET <= action < PASS_FINAL_ACTION:
            return f"pass and take booster {action - PASS_BOOSTER_OFFSET}"
        if action == PASS_FINAL_ACTION:
            return "pass"
        return f"unknown action {action}"

    def render(self) -> str:
        lines = [f"Round {min(self.round_number, MAX_ROUNDS)}/{MAX_ROUNDS}"]
        if not self.is_terminal:
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
        if not self.is_terminal:
            current_scoring = ROUND_SCORING_TILES[
                self.round_scoring_tiles[self.round_number - 1]
            ].key
        return {
            "ruleset": "standard-v3",
            "round": min(self.round_number, MAX_ROUNDS),
            "max_rounds": MAX_ROUNDS,
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
                    "federations": info.federation_tokens,
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
                    "method": "advanced-random-sectors",
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
                        "starting_planets": list(self.starting_planets[player]),
                    }
                    for player, info in enumerate(self.players)
                ],
                "faction_catalog": [
                    {
                        "id": faction_id,
                        "board": faction.board + 1,
                        "side": FACTION_BOARD_SIDES[faction_id],
                        "name": faction.name,
                        "home_terrain": int(faction.home),
                        "start_track": faction.start_track.name.lower(),
                        "ability": faction.ability,
                        "starting_power": list(faction.power),
                        "starting_qic": faction.starting_qic,
                        "starting_structures": faction.starting_structures,
                        "starts_with_pi": faction.starts_with_pi,
                        "federation_threshold": faction.federation_threshold,
                    }
                    for faction_id, faction in enumerate(FACTIONS)
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
            },
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
            elif TECH_OFFSET <= action < FEDERATION_ACTION:
                score = 1.5
            elif action == FEDERATION_ACTION:
                score = 2.0
            elif len(legal) == 1:
                score = 0.5
            else:
                score = -1.5
            weights.append(exp(score))
        total = sum(weights)
        for action, weight in zip(legal, weights, strict=True):
            priors[action] = weight / total
        return priors, state.heuristic_values()
