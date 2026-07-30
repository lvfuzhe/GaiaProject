from __future__ import annotations

from dataclasses import dataclass, replace
from enum import IntEnum
from itertools import combinations
from math import exp

import numpy as np

from gaiazero.core import BoolArray, FloatArray
from gaiazero.game.mini_gaia import DISTANCES, PLANETS


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
    starting_qic: int = 1
    federation_threshold: int = 7
    gaia_to_bowl_two: bool = False
    passive_power_token: bool = False
    knowledge_for_new_type: bool = False


FACTIONS: tuple[FactionSpec, ...] = (
    FactionSpec("Terrans", Terrain.TERRA, Track.GAIA_PROJECT, (4, 4, 0), gaia_to_bowl_two=True),
    FactionSpec("Xenos", Terrain.DESERT, Track.ARTIFICIAL_INTELLIGENCE, (2, 4, 0), federation_threshold=6),
    FactionSpec("Taklons", Terrain.SWAMP, Track.ECONOMY, (2, 4, 0), passive_power_token=True),
    FactionSpec("Geodens", Terrain.VOLCANIC, Track.TERRAFORMING, (2, 4, 0), knowledge_for_new_type=True),
)


PLANET_TERRAINS: tuple[int, ...] = (
    Terrain.TERRA,
    Terrain.OXIDE,
    Terrain.SWAMP,
    Terrain.VOLCANIC,
    Terrain.TITANIUM,
    Terrain.DESERT,
    Terrain.TRANSDIM,
    Terrain.TERRA,
    Terrain.ICE,
    Terrain.TRANSDIM,
    Terrain.DESERT,
    Terrain.GAIA,
    Terrain.OXIDE,
    Terrain.SWAMP,
    Terrain.TRANSDIM,
    Terrain.TITANIUM,
    Terrain.VOLCANIC,
    Terrain.ICE,
    Terrain.OXIDE,
)

START_PLANETS: tuple[tuple[int, int], ...] = ((7, 0), (10, 5), (13, 2), (16, 3))
MAX_ROUNDS = 6
TRACK_COUNT = len(Track)
TECH_COUNT = TRACK_COUNT
BOOSTER_COUNT = 7
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
ROUND_SCORING = ("mine", "research", "terraform", "gaia", "trading", "big")
ROUND_POINTS = {"mine": 2, "research": 2, "terraform": 2, "gaia": 3, "trading": 3, "big": 5}

N = len(PLANETS)
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
    round_number: int
    player_to_move: int
    first_player: int
    next_first_player: int
    players: tuple[PlayerState, ...]
    owners: tuple[int, ...]
    buildings: tuple[int, ...]
    terrains: tuple[int, ...]
    gaiaformer_owner: tuple[int, ...]
    federated: tuple[bool, ...]
    booster_owner: tuple[int, ...]
    used_power_actions: int = 0
    pending_tech_player: int = -1

    @classmethod
    def initial(cls, num_players: int = 2, seed: int = 0) -> GaiaState:
        if not 2 <= num_players <= 4:
            raise ValueError("GaiaState supports two to four players")
        owners = [-1] * N
        buildings = [Building.EMPTY] * N
        players: list[PlayerState] = []
        for player in range(num_players):
            faction = FACTIONS[player]
            tracks = [0] * TRACK_COUNT
            tracks[faction.start_track] = 1
            gaiaformers = 1 if faction.start_track == Track.GAIA_PROJECT else 0
            info = PlayerState(
                faction=player,
                qic=faction.starting_qic,
                bowl_one=faction.power[0],
                bowl_two=faction.power[1],
                bowl_three=faction.power[2],
                tracks=tuple(tracks),
                gaiaformers=gaiaformers,
                colonized_types=1 << int(faction.home),
            )
            players.append(info)
            for planet in START_PLANETS[player]:
                owners[planet] = player
                buildings[planet] = Building.MINE

        boosters = [-1] * BOOSTER_COUNT
        for player in range(num_players):
            boosters[player] = player
        first = seed % num_players
        state = cls(
            player_count=num_players,
            round_number=1,
            player_to_move=first,
            first_player=first,
            next_first_player=-1,
            players=tuple(players),
            owners=tuple(owners),
            buildings=tuple(int(value) for value in buildings),
            terrains=tuple(int(value) for value in PLANET_TERRAINS),
            gaiaformer_owner=tuple([-1] * N),
            federated=tuple([False] * N),
            booster_owner=tuple(boosters),
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
                if not info.tech_tiles & (1 << int(track)) and self._can_advance(info, track)
            )

        actions: list[int] = []
        has_tech_choice = self._has_tech_choice(info)
        for planet in range(N):
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
                if owner in (-1, player)
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
            if info.tech_tiles & (1 << Track.GAIA_PROJECT):
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
        info = replace(info, tech_tiles=info.tech_tiles | (1 << track))
        if track == Track.ARTIFICIAL_INTELLIGENCE:
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
            if info.tech_tiles & (1 << Track.ECONOMY):
                info = replace(info, ore=min(15, info.ore + 1))
                charge += 1
            if info.tech_tiles & (1 << Track.SCIENCE):
                info = replace(info, knowledge=min(15, info.knowledge + 1))
            if info.tech_tiles & (1 << Track.NAVIGATION):
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

    @staticmethod
    def _has_tech_choice(info: PlayerState) -> bool:
        return any(
            not info.tech_tiles & (1 << int(track)) and GaiaState._can_advance(info, track)
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
                owner == opponent and DISTANCES[planet][other] <= 2
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

    @staticmethod
    def _minimum_satellites(planets: tuple[int, ...]) -> int:
        if len(planets) < 2:
            return 0
        connected = {planets[0]}
        remaining = set(planets[1:])
        cost = 0
        while remaining:
            distance, target = min(
                (DISTANCES[source][candidate], candidate)
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
            owner == player and DISTANCES[source][destination] <= reach
            for source, owner in enumerate(self.owners)
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
            owner not in (-1, player) and DISTANCES[planet][other] <= 2
            for other, owner in enumerate(self.owners)
        )

    def _replace_player(self, player: int, info: PlayerState) -> tuple[PlayerState, ...]:
        players = list(self.players)
        players[player] = info
        return tuple(players)

    def _score(self, info: PlayerState, kind: str, amount: int = 1) -> PlayerState:
        if ROUND_SCORING[self.round_number - 1] != kind:
            return info
        return replace(info, vp=info.vp + ROUND_POINTS[kind] * amount)

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
        planet_metric: list[float] = []
        research_metric: list[float] = []
        for player, info in enumerate(self.players):
            research_points = sum(max(0, level - 2) * 4 for level in info.tracks)
            resources = (info.credits + info.ore + info.knowledge + info.qic) // 3
            base.append(float(info.vp + research_points + resources))
            planet_metric.append(float(sum(owner == player for owner in self.owners)))
            research_metric.append(float(sum(info.tracks)))
        planet_awards = self._ranking_awards(planet_metric)
        research_awards = self._ranking_awards(research_metric)
        return tuple(base[player] + planet_awards[player] + research_awards[player] for player in range(self.num_players))

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
        values.extend(float(ROUND_SCORING[self.round_number - 1] == kind) if not self.is_terminal else 0.0 for kind in ROUND_SCORING)
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
            values.extend(float(info.tech_tiles & (1 << tech) != 0) for tech in range(TECH_COUNT))
        for planet in range(N):
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
            lines[0] += f" | player {self.player_to_move} to move | scoring {ROUND_SCORING[self.round_number - 1]}"
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
        return {
            "ruleset": "standard-v2",
            "round": min(self.round_number, MAX_ROUNDS),
            "max_rounds": MAX_ROUNDS,
            "round_scoring": None if self.is_terminal else ROUND_SCORING[self.round_number - 1],
            "current_player": None if self.is_terminal else self.player_to_move,
            "first_player": self.first_player,
            "terminal": self.is_terminal,
            "scores": list(self.final_scores()),
            "players": [
                {
                    "id": player,
                    "faction": FACTIONS[info.faction].name,
                    "credits": info.credits,
                    "ore": info.ore,
                    "knowledge": info.knowledge,
                    "qic": info.qic,
                    "vp": info.vp,
                    "power": [info.bowl_one, info.bowl_two, info.bowl_three],
                    "gaia_power": info.gaia_power,
                    "gaiaformers": info.gaiaformers,
                    "tracks": list(info.tracks),
                    "federations": info.federation_tokens,
                    "passed": info.passed,
                }
                for player, info in enumerate(self.players)
            ],
            "planets": [
                {
                    "id": index,
                    "q": planet.q,
                    "r": planet.r,
                    "terrain": self.terrains[index],
                    "owner": self.owners[index],
                    "building": Building(self.buildings[index]).name.lower(),
                    "gaiaformer": self.gaiaformer_owner[index],
                    "federated": self.federated[index],
                }
                for index, planet in enumerate(PLANETS)
            ],
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
