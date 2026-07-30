from __future__ import annotations

from dataclasses import dataclass, replace
from enum import IntEnum
from math import exp

import numpy as np

from gaiazero.core import BoolArray, FloatArray


class Building(IntEnum):
    EMPTY = 0
    MINE = 1
    TRADING_STATION = 2
    RESEARCH_LAB = 3


class Track(IntEnum):
    TERRAFORMING = 0
    NAVIGATION = 1
    ECONOMY = 2
    SCIENCE = 3


@dataclass(frozen=True, slots=True)
class Planet:
    q: int
    r: int
    terrain: int


# A compact radius-two board. The outer-ring home planets are spaced so that
# two to four players all receive a legal and strategically distinct opening.
PLANETS: tuple[Planet, ...] = (
    Planet(0, 0, 0),
    Planet(1, 0, 1),
    Planet(1, -1, 2),
    Planet(0, -1, 3),
    Planet(-1, 0, 0),
    Planet(-1, 1, 1),
    Planet(0, 1, 2),
    Planet(2, 0, 0),
    Planet(2, -1, 2),
    Planet(2, -2, 3),
    Planet(1, -2, 1),
    Planet(0, -2, 3),
    Planet(-1, -1, 0),
    Planet(-2, 0, 2),
    Planet(-2, 1, 0),
    Planet(-2, 2, 1),
    Planet(-1, 2, 3),
    Planet(0, 2, 1),
    Planet(1, 1, 2),
)

START_PLANETS = (7, 10, 13, 16)
MAX_ROUNDS = 6
MAX_RESOURCE = 30
TRACK_COUNT = len(Track)


def _hex_distance(a: Planet, b: Planet) -> int:
    dq = a.q - b.q
    dr = a.r - b.r
    return (abs(dq) + abs(dr) + abs(dq + dr)) // 2


DISTANCES: tuple[tuple[int, ...], ...] = tuple(
    tuple(_hex_distance(a, b) for b in PLANETS) for a in PLANETS
)


@dataclass(frozen=True, slots=True)
class PlayerState:
    credits: int = 15
    ore: int = 4
    knowledge: int = 3
    vp: int = 10
    tracks: tuple[int, ...] = (0, 0, 0, 0)
    passed: bool = False

    def spend(self, credits: int = 0, ore: int = 0, knowledge: int = 0) -> PlayerState:
        if self.credits < credits or self.ore < ore or self.knowledge < knowledge:
            raise ValueError("insufficient resources")
        return replace(
            self,
            credits=self.credits - credits,
            ore=self.ore - ore,
            knowledge=self.knowledge - knowledge,
        )


@dataclass(frozen=True, slots=True)
class MiniGaiaState:
    """A deterministic Gaia rules slice used to exercise the full AI stack.

    Implemented: 2-4 players, six rounds, income, mine construction with
    navigation/terraforming, two building upgrades, four research tracks,
    passing, round scoring, and final ranking. The representation is immutable,
    which makes tree search safe and easy to test.
    """

    player_count: int
    round_number: int
    player_to_move: int
    first_player: int
    next_first_player: int
    players: tuple[PlayerState, ...]
    owners: tuple[int, ...]
    buildings: tuple[int, ...]

    @classmethod
    def initial(cls, num_players: int = 2, seed: int = 0) -> MiniGaiaState:
        if not 2 <= num_players <= 4:
            raise ValueError("MiniGaia supports two to four players")
        first_player = seed % num_players
        owners = [-1] * len(PLANETS)
        buildings = [Building.EMPTY] * len(PLANETS)
        players = [PlayerState() for _ in range(num_players)]
        for player in range(num_players):
            planet = START_PLANETS[player]
            owners[planet] = player
            buildings[planet] = Building.MINE
        state = cls(
            player_count=num_players,
            round_number=1,
            player_to_move=first_player,
            first_player=first_player,
            next_first_player=-1,
            players=tuple(players),
            owners=tuple(owners),
            buildings=tuple(int(value) for value in buildings),
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
        return len(PLANETS) * 2 + TRACK_COUNT + 1

    @property
    def observation_size(self) -> int:
        return int(self.observation().shape[0])

    @property
    def pass_action(self) -> int:
        return len(PLANETS) * 2 + TRACK_COUNT

    @property
    def is_terminal(self) -> bool:
        return self.round_number > MAX_ROUNDS

    @staticmethod
    def build_action(planet: int) -> int:
        return planet

    @staticmethod
    def upgrade_action(planet: int) -> int:
        return len(PLANETS) + planet

    @staticmethod
    def research_action(track: Track | int) -> int:
        return len(PLANETS) * 2 + int(track)

    def legal_actions(self) -> tuple[int, ...]:
        if self.is_terminal:
            return ()
        player = self.player_to_move
        info = self.players[player]
        actions: list[int] = []

        for planet in range(len(PLANETS)):
            if self.owners[planet] == -1:
                credits, ore = self._build_cost(player, planet)
                if info.credits >= credits and info.ore >= ore and self._is_reachable(player, planet):
                    actions.append(self.build_action(planet))
            elif self.owners[planet] == player:
                level = Building(self.buildings[planet])
                if level == Building.MINE:
                    credits = 3 if self._has_nearby_opponent(player, planet) else 6
                    if info.credits >= credits and info.ore >= 2:
                        actions.append(self.upgrade_action(planet))
                elif level == Building.TRADING_STATION:
                    if info.credits >= 5 and info.ore >= 3:
                        actions.append(self.upgrade_action(planet))

        for track in Track:
            if info.knowledge >= 4 and info.tracks[track] < 5:
                actions.append(self.research_action(track))
        actions.append(self.pass_action)
        return tuple(actions)

    def legal_action_mask(self) -> BoolArray:
        mask = np.zeros(self.action_size, dtype=np.bool_)
        if not self.is_terminal:
            mask[list(self.legal_actions())] = True
        return mask

    def apply(self, action: int) -> MiniGaiaState:
        if self.is_terminal:
            raise ValueError("cannot act in a terminal state")
        if action not in self.legal_actions():
            raise ValueError(f"illegal action {action}: {self.describe_action(action)}")
        if action == self.pass_action:
            return self._apply_pass()
        if action < len(PLANETS):
            state = self._apply_build(action)
        elif action < len(PLANETS) * 2:
            state = self._apply_upgrade(action - len(PLANETS))
        else:
            state = self._apply_research(action - len(PLANETS) * 2)
        return state._advance_turn()

    def _apply_build(self, planet: int) -> MiniGaiaState:
        player = self.player_to_move
        credits, ore = self._build_cost(player, planet)
        info = self.players[player].spend(credits=credits, ore=ore)
        info = replace(info, vp=info.vp + self._round_bonus("build"))
        players = self._replace_player(player, info)
        owners = list(self.owners)
        buildings = list(self.buildings)
        owners[planet] = player
        buildings[planet] = Building.MINE
        return replace(self, players=players, owners=tuple(owners), buildings=tuple(int(x) for x in buildings))

    def _apply_upgrade(self, planet: int) -> MiniGaiaState:
        player = self.player_to_move
        level = Building(self.buildings[planet])
        if level == Building.MINE:
            credits = 3 if self._has_nearby_opponent(player, planet) else 6
            ore = 2
            new_level = Building.TRADING_STATION
        else:
            credits, ore = 5, 3
            new_level = Building.RESEARCH_LAB
        info = self.players[player].spend(credits=credits, ore=ore)
        info = replace(info, vp=info.vp + self._round_bonus("upgrade"))
        buildings = list(self.buildings)
        buildings[planet] = new_level
        return replace(
            self,
            players=self._replace_player(player, info),
            buildings=tuple(int(x) for x in buildings),
        )

    def _apply_research(self, track: int) -> MiniGaiaState:
        player = self.player_to_move
        info = self.players[player].spend(knowledge=4)
        tracks = list(info.tracks)
        tracks[track] += 1
        info = replace(info, tracks=tuple(tracks), vp=info.vp + self._round_bonus("research"))
        return replace(self, players=self._replace_player(player, info))

    def _apply_pass(self) -> MiniGaiaState:
        player = self.player_to_move
        info = replace(self.players[player], passed=True)
        players = self._replace_player(player, info)
        next_first = player if self.next_first_player == -1 else self.next_first_player
        state = replace(self, players=players, next_first_player=next_first)
        if all(candidate.passed for candidate in players):
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
            )._grant_income()
        return state._advance_turn()

    def _advance_turn(self) -> MiniGaiaState:
        for offset in range(1, self.num_players + 1):
            candidate = (self.player_to_move + offset) % self.num_players
            if not self.players[candidate].passed:
                return replace(self, player_to_move=candidate)
        raise RuntimeError("no active player after turn")

    def _grant_income(self) -> MiniGaiaState:
        updated: list[PlayerState] = []
        for player, info in enumerate(self.players):
            mines = self._building_count(player, Building.MINE)
            trading = self._building_count(player, Building.TRADING_STATION)
            labs = self._building_count(player, Building.RESEARCH_LAB)
            economy = info.tracks[Track.ECONOMY]
            science = info.tracks[Track.SCIENCE]
            updated.append(
                replace(
                    info,
                    credits=min(MAX_RESOURCE, info.credits + 2 + 2 * trading + economy),
                    ore=min(15, info.ore + 1 + mines + economy // 2),
                    knowledge=min(15, info.knowledge + 1 + labs + science),
                )
            )
        return replace(self, players=tuple(updated))

    def _replace_player(self, player: int, info: PlayerState) -> tuple[PlayerState, ...]:
        players = list(self.players)
        players[player] = info
        return tuple(players)

    def _building_count(self, player: int, level: Building) -> int:
        return sum(
            owner == player and building == level
            for owner, building in zip(self.owners, self.buildings, strict=True)
        )

    def _is_reachable(self, player: int, destination: int) -> bool:
        navigation = self.players[player].tracks[Track.NAVIGATION]
        reach = 1 + navigation
        return any(
            owner == player and DISTANCES[source][destination] <= reach
            for source, owner in enumerate(self.owners)
        )

    def _has_nearby_opponent(self, player: int, planet: int) -> bool:
        return any(
            owner not in (-1, player) and DISTANCES[planet][other] <= 2
            for other, owner in enumerate(self.owners)
        )

    def _build_cost(self, player: int, planet: int) -> tuple[int, int]:
        home_terrain = player
        destination = PLANETS[planet].terrain
        terrain_steps = min((destination - home_terrain) % 4, (home_terrain - destination) % 4)
        terraforming = self.players[player].tracks[Track.TERRAFORMING]
        ore_per_step = max(1, 3 - terraforming // 2)
        return 2, 1 + terrain_steps * ore_per_step

    def _round_bonus(self, action_kind: str) -> int:
        schedule = (
            {"build": 2, "upgrade": 0, "research": 0},
            {"build": 0, "upgrade": 3, "research": 0},
            {"build": 0, "upgrade": 0, "research": 2},
            {"build": 3, "upgrade": 0, "research": 0},
            {"build": 0, "upgrade": 4, "research": 0},
            {"build": 0, "upgrade": 0, "research": 3},
        )
        return schedule[self.round_number - 1][action_kind]

    def final_scores(self) -> tuple[float, ...]:
        scores: list[float] = []
        for player, info in enumerate(self.players):
            building_points = sum(
                int(building) * 2
                for owner, building in zip(self.owners, self.buildings, strict=True)
                if owner == player
            )
            track_points = sum(level * level for level in info.tracks)
            resource_points = (info.credits + info.ore * 2 + info.knowledge * 2) / 6.0
            scores.append(info.vp + building_points + track_points + resource_points)
        return tuple(scores)

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
        scale = max(12.0, float(np.max(np.abs(centered))))
        return np.tanh(centered / scale).astype(np.float32)

    def observation(self) -> FloatArray:
        values: list[float] = [
            self.round_number / MAX_ROUNDS,
            self.num_players / 4.0,
        ]
        values.extend(float(self.player_to_move == player) for player in range(self.num_players))
        values.extend(float(self.first_player == player) for player in range(self.num_players))

        for player, info in enumerate(self.players):
            values.extend(
                (
                    info.credits / 30.0,
                    info.ore / 15.0,
                    info.knowledge / 15.0,
                    info.vp / 100.0,
                    float(info.passed),
                )
            )
            values.extend(level / 5.0 for level in info.tracks)
            values.extend(float(player == terrain) for terrain in range(4))

        for planet, (owner, building) in enumerate(zip(self.owners, self.buildings, strict=True)):
            values.extend(float(PLANETS[planet].terrain == terrain) for terrain in range(4))
            values.append(float(owner == -1))
            values.extend(float(owner == player) for player in range(self.num_players))
            values.extend(float(building == level) for level in range(4))
        return np.asarray(values, dtype=np.float32)

    def describe_action(self, action: int) -> str:
        if action == self.pass_action:
            return "pass"
        if 0 <= action < len(PLANETS):
            planet = PLANETS[action]
            return f"build mine at {action} ({planet.q},{planet.r})"
        if len(PLANETS) <= action < len(PLANETS) * 2:
            planet = action - len(PLANETS)
            return f"upgrade building at {planet}"
        if len(PLANETS) * 2 <= action < self.pass_action:
            track = Track(action - len(PLANETS) * 2)
            return f"research {track.name.lower()}"
        return f"unknown action {action}"

    def render(self) -> str:
        lines = [f"Round {min(self.round_number, MAX_ROUNDS)}/{MAX_ROUNDS}"]
        if not self.is_terminal:
            lines[0] += f" | player {self.player_to_move} to move"
        for player, info in enumerate(self.players):
            buildings = [
                f"{index}:{Building(level).name[0]}"
                for index, (owner, level) in enumerate(zip(self.owners, self.buildings, strict=True))
                if owner == player
            ]
            lines.append(
                f"P{player} C{info.credits} O{info.ore} K{info.knowledge} VP{info.vp} "
                f"tracks={info.tracks} buildings={','.join(buildings)}"
            )
        if self.is_terminal:
            lines.append(f"Final scores: {self.final_scores()}")
        return "\n".join(lines)


class MiniGaiaHeuristicEvaluator:
    """Fast non-learning baseline for PIMCTS and regression tests."""

    def evaluate(self, state: MiniGaiaState) -> tuple[FloatArray, FloatArray]:
        priors = np.zeros(state.action_size, dtype=np.float32)
        legal = state.legal_actions()
        if not legal:
            return priors, state.returns()
        weights: list[float] = []
        for action in legal:
            score = 0.0
            if action < len(PLANETS):
                score = 0.7 + state._round_bonus("build")
            elif action < len(PLANETS) * 2:
                score = 0.5 + state._round_bonus("upgrade")
            elif action < state.pass_action:
                score = 0.8 + state._round_bonus("research")
            elif len(legal) == 1:
                score = 1.0
            else:
                score = -1.5
            weights.append(exp(score / 2.0))
        total = sum(weights)
        for action, weight in zip(legal, weights, strict=True):
            priors[action] = weight / total
        return priors, state.heuristic_values()
