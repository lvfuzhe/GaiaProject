import unittest
from dataclasses import replace

import numpy as np

from gaiazero.game.gaia_state import (
    ADVANCED_TECH_ACTION_OFFSET,
    ADVANCED_TECH_TILES,
    BRAINSTONE_ACTION,
    BOOSTER_RANGE_ACTION,
    BOOSTER_TERRAFORM_ACTION,
    BOOSTER_COUNT,
    BOOSTER_LABELS,
    FEDERATION_OFFSET,
    FEDERATION_ACTION,
    FEDERATION_TILES,
    FACTIONS,
    FINAL_SCORING_TILES,
    MAX_BUILDINGS,
    ROUND_SCORING_TILES,
    STANDARD_TECH_TILES,
    STANDARD_TECH_COUNT,
    STANDARD_TECH_ACTION,
    TAKLONS_PASSIVE_AFTER_ACTION,
    TAKLONS_PASSIVE_BEFORE_ACTION,
    TERRANS_GAIA_CREDIT_ACTION,
    TERRANS_GAIA_FINISH_ACTION,
    TERRANS_GAIA_KNOWLEDGE_ACTION,
    TERRANS_GAIA_ORE_ACTION,
    TERRANS_GAIA_QIC_ACTION,
    Building,
    GaiaHeuristicEvaluator,
    GaiaState,
    PlayerState,
    PowerAction,
    QIC_ACADEMY_ACTION,
    QIC_FEDERATION_ACTION_OFFSET,
    QIC_PLANET_TYPES_ACTION,
    QIC_TECH_ACTION,
    Terrain,
    Track,
)
from gaiazero.game.gaia_setup import hex_distance
from gaiazero.mcts import PUCTSearch, SearchConfig


def finish_starting_placement(state: GaiaState) -> GaiaState:
    while state.is_starting_placement:
        legal = state.legal_actions()
        if not legal:
            raise AssertionError("starting placement has no legal home planet")
        state = state.apply(legal[0])
    while state.is_booster_selection:
        legal = state.legal_actions()
        if not legal:
            raise AssertionError("starting booster selection has no available booster")
        state = state.apply(legal[0])
    return state


class StandardGaiaRulesTests(unittest.TestCase):
    def test_setup_has_full_research_and_building_model(self) -> None:
        state = GaiaState.initial(4, seed=3)

        self.assertEqual(state.action_size, 559)
        self.assertEqual(len(state.players[0].tracks), 6)
        self.assertEqual(state.first_player, 3)
        self.assertEqual(state.current_player, state.placement_order[0])
        self.assertEqual(sum(state.active_planets), 61)
        self.assertEqual(len(state.sector_tiles), 10)
        self.assertEqual(sum(owner >= 0 for owner in state.owners), 0)
        self.assertEqual(sum(len(planets) for planets in state.starting_planets), 0)
        self.assertEqual(state.round_number, 0)
        self.assertTrue(state.is_starting_placement)
        self.assertEqual(state.snapshot()["ruleset"], "standard-v12")

    def test_random_setup_is_seeded_and_respects_component_counts(self) -> None:
        first = GaiaState.initial(2, seed=19)
        repeated = GaiaState.initial(2, seed=19)
        different = GaiaState.initial(2, seed=20)

        self.assertEqual(first.sector_tiles, repeated.sector_tiles)
        self.assertEqual(first.sector_rotations, repeated.sector_rotations)
        self.assertEqual(first.round_scoring_tiles, repeated.round_scoring_tiles)
        self.assertNotEqual(
            (first.sector_tiles, first.sector_rotations, first.players),
            (different.sector_tiles, different.sector_rotations, different.players),
        )
        self.assertEqual(sum(first.active_planets), 40)
        self.assertEqual(len(first.sector_tiles), 7)
        self.assertEqual(len(set(first.round_scoring_tiles)), 6)
        self.assertEqual(len(set(first.final_scoring_tiles)), 2)
        self.assertEqual(len(set(first.standard_tech_tiles)), 9)
        self.assertEqual(len(set(first.advanced_tech_tiles)), 6)
        self.assertEqual(sum(owner != -2 for owner in first.booster_owner), first.num_players + 3)
        self.assertEqual(len(first.booster_owner), BOOSTER_COUNT)

    def test_faction_starting_qic_matches_initial_setup_rules(self) -> None:
        state = GaiaState.initial(
            2,
            faction_indices=(13, 9),
            first_player=0,
        )

        self.assertEqual(state.players[0].qic, 1)  # Itars
        self.assertEqual(state.players[1].qic, 0)  # Bal T'aks
        catalog = state.snapshot()["setup"]["faction_catalog"]
        self.assertEqual(next(f["starting_qic"] for f in catalog if f["name"] == "Itars"), 1)
        self.assertEqual(next(f["starting_qic"] for f in catalog if f["name"] == "Bal T'aks"), 0)

    def test_ambas_and_ivits_use_default_starting_power(self) -> None:
        state = GaiaState.initial(
            2,
            faction_indices=(5, 7),
            first_player=0,
        )
        self.assertEqual(
            (state.players[0].bowl_one, state.players[0].bowl_two, state.players[0].bowl_three),
            (2, 4, 0),
        )
        self.assertEqual(
            (state.players[1].bowl_one, state.players[1].bowl_two, state.players[1].bowl_three),
            (2, 4, 0),
        )
        catalog = {
            faction["name"]: faction
            for faction in state.snapshot()["setup"]["faction_catalog"]
        }
        self.assertEqual(catalog["Ambas"]["starting_power"], [2, 4, 0])
        self.assertEqual(catalog["Ivits"]["starting_power"], [2, 4, 0])

    def test_initial_research_levels_grant_immediate_bga_rewards(self) -> None:
        state = GaiaState.initial(
            4,
            faction_indices=(8, 2, 0, 4),
            first_player=0,
        )
        geodens, xenos, terrans, taklons = state.players

        self.assertEqual(geodens.tracks[Track.TERRAFORMING], 1)
        self.assertEqual(geodens.ore, 6)
        self.assertEqual(xenos.tracks[Track.ARTIFICIAL_INTELLIGENCE], 1)
        self.assertEqual(xenos.qic, 2)
        self.assertEqual(terrans.tracks[Track.GAIA_PROJECT], 1)
        self.assertEqual(terrans.gaiaformers, 1)
        self.assertEqual(taklons.tracks, (0,) * len(Track))
        self.assertEqual((taklons.credits, taklons.ore), (15, 4))

        gleens = GaiaState.initial(
            2,
            faction_indices=(3, 0),
            first_player=0,
        ).players[0]
        self.assertEqual(gleens.tracks[Track.NAVIGATION], 1)
        self.assertEqual((gleens.ore, gleens.qic), (5, 0))

    def test_taklons_brainstone_starts_in_bowl_one_and_charges_normally(self) -> None:
        state = GaiaState.initial(
            2,
            faction_indices=(4, 0),
            first_player=0,
        )
        taklons = state.players[0]

        self.assertEqual(
            (taklons.bowl_one, taklons.bowl_two, taklons.bowl_three),
            (3, 4, 0),
        )
        self.assertEqual(taklons.brainstone_bowl, 1)
        snapshot = state.snapshot()
        self.assertEqual(snapshot["players"][0]["brainstone_bowl"], 1)
        catalog = next(
            faction
            for faction in snapshot["setup"]["faction_catalog"]
            if faction["name"] == "Taklons"
        )
        self.assertEqual(catalog["starting_power"], [2, 4, 0])
        self.assertEqual(catalog["starting_brainstone_bowl"], 1)

        brainstone_only = replace(
            taklons,
            bowl_one=1,
            bowl_two=0,
            bowl_three=0,
            brainstone_bowl=1,
        )
        charged, amount = GaiaState._charge_power(brainstone_only, 2)
        self.assertEqual(amount, 2)
        self.assertEqual(
            (charged.bowl_one, charged.bowl_two, charged.bowl_three),
            (0, 0, 1),
        )
        self.assertEqual(charged.brainstone_bowl, 3)

    def test_taklons_can_select_brainstone_as_three_power(self) -> None:
        state = finish_starting_placement(GaiaState.initial(
            2,
            faction_indices=(4, 0),
            first_player=0,
        ))
        players = list(state.players)
        players[0] = replace(
            players[0],
            bowl_one=0,
            bowl_two=0,
            bowl_three=2,
            brainstone_bowl=3,
            ore=0,
        )
        state = replace(state, players=tuple(players), player_to_move=0)
        ore_action = state.power_action(PowerAction.ORE_TWO)

        self.assertNotIn(ore_action, state.legal_actions())
        self.assertIn(BRAINSTONE_ACTION, state.legal_actions())
        selected = state.apply(BRAINSTONE_ACTION)
        self.assertEqual(selected.player_to_move, 0)
        self.assertTrue(selected.brainstone_selected)
        self.assertNotIn(BRAINSTONE_ACTION, selected.legal_actions())
        self.assertIn(ore_action, selected.legal_actions())

        after = selected.apply(ore_action)
        taklons = after.players[0]
        self.assertFalse(after.brainstone_selected)
        self.assertEqual(taklons.ore, 2)
        self.assertEqual(
            (taklons.bowl_one, taklons.bowl_two, taklons.bowl_three),
            (2, 0, 0),
        )
        self.assertEqual(taklons.brainstone_bowl, 1)

    def test_taklons_pi_keeps_normal_token_income(self) -> None:
        state = finish_starting_placement(
            GaiaState.initial(2, faction_indices=(4, 0), first_player=0)
        )
        pi_planet = next(
            planet for planet, owner in enumerate(state.owners) if owner == 1
        )
        buildings = list(state.buildings)
        buildings[pi_planet] = Building.PLANETARY_INSTITUTE
        state = replace(state, buildings=tuple(int(value) for value in buildings))

        income = state._income_preview(1)
        self.assertEqual(income["power_tokens"], 1)
        self.assertEqual(income["power_charge"], 4)

    def test_taklons_pi_can_take_passive_token_before_or_after_charge(self) -> None:
        state = GaiaState.initial(2, faction_indices=(0, 4), first_player=0)
        target = next(planet for planet, active in enumerate(state.active_planets) if active)
        pi_planet = next(
            planet
            for planet, active in enumerate(state.active_planets)
            if active and planet != target and state._distance(target, planet) <= 2
        )
        owners = [-1] * len(state.owners)
        buildings = [Building.EMPTY] * len(state.buildings)
        owners[pi_planet] = 1
        buildings[pi_planet] = Building.PLANETARY_INSTITUTE
        players = list(state.players)
        players[1] = replace(
            players[1],
            bowl_one=1,
            bowl_two=2,
            bowl_three=0,
            brainstone_bowl=1,
            vp=10,
        )
        state = replace(
            state,
            owners=tuple(owners),
            buildings=tuple(int(value) for value in buildings),
            players=tuple(players),
            player_to_move=0,
            round_number=1,
            placement_step=len(state.placement_order),
        )

        pending = state._trigger_passive_charge(0, target, 2)
        self.assertEqual(pending.player_to_move, 1)
        self.assertEqual(pending.pending_taklons_charge_player, 1)
        self.assertEqual(
            pending.legal_actions(),
            (TAKLONS_PASSIVE_BEFORE_ACTION, TAKLONS_PASSIVE_AFTER_ACTION),
        )
        self.assertEqual(pending.snapshot()["phase"], "taklons_passive_charge")

        before = pending.apply(TAKLONS_PASSIVE_BEFORE_ACTION).players[1]
        after = pending.apply(TAKLONS_PASSIVE_AFTER_ACTION).players[1]
        self.assertEqual(before.vp, 9)
        self.assertEqual(after.vp, 9)
        self.assertEqual(before.brainstone_bowl, 2)
        self.assertEqual(after.brainstone_bowl, 2)
        self.assertEqual((before.bowl_one, before.bowl_two, before.bowl_three), (0, 4, 0))
        self.assertEqual((after.bowl_one, after.bowl_two, after.bowl_three), (1, 2, 1))
        self.assertEqual(pending.apply(TAKLONS_PASSIVE_AFTER_ACTION).player_to_move, 1)

        tech_pending = replace(pending, pending_tech_player=0)
        resumed = tech_pending.apply(TAKLONS_PASSIVE_AFTER_ACTION)
        self.assertEqual(resumed.player_to_move, 0)
        self.assertEqual(resumed.pending_tech_player, 0)

    def test_ambas_pi_swap_is_once_per_round_and_has_no_upgrade_effects(self) -> None:
        state = finish_starting_placement(
            GaiaState.initial(2, faction_indices=(5, 0), first_player=0)
        )
        pi_planet, mine_planet = state.starting_planets[0]
        buildings = list(state.buildings)
        buildings[pi_planet] = Building.PLANETARY_INSTITUTE
        federated = list(state.federated)
        federated[pi_planet] = True
        federated[mine_planet] = True
        state = replace(
            state,
            player_to_move=0,
            buildings=tuple(int(value) for value in buildings),
            federated=tuple(federated),
        )

        swap = state.upgrade_pi_action(mine_planet)
        self.assertIn(swap, state.legal_actions())
        self.assertIn("Ambas swap", state.describe_action(swap))
        before = state.players[0]
        income_before = state._income_preview(0)
        swapped = state.apply(swap)
        after = swapped.players[0]

        self.assertEqual(swapped.buildings[pi_planet], Building.MINE)
        self.assertEqual(swapped.buildings[mine_planet], Building.PLANETARY_INSTITUTE)
        self.assertTrue(swapped.federated[pi_planet])
        self.assertTrue(swapped.federated[mine_planet])
        self.assertTrue(after.used_ambas_swap_action)
        self.assertEqual(
            (after.credits, after.ore, after.knowledge, after.vp),
            (before.credits, before.ore, before.knowledge, before.vp),
        )
        self.assertEqual(swapped._income_preview(0), income_before)
        self.assertNotIn(
            swap,
            replace(swapped, player_to_move=0).legal_actions(),
        )

        reset = replace(
            swapped,
            player_to_move=0,
            players=(replace(after, used_ambas_swap_action=False), swapped.players[1]),
        )
        self.assertIn(reset.upgrade_pi_action(pi_planet), reset.legal_actions())

    def test_hadsch_hallas_pi_converts_credits_without_ending_turn(self) -> None:
        state = finish_starting_placement(
            GaiaState.initial(2, faction_indices=(6, 0), first_player=0)
        )
        players = list(state.players)
        players[0] = replace(
            players[0],
            credits=12,
            ore=0,
            knowledge=0,
            qic=0,
        )
        state = replace(state, player_to_move=0, players=tuple(players))
        conversions = (
            TERRANS_GAIA_ORE_ACTION,
            TERRANS_GAIA_KNOWLEDGE_ACTION,
            TERRANS_GAIA_QIC_ACTION,
        )
        self.assertTrue(all(action not in state.legal_actions() for action in conversions))
        income_without_pi = state._income_preview(0)

        pi_planet = state.starting_planets[0][0]
        buildings = list(state.buildings)
        buildings[pi_planet] = Building.PLANETARY_INSTITUTE
        state = replace(state, buildings=tuple(int(value) for value in buildings))
        income_with_pi = state._income_preview(0)
        self.assertEqual(
            income_with_pi["power_tokens"],
            income_without_pi["power_tokens"] + 1,
        )
        self.assertEqual(
            income_with_pi["power_charge"],
            income_without_pi["power_charge"] + 4,
        )
        self.assertTrue(all(action in state.legal_actions() for action in conversions))

        ore = state.apply(TERRANS_GAIA_ORE_ACTION)
        self.assertEqual(ore.player_to_move, 0)
        self.assertEqual((ore.players[0].credits, ore.players[0].ore), (9, 1))
        knowledge = ore.apply(TERRANS_GAIA_KNOWLEDGE_ACTION)
        self.assertEqual(knowledge.player_to_move, 0)
        self.assertEqual(
            (knowledge.players[0].credits, knowledge.players[0].knowledge),
            (5, 1),
        )
        qic = knowledge.apply(TERRANS_GAIA_QIC_ACTION)
        self.assertEqual(qic.player_to_move, 0)
        self.assertEqual((qic.players[0].credits, qic.players[0].qic), (1, 1))
        self.assertEqual(qic.used_power_actions, 0)
        self.assertTrue(all(action not in qic.legal_actions() for action in conversions))

        capped = replace(
            state,
            players=(replace(state.players[0], ore=15, knowledge=15), state.players[1]),
        )
        self.assertNotIn(TERRANS_GAIA_ORE_ACTION, capped.legal_actions())
        self.assertNotIn(TERRANS_GAIA_KNOWLEDGE_ACTION, capped.legal_actions())
        self.assertIn(TERRANS_GAIA_QIC_ACTION, capped.legal_actions())

    def test_brainstone_counts_as_one_token_for_gaia_area(self) -> None:
        info = PlayerState(
            faction=4,
            bowl_one=0,
            bowl_two=0,
            bowl_three=1,
            brainstone_bowl=3,
        )

        moved = GaiaState._move_power_to_gaia(info, 1)
        self.assertEqual(moved.gaia_power, 1)
        self.assertEqual(moved.brainstone_bowl, 4)

        state = GaiaState.initial(2, faction_indices=(4, 0), first_player=0)
        players = list(state.players)
        players[0] = moved
        returned = replace(state, players=tuple(players))._gaia_phase().players[0]
        self.assertEqual(returned.gaia_power, 0)
        self.assertEqual(returned.bowl_one, 1)
        self.assertEqual(returned.brainstone_bowl, 1)

    def test_factions_without_starting_research_begin_at_level_zero(self) -> None:
        no_starting_research = (1, 4, 7, 10, 11, 13)
        for faction_id in no_starting_research:
            opponent = 2 if faction_id == 1 else 0
            with self.subTest(faction=FACTIONS[faction_id].name):
                state = GaiaState.initial(
                    2,
                    faction_indices=(faction_id, opponent),
                    first_player=0,
                )
                self.assertEqual(state.players[0].tracks, (0,) * len(Track))
                selected = state.snapshot()["setup"]["factions"][0]
                self.assertIsNone(selected["start_track"])

                catalog = state.snapshot()["setup"]["faction_catalog"]
                entry = next(faction for faction in catalog if faction["id"] == faction_id)
                self.assertIsNone(entry["start_track"])

    def test_xenos_third_starting_mine_does_not_increase_round_one_income(self) -> None:
        state = GaiaState.initial(
            2,
            faction_indices=(2, 0),
            first_player=0,
        )
        while state.is_starting_placement:
            state = state.apply(state.legal_actions()[0])

        self.assertTrue(state.is_booster_selection)
        self.assertEqual(state._building_count(0, Building.MINE), 3)
        first_income = state._grant_income().players[0]
        self.assertEqual(first_income.ore - state.players[0].ore, 3)

        owners = list(state.owners)
        buildings = list(state.buildings)
        fourth_mine = next(
            planet
            for planet, active in enumerate(state.active_planets)
            if active and owners[planet] < 0
        )
        owners[fourth_mine] = 0
        buildings[fourth_mine] = Building.MINE
        expanded = replace(
            state,
            owners=tuple(owners),
            buildings=tuple(buildings),
        )
        later_income = expanded._grant_income().players[0]
        self.assertEqual(later_income.ore - expanded.players[0].ore, 4)

    def test_faction_catalog_includes_initial_research_rewards(self) -> None:
        state = GaiaState.initial(2, faction_indices=(0, 9), first_player=0)
        catalog = {
            faction["name"]: faction
            for faction in state.snapshot()["setup"]["faction_catalog"]
        }

        self.assertEqual(catalog["Lantids"]["starting_credits"], 13)
        self.assertEqual(catalog["Firaks"]["starting_knowledge"], 2)
        self.assertEqual(catalog["Bescods"]["starting_knowledge"], 1)
        self.assertEqual(catalog["Nevlas"]["starting_knowledge"], 2)
        self.assertEqual(catalog["Geodens"]["starting_ore"], 6)
        self.assertEqual(catalog["Gleens"]["starting_ore"], 5)
        self.assertEqual(catalog["Gleens"]["starting_qic"], 0)
        self.assertEqual(catalog["Xenos"]["starting_qic"], 2)
        self.assertEqual(catalog["Ambas"]["starting_qic"], 2)
        self.assertEqual(catalog["Ivits"]["starting_qic"], 1)
        self.assertEqual(catalog["Terrans"]["starting_gaiaformers"], 1)
        self.assertEqual(catalog["Bal T'aks"]["starting_gaiaformers"], 1)
        self.assertEqual(catalog["Itars"]["starting_gaiaformers"], 0)

    def test_lantids_pi_rewards_coexisting_mines_without_replacing_host(self) -> None:
        state = finish_starting_placement(
            GaiaState.initial(2, faction_indices=(1, 2), first_player=0, seed=17)
        )
        source = state.starting_planets[0][0]
        target = next(
            planet
            for planet, active in enumerate(state.active_planets)
            if active
            and state.owners[planet] != 0
            and state._distance(source, planet) <= 4
        )
        owners = list(state.owners)
        buildings = list(state.buildings)
        terrains = list(state.terrains)
        owners[target] = 1
        buildings[target] = Building.MINE
        terrains[target] = Terrain.GAIA
        tracks = list(state.players[0].tracks)
        tracks[Track.NAVIGATION] = 5
        players = list(state.players)
        players[0] = replace(
            players[0],
            credits=10,
            ore=10,
            knowledge=0,
            qic=1,
            tracks=tuple(tracks),
        )
        state = replace(
            state,
            player_to_move=0,
            players=tuple(players),
            owners=tuple(owners),
            buildings=tuple(int(value) for value in buildings),
            terrains=tuple(int(value) for value in terrains),
        )
        action = state.build_action(target)

        self.assertIn(action, state.legal_actions())
        self.assertEqual(state._build_cost(0, target), (2, 1, 0))
        self.assertIn("coexisting mine", state.describe_action(action))
        without_pi = state.apply(action)
        self.assertEqual(without_pi.players[0].knowledge, 0)

        buildings[source] = Building.PLANETARY_INSTITUTE
        with_pi = replace(
            state,
            buildings=tuple(int(value) for value in buildings),
        )
        types_before = with_pi.players[0].colonized_types
        gaia_before = with_pi._final_scoring_metric(0, 3)
        mines_before = with_pi._building_count(0, Building.MINE)
        ore_income_before = with_pi._income_preview(0)["ore"]
        built = with_pi.apply(action)

        self.assertEqual((built.players[0].credits, built.players[0].ore), (8, 9))
        self.assertEqual(built.players[0].knowledge, 2)
        self.assertEqual(built.players[0].qic, 1)
        self.assertEqual(built.owners[target], 1)
        self.assertEqual(built.buildings[target], Building.MINE)
        self.assertEqual(built.coexisting_mine_owner[target], 0)
        self.assertEqual(built._building_count(0, Building.MINE), mines_before + 1)
        self.assertEqual(built._income_preview(0)["ore"], ore_income_before + 1)
        self.assertEqual(built.players[0].colonized_types, types_before)
        self.assertEqual(built._final_scoring_metric(0, 3), gaia_before)
        self.assertNotIn(
            built.upgrade_trading_action(target),
            replace(built, player_to_move=0).legal_actions(),
        )
        host_upgrade = built.upgrade_trading_action(target)
        self.assertIn(host_upgrade, built.legal_actions())
        host_upgraded = built.apply(host_upgrade)
        self.assertEqual(host_upgraded.buildings[target], Building.TRADING_STATION)
        self.assertEqual(host_upgraded.coexisting_mine_owner[target], 0)
        self.assertEqual(
            built._income_preview(0)["power_tokens"],
            0,
        )
        self.assertEqual(built._income_preview(0)["power_charge"], 4)
        planet = next(item for item in built.snapshot()["planets"] if item["id"] == target)
        self.assertEqual(planet["coexisting_mine_owner"], 0)
        self.assertFalse(planet["coexisting_mine_federated"])

    def test_lantids_coexisting_mine_can_join_a_federation(self) -> None:
        state = finish_starting_placement(
            GaiaState.initial(2, faction_indices=(1, 2), first_player=0, seed=23)
        )
        planets = [
            planet
            for planet, active in enumerate(state.active_planets)
            if active
        ][:3]
        pi_planet, academy_planet, target = planets
        owners = [-1] * len(state.owners)
        buildings = [Building.EMPTY] * len(state.buildings)
        owners[pi_planet] = 0
        owners[academy_planet] = 0
        owners[target] = 1
        buildings[pi_planet] = Building.PLANETARY_INSTITUTE
        buildings[academy_planet] = Building.ACADEMY
        buildings[target] = Building.MINE
        coexisting = [-1] * len(state.coexisting_mine_owner)
        coexisting[target] = 0
        players = list(state.players)
        players[0] = replace(
            players[0],
            bowl_one=30,
            bowl_two=0,
            bowl_three=0,
        )
        state = replace(
            state,
            player_to_move=0,
            players=tuple(players),
            owners=tuple(owners),
            buildings=tuple(int(value) for value in buildings),
            coexisting_mine_owner=tuple(coexisting),
        )

        plan = state._federation_plan(0)
        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertIn(len(state.owners) + target, plan[0])
        formed = state.apply(state.federation_action(0))

        self.assertTrue(formed.coexisting_mine_federated[target])
        self.assertTrue(formed.federated[pi_planet])
        self.assertTrue(formed.federated[academy_planet])
        self.assertEqual(formed._final_scoring_metric(0, 0), 3)

    def test_snapshot_preserves_complete_initial_setup(self) -> None:
        snapshot = GaiaState.initial(3, seed=41).snapshot()
        setup = snapshot["setup"]

        self.assertEqual(snapshot["ruleset"], "standard-v12")
        self.assertEqual(setup["seed"], 41)
        self.assertEqual(setup["map"]["sector_count"], 10)
        self.assertEqual(len(setup["map"]["sectors"]), 10)
        self.assertEqual(setup["map"]["method"], "bga-random")
        self.assertEqual(len(setup["factions"]), 3)
        self.assertEqual(len(setup["faction_catalog"]), 14)
        self.assertEqual(
            {faction["name"] for faction in setup["faction_catalog"]},
            {faction.name for faction in FACTIONS},
        )
        self.assertEqual(
            len({(faction["board"], faction["side"]) for faction in setup["faction_catalog"]}),
            14,
        )
        self.assertTrue(all(
            faction["side"] in ("A", "B")
            for faction in setup["factions"]
        ))
        ivits = next(
            faction for faction in setup["faction_catalog"] if faction["name"] == "Ivits"
        )
        self.assertTrue(ivits["starts_with_pi"])
        self.assertTrue(ivits["places_last"])
        self.assertEqual(len(setup["boosters"]), 6)
        self.assertEqual(len(setup["round_scoring"]), 6)
        self.assertEqual(len(setup["final_scoring"]), 2)
        self.assertEqual(len(setup["standard_tech"]), 9)
        self.assertEqual(len(setup["advanced_tech"]), 6)
        self.assertIn("terraforming_federation", setup)
        self.assertTrue(all(sector["side"] == "solid" for sector in setup["map"]["sectors"]))
        self.assertTrue(all(isinstance(player["satellites"], int) for player in snapshot["players"]))
        self.assertTrue(all(isinstance(player["colonized_types"], int) for player in snapshot["players"]))
        for player in snapshot["players"]:
            self.assertIsInstance(player["gaiaformers_on_board"], int)
            for building, maximum in MAX_BUILDINGS.items():
                inventory = player["structures"][building.name.lower()]
                self.assertEqual(inventory["built"] + inventory["supply"], maximum)
                self.assertEqual(
                    inventory["built"],
                    sum(
                        planet["owner"] == player["id"]
                        and planet["building"] == building.name.lower()
                        for planet in snapshot["planets"]
                    )
                    + (
                        sum(
                            planet["coexisting_mine_owner"] == player["id"]
                            for planet in snapshot["planets"]
                        )
                        if building == Building.MINE
                        else 0
                    ),
                )

        two_player_sectors = GaiaState.initial(2, seed=41).snapshot()["setup"]["map"]["sectors"]
        self.assertTrue(all(
            sector["side"] == ("outlined" if sector["tile"] in (5, 6, 7) else "solid")
            for sector in two_player_sectors
        ))

    def test_observation_encodes_public_booster_pool(self) -> None:
        state = GaiaState.initial(2, seed=17)
        available = next(
            booster for booster, owner in enumerate(state.booster_owner) if owner == -1
        )
        excluded = next(
            booster for booster, owner in enumerate(state.booster_owner) if owner == -2
        )
        owners = list(state.booster_owner)
        owners[available], owners[excluded] = owners[excluded], owners[available]
        changed = replace(state, booster_owner=tuple(owners))

        self.assertFalse(np.array_equal(state.observation(), changed.observation()))

    def test_component_catalogs_follow_bga_sprite_order(self) -> None:
        self.assertEqual(
            tuple(tile.key for tile in STANDARD_TECH_TILES),
            (
                "ore-qic",
                "planet-type-knowledge",
                "vp-7",
                "gaia-mine-vp",
                "structure-power",
                "ore-power-income",
                "knowledge-credit-income",
                "credits-income",
                "power-action",
            ),
        )
        self.assertEqual(
            tuple(tile.key for tile in ROUND_SCORING_TILES),
            (
                "terraform-2",
                "research-2",
                "mine-2",
                "federation-5",
                "trading-3",
                "trading-4",
                "gaia-mine-3",
                "gaia-mine-4",
                "big-5a",
                "big-5b",
            ),
        )
        self.assertEqual(
            tuple(tile.key for tile in FINAL_SCORING_TILES),
            (
                "federation-structures",
                "structures",
                "planet-types",
                "gaia-planets",
                "sectors",
                "satellites",
            ),
        )
        self.assertEqual(len(ADVANCED_TECH_TILES), 15)
        self.assertTrue(ADVANCED_TECH_TILES[0].label.startswith("Action: gain 1 Q.I.C."))
        self.assertTrue(ADVANCED_TECH_TILES[-1].label.startswith("3 VP per trading station"))
        self.assertEqual(len(BOOSTER_LABELS), 10)

    def test_bga_booster_income_order(self) -> None:
        self.assertEqual(
            tuple(GaiaState._booster_income(booster) for booster in range(BOOSTER_COUNT)),
            (
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
            ),
        )

    def test_bga_standard_tech_immediate_rewards(self) -> None:
        state = finish_starting_placement(GaiaState.initial(2, seed=7))
        player = state.current_player
        players = list(state.players)
        players[player] = replace(
            players[player],
            ore=4,
            knowledge=3,
            qic=1,
            vp=10,
            colonized_types=(1 << Terrain.TERRA) | (1 << Terrain.GAIA),
            tracks=(0, 0, 0, 0, 0, 0),
        )
        state = replace(
            state,
            players=tuple(players),
            round_scoring_tiles=(0, 1, 2, 3, 4, 5),
        )

        ore_qic = replace(
            state,
            standard_tech_tiles=(0, 3, 4, 5, 6, 7, 1, 2, 8),
        )._apply_tech(Track.TERRAFORMING).players[player]
        self.assertEqual((ore_qic.ore, ore_qic.qic), (7, 2))

        knowledge = replace(
            state,
            standard_tech_tiles=(1, 3, 4, 5, 6, 7, 0, 2, 8),
        )._apply_tech(Track.TERRAFORMING).players[player]
        self.assertEqual(knowledge.knowledge, 5)

        victory_points = replace(
            state,
            standard_tech_tiles=(2, 3, 4, 5, 6, 7, 0, 1, 8),
        )._apply_tech(Track.TERRAFORMING).players[player]
        self.assertEqual(victory_points.vp, 17)

    def test_bga_research_track_immediate_rewards(self) -> None:
        state = replace(
            GaiaState.initial(2, faction_indices=(0, 2), first_player=0),
            round_number=1,
            round_scoring_tiles=(0, 1, 2, 3, 4, 5),
            terraforming_federation_tile=5,
        )

        def advance_to_top(track: Track, info: PlayerState) -> PlayerState:
            for level in range(1, 6):
                if level == 5:
                    info = replace(info, federation_keys=1)
                info = state._advance_research(
                    0,
                    info,
                    track,
                    score_round=False,
                )
            return info

        base = PlayerState(
            faction=0,
            credits=0,
            ore=0,
            knowledge=0,
            qic=0,
            vp=0,
            bowl_one=12,
            bowl_two=0,
            bowl_three=0,
        )
        terraforming = advance_to_top(Track.TERRAFORMING, base)
        self.assertEqual(terraforming.ore, 4)
        self.assertEqual(terraforming.vp, 12)
        self.assertEqual(terraforming.federation_tokens, 1)
        self.assertEqual(
            terraforming.federation_keys,
            int(state.terraforming_federation_tile != 5),
        )

        navigation = advance_to_top(Track.NAVIGATION, base)
        self.assertEqual(navigation.qic, 2)

        artificial_intelligence = advance_to_top(
            Track.ARTIFICIAL_INTELLIGENCE,
            base,
        )
        self.assertEqual(artificial_intelligence.qic, 10)

        gaia_planets = [
            planet
            for planet, terrain in enumerate(state.terrains)
            if state.active_planets[planet] and terrain == Terrain.GAIA
        ][:2]
        owners = list(state.owners)
        for planet in gaia_planets:
            owners[planet] = 0
        gaia_state = replace(state, owners=tuple(owners))
        gaia = base
        for level in range(1, 6):
            if level == 5:
                gaia = replace(gaia, federation_keys=1)
            gaia = gaia_state._advance_research(
                0,
                gaia,
                Track.GAIA_PROJECT,
                score_round=False,
            )
        self.assertEqual(gaia.gaiaformers, 3)
        self.assertEqual(gaia.bowl_one, 12)
        self.assertEqual(gaia.vp, 4 + len(gaia_planets))

        economy = advance_to_top(Track.ECONOMY, base)
        self.assertEqual((economy.credits, economy.ore), (6, 3))
        self.assertEqual((economy.bowl_one, economy.bowl_two), (3, 9))

        science = advance_to_top(Track.SCIENCE, base)
        self.assertEqual(science.knowledge, 9)

    def test_bga_economy_and_science_track_income(self) -> None:
        state = GaiaState.initial(2, faction_indices=(0, 2), first_player=0)
        economy_income = (
            (0, 0, 0),
            (2, 0, 1),
            (2, 1, 2),
            (3, 1, 3),
            (4, 2, 4),
            (0, 0, 0),
        )
        for level, (credits, ore, charge) in enumerate(economy_income):
            tracks = [0] * 6
            tracks[Track.ECONOMY] = level
            players = list(state.players)
            players[0] = replace(
                players[0],
                credits=0,
                ore=0,
                knowledge=0,
                bowl_one=12,
                bowl_two=0,
                bowl_three=0,
                tracks=tuple(tracks),
                tech_tiles=0,
            )
            income = replace(state, players=tuple(players))._grant_income().players[0]
            self.assertEqual(income.credits, credits)
            self.assertEqual(income.ore, 1 + ore)
            self.assertEqual((income.bowl_one, income.bowl_two), (12 - charge, charge))

        for level, knowledge in enumerate((0, 1, 2, 3, 4, 0)):
            tracks = [0] * 6
            tracks[Track.SCIENCE] = level
            players = list(state.players)
            players[0] = replace(
                players[0],
                credits=0,
                ore=0,
                knowledge=0,
                tracks=tuple(tracks),
                tech_tiles=0,
            )
            income = replace(state, players=tuple(players))._grant_income().players[0]
            self.assertEqual(income.knowledge, 1 + knowledge)

    def test_faction_income_panels_match_board_slots(self) -> None:
        def panel_state(
            faction: int,
            buildings_to_place: tuple[Building, ...] = (),
            **player_changes: object,
        ) -> GaiaState:
            opponent = 2 if faction == 0 else 0
            state = GaiaState.initial(
                2,
                faction_indices=(faction, opponent),
                first_player=0,
                standard_tech_tiles=tuple(range(STANDARD_TECH_COUNT)),
            )
            owners = [-1] * len(state.owners)
            buildings = [Building.EMPTY] * len(state.buildings)
            planets = [
                planet
                for planet, active in enumerate(state.active_planets)
                if active
            ]
            for planet, building in zip(planets, buildings_to_place, strict=False):
                owners[planet] = 0
                buildings[planet] = building
            players = list(state.players)
            players[0] = replace(
                players[0],
                credits=0,
                ore=0,
                knowledge=0,
                qic=0,
                tracks=(0, 0, 0, 0, 0, 0),
                tech_tiles=0,
                bowl_one=12,
                bowl_two=0,
                bowl_three=0,
                **player_changes,
            )
            return replace(
                state,
                players=tuple(players),
                owners=tuple(owners),
                buildings=tuple(int(value) for value in buildings),
            )

        generic = panel_state(
            0,
            (Building.MINE,) * 3
            + (Building.TRADING_STATION,) * 4
            + (Building.RESEARCH_LAB,) * 3,
        )._grant_income().players[0]
        self.assertEqual((generic.credits, generic.ore, generic.knowledge), (16, 3, 4))

        firaks = panel_state(10)._grant_income().players[0]
        self.assertEqual(firaks.knowledge, 2)
        hadsch_hallas = panel_state(6)._grant_income().players[0]
        self.assertEqual(hadsch_hallas.credits, 3)
        ambas = panel_state(5, (Building.MINE,) * 2)._grant_income().players[0]
        self.assertEqual(ambas.ore, 4)

        bescods_trading = panel_state(
            11,
            (Building.TRADING_STATION,) * 4,
        )._grant_income().players[0]
        self.assertEqual((bescods_trading.credits, bescods_trading.knowledge), (0, 4))
        bescods_labs = panel_state(
            11,
            (Building.RESEARCH_LAB,) * 3,
        )._grant_income().players[0]
        self.assertEqual((bescods_labs.credits, bescods_labs.knowledge), (12, 0))

        nevlas = panel_state(
            12,
            (Building.RESEARCH_LAB,) * 3,
        )._grant_income().players[0]
        self.assertEqual(nevlas.knowledge, 1)
        self.assertEqual((nevlas.bowl_one, nevlas.bowl_two), (6, 6))

        itars = panel_state(13, knowledge_academies=1)._grant_income().players[0]
        self.assertEqual(itars.knowledge, 4)

        bal_taks_state = replace(
            panel_state(9, qic_academies=1),
            round_number=1,
            player_to_move=0,
        )
        self.assertIn(QIC_ACADEMY_ACTION, bal_taks_state.legal_actions())
        bal_taks = bal_taks_state.apply(QIC_ACADEMY_ACTION).players[0]
        self.assertEqual((bal_taks.credits, bal_taks.qic), (4, 0))

    def test_snapshot_round_income_preview_matches_granted_income(self) -> None:
        state = GaiaState.initial(
            2,
            faction_indices=(0, 2),
            first_player=0,
            standard_tech_tiles=tuple(range(STANDARD_TECH_COUNT)),
        )
        planets = [
            planet
            for planet, active in enumerate(state.active_planets)
            if active
        ]
        placed = (
            (Building.MINE,) * 3
            + (Building.TRADING_STATION,) * 2
            + (Building.RESEARCH_LAB, Building.PLANETARY_INSTITUTE, Building.ACADEMY)
        )
        owners = [-1] * len(state.owners)
        buildings = [Building.EMPTY] * len(state.buildings)
        for planet, building in zip(planets[:len(placed)], placed, strict=True):
            owners[planet] = 0
            buildings[planet] = building

        tracks = [0] * len(Track)
        tracks[Track.ECONOMY] = 3
        tracks[Track.SCIENCE] = 2
        players = list(state.players)
        players[0] = replace(
            players[0],
            credits=0,
            ore=0,
            knowledge=0,
            qic=0,
            bowl_one=12,
            bowl_two=0,
            bowl_three=0,
            tracks=tuple(tracks),
            tech_tiles=(1 << 5) | (1 << 6) | (1 << 7),
            knowledge_academies=1,
        )
        booster_owner = [-1] * len(state.booster_owner)
        booster_owner[4] = 0
        state = replace(
            state,
            players=tuple(players),
            owners=tuple(owners),
            buildings=tuple(int(value) for value in buildings),
            booster_owner=tuple(booster_owner),
        )

        expected = {
            "credits": 17,
            "ore": 5,
            "knowledge": 7,
            "qic": 1,
            "power_tokens": 1,
            "power_charge": 8,
        }
        self.assertEqual(state._income_preview(0), expected)
        self.assertEqual(state.snapshot()["players"][0]["round_income"], expected)

        after = state._grant_income().players[0]
        self.assertEqual(
            (after.credits, after.ore, after.knowledge, after.qic),
            (17, 5, 7, 1),
        )
        self.assertEqual((after.bowl_one, after.bowl_two, after.bowl_three), (5, 8, 0))

    def test_bescods_swap_academy_and_planetary_institute_upgrades(self) -> None:
        state = finish_starting_placement(
            GaiaState.initial(
                2,
                faction_indices=(11, 0),
                first_player=0,
                standard_tech_tiles=tuple(range(STANDARD_TECH_COUNT)),
            )
        )
        planet = state.starting_planets[0][0]
        owners = list(state.owners)
        buildings = list(state.buildings)
        owners[planet] = 0
        buildings[planet] = Building.TRADING_STATION
        players = list(state.players)
        players[0] = replace(players[0], credits=30, ore=15, tech_tiles=0)
        state = replace(
            state,
            player_to_move=0,
            players=tuple(players),
            owners=tuple(owners),
            buildings=tuple(int(value) for value in buildings),
        )

        legal = state.legal_actions()
        self.assertIn(state.upgrade_lab_action(planet), legal)
        self.assertIn(state.upgrade_academy_action(planet), legal)
        self.assertIn(state.upgrade_qic_academy_action(planet), legal)
        self.assertNotIn(state.upgrade_pi_action(planet), legal)
        academy = state.apply(state.upgrade_academy_action(planet))
        self.assertEqual(academy.buildings[planet], Building.ACADEMY)
        self.assertEqual(academy.pending_tech_player, 0)

        buildings[planet] = Building.RESEARCH_LAB
        state = replace(state, buildings=tuple(int(value) for value in buildings))
        legal = state.legal_actions()
        self.assertIn(state.upgrade_pi_action(planet), legal)
        self.assertNotIn(state.upgrade_academy_action(planet), legal)
        self.assertNotIn(state.upgrade_qic_academy_action(planet), legal)
        institute = state.apply(state.upgrade_pi_action(planet))
        self.assertEqual(institute.buildings[planet], Building.PLANETARY_INSTITUTE)
        self.assertEqual(institute.pending_tech_player, -1)

    def test_tech_tile_research_advance_scores_the_round_tile(self) -> None:
        state = replace(
            finish_starting_placement(
                GaiaState.initial(2, faction_indices=(0, 2), first_player=0)
            ),
            player_to_move=0,
            round_scoring_tiles=(1, 0, 2, 3, 4, 5),
            standard_tech_tiles=(0, 1, 2, 3, 4, 5, 6, 7, 8),
        )
        players = list(state.players)
        players[0] = replace(
            players[0],
            vp=10,
            tracks=(0, 0, 0, 0, 0, 0),
        )
        state = replace(state, players=tuple(players))

        researched = state._apply_tech(Track.TERRAFORMING).players[0]

        self.assertEqual(researched.tracks[Track.TERRAFORMING], 1)
        self.assertEqual(researched.vp, 12)

    def test_bga_booster_pass_scoring_order(self) -> None:
        state = finish_starting_placement(GaiaState.initial(2, seed=9))
        planets = [index for index, active in enumerate(state.active_planets) if active][:5]
        owners = [-1] * len(state.owners)
        buildings = [Building.EMPTY] * len(state.buildings)
        terrains = list(state.terrains)
        for planet in planets:
            owners[planet] = 0
        for planet, building in zip(
            planets,
            (
                Building.MINE,
                Building.TRADING_STATION,
                Building.RESEARCH_LAB,
                Building.PLANETARY_INSTITUTE,
                Building.ACADEMY,
            ),
            strict=True,
        ):
            buildings[planet] = building
        terrains[planets[0]] = Terrain.GAIA
        state = replace(
            state,
            owners=tuple(owners),
            buildings=tuple(int(building) for building in buildings),
            terrains=tuple(int(terrain) for terrain in terrains),
        )

        self.assertEqual(
            tuple(state._booster_pass_points(0, booster) for booster in range(BOOSTER_COUNT)),
            (0, 0, 0, 0, 0, 1, 2, 3, 8, 1),
        )

    def test_random_factions_use_different_boards_and_home_planets(self) -> None:
        for seed in range(20):
            state = GaiaState.initial(4, seed)
            factions = [FACTIONS[player.faction] for player in state.players]
            self.assertEqual(len({faction.board for faction in factions}), 4)
            state = finish_starting_placement(state)
            for player, faction in enumerate(factions):
                for planet in state.starting_planets[player]:
                    self.assertEqual(state.terrains[planet], faction.home)
                    self.assertEqual(state.owners[planet], player)

    def test_manual_setup_applies_factions_first_player_and_home_planets(self) -> None:
        state = GaiaState.initial(
            3,
            seed=17,
            faction_indices=(0, 2, 4),
            first_player=2,
        )

        self.assertEqual(state.first_player, 2)
        self.assertEqual(state.current_player, 2)
        self.assertEqual(tuple(player.faction for player in state.players), (0, 2, 4))
        self.assertEqual(state.starting_planets, ((), (), ()))
        state = finish_starting_placement(state)
        for player, faction_id in enumerate((0, 2, 4)):
            faction = FACTIONS[faction_id]
            for planet in state.starting_planets[player]:
                self.assertEqual(state.terrains[planet], faction.home)
                self.assertEqual(state.owners[planet], player)

    def test_starting_bases_are_placed_in_snake_order_before_round_one(self) -> None:
        state = GaiaState.initial(
            3,
            seed=17,
            faction_indices=(0, 2, 4),
            first_player=2,
        )

        self.assertEqual(state.placement_order, (2, 0, 1, 1, 0, 2, 1))
        self.assertEqual(state.snapshot()["phase"], "starting_placement")
        self.assertEqual(sum(owner >= 0 for owner in state.owners), 0)
        for step, expected_player in enumerate(state.placement_order):
            self.assertTrue(state.is_starting_placement)
            self.assertEqual(state.current_player, expected_player)
            self.assertEqual(state.placement_step, step)
            legal = state.legal_actions()
            self.assertTrue(legal)
            self.assertTrue(all(
                state.owners[action] == -1
                and state.terrains[action] == FACTIONS[state.players[expected_player].faction].home
                for action in legal
            ))
            self.assertTrue(state.describe_action(legal[0]).startswith("place starting"))
            state = state.apply(legal[0])

        self.assertFalse(state.is_starting_placement)
        self.assertTrue(state.is_booster_selection)
        self.assertEqual(state.round_number, 0)
        self.assertEqual(state.current_player, 1)
        self.assertEqual(tuple(map(len, state.starting_planets)), (2, 3, 2))
        self.assertEqual(sum(owner >= 0 for owner in state.owners), 7)
        self.assertEqual(state.snapshot()["phase"], "booster_selection")

        self.assertEqual(state.booster_selection_order, (1, 0, 2))
        selected = []
        for step, expected_player in enumerate((1, 0, 2)):
            self.assertTrue(state.is_booster_selection)
            self.assertEqual(state.current_player, expected_player)
            self.assertEqual(state.booster_selection_step, step)
            legal = state.legal_actions()
            self.assertEqual(len(legal), 6 - step)
            self.assertTrue(all("starting booster" in state.describe_action(action) for action in legal))
            booster = legal[-1]
            selected.append(booster)
            state = state.apply(booster)

        self.assertFalse(state.is_booster_selection)
        self.assertEqual(state.round_number, 1)
        self.assertEqual(state.current_player, 2)
        self.assertEqual(state.snapshot()["phase"], "round")
        self.assertEqual(sum(owner >= 0 for owner in state.booster_owner), 3)
        self.assertEqual(len(set(selected)), 3)

    def test_ivits_places_one_starting_planetary_institute(self) -> None:
        state = GaiaState.initial(
            2,
            seed=9,
            faction_indices=(7, 0),
            first_player=0,
        )

        self.assertEqual(state.placement_order, (1, 1, 0))
        self.assertEqual(state.current_player, 1)
        state = state.apply(state.legal_actions()[0])
        self.assertEqual(state.current_player, 1)
        state = state.apply(state.legal_actions()[0])
        self.assertEqual(state.current_player, 0)
        self.assertIn("planetary institute", state.describe_action(state.legal_actions()[0]))
        state = state.apply(state.legal_actions()[0])
        planet = state.starting_planets[0][0]
        self.assertEqual(state.buildings[planet], Building.PLANETARY_INSTITUTE)
        self.assertFalse(state.is_starting_placement)

    def test_ivits_places_after_xenos_third_starting_mine(self) -> None:
        state = GaiaState.initial(
            3,
            seed=12,
            faction_indices=(7, 2, 4),
            first_player=0,
        )

        self.assertEqual(state.placement_order, (1, 2, 2, 1, 1, 0))
        self.assertEqual(state.placement_order[-1], 0)
        for expected_player in state.placement_order[:-1]:
            self.assertEqual(state.current_player, expected_player)
            action = state.legal_actions()[0]
            self.assertIn("mine", state.describe_action(action))
            state = state.apply(action)
        self.assertEqual(state.current_player, 0)
        self.assertIn("planetary institute", state.describe_action(state.legal_actions()[0]))

    def test_manual_setup_rejects_two_sides_of_the_same_faction_board(self) -> None:
        with self.assertRaisesRegex(ValueError, "different double-sided boards"):
            GaiaState.initial(2, faction_indices=(0, 1), first_player=0)

    def test_manual_setup_can_override_every_random_component(self) -> None:
        donor = GaiaState.initial(3, seed=18, faction_indices=(0, 2, 4), first_player=2)
        state = GaiaState.initial(
            3,
            seed=17,
            faction_indices=(0, 2, 4),
            first_player=2,
            sector_tiles=donor.sector_tiles,
            sector_rotations=donor.sector_rotations,
            booster_tiles=(0, 1, 2, 3, 4, 5),
            round_scoring_tiles=(9, 8, 7, 6, 5, 4),
            final_scoring_tiles=(5, 3),
            standard_tech_tiles=(8, 7, 6, 5, 4, 3, 2, 1, 0),
            advanced_tech_tiles=(14, 13, 12, 11, 10, 9),
            terraforming_federation_tile=5,
            map_mode="manual",
        )

        self.assertEqual(state.sector_tiles, donor.sector_tiles)
        self.assertEqual(state.sector_rotations, donor.sector_rotations)
        self.assertEqual(state.round_scoring_tiles, (9, 8, 7, 6, 5, 4))
        self.assertEqual(state.final_scoring_tiles, (5, 3))
        self.assertEqual(state.standard_tech_tiles, (8, 7, 6, 5, 4, 3, 2, 1, 0))
        self.assertEqual(state.advanced_tech_tiles, (14, 13, 12, 11, 10, 9))
        self.assertEqual(state.terraforming_federation_tile, 5)
        self.assertEqual(state.map_mode, "manual")
        self.assertEqual(state.booster_owner[:6], (-1, -1, -1, -1, -1, -1))

    def test_manual_setup_rejects_duplicate_random_tiles(self) -> None:
        with self.assertRaisesRegex(ValueError, "sector tiles must not contain duplicates"):
            GaiaState.initial(
                2,
                sector_tiles=(0, 0, 1, 2, 3, 4, 5),
                sector_rotations=(0, 0, 0, 0, 0, 0, 0),
            )
        with self.assertRaisesRegex(ValueError, "round scoring tiles must not contain duplicates"):
            GaiaState.initial(
                2,
                round_scoring_tiles=(0, 0, 1, 2, 3, 4),
            )

    def test_random_map_never_places_equal_home_types_adjacent(self) -> None:
        for seed in range(20):
            state = GaiaState.initial(2 + seed % 3, seed)
            active = [index for index, value in enumerate(state.active_planets) if value]
            for offset, left in enumerate(active):
                for right in active[offset + 1 :]:
                    if state.terrains[left] == state.terrains[right] and state.terrains[left] < 7:
                        self.assertNotEqual(state._distance(left, right), 1)

    def test_bga_map_uses_complete_planet_inventory(self) -> None:
        full = GaiaState.initial(4, seed=23)
        full_counts = {
            terrain: sum(
                active and value == terrain
                for active, value in zip(full.active_planets, full.terrains, strict=True)
            )
            for terrain in range(9)
        }
        self.assertEqual(full_counts, {
            Terrain.TERRA: 6,
            Terrain.DESERT: 6,
            Terrain.SWAMP: 6,
            Terrain.VOLCANIC: 6,
            Terrain.OXIDE: 6,
            Terrain.TITANIUM: 6,
            Terrain.ICE: 6,
            Terrain.TRANSDIM: 12,
            Terrain.GAIA: 7,
        })

        two_player = GaiaState.initial(2, seed=23)
        two_counts = {
            terrain: sum(
                active and value == terrain
                for active, value in zip(
                    two_player.active_planets,
                    two_player.terrains,
                    strict=True,
                )
            )
            for terrain in range(9)
        }
        self.assertEqual(two_counts, {
            Terrain.TERRA: 4,
            Terrain.DESERT: 4,
            Terrain.SWAMP: 4,
            Terrain.VOLCANIC: 4,
            Terrain.OXIDE: 4,
            Terrain.TITANIUM: 4,
            Terrain.ICE: 4,
            Terrain.TRANSDIM: 7,
            Terrain.GAIA: 5,
        })

    def test_manual_map_mode_requires_an_explicit_sector_layout(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires sector tiles"):
            GaiaState.initial(2, map_mode="manual")

    def test_manual_map_can_position_every_planet_on_the_full_board(self) -> None:
        donor = GaiaState.initial(2, seed=31)
        positions = tuple(
            (planet, -donor.planet_r[planet], donor.planet_q[planet] + donor.planet_r[planet])
            for planet, active in enumerate(donor.active_planets)
            if active
        )

        state = GaiaState.initial(
            2,
            seed=31,
            sector_tiles=donor.sector_tiles,
            sector_rotations=donor.sector_rotations,
            planet_positions=positions,
            map_mode="manual",
        )

        expected = {planet: (q, r) for planet, q, r in positions}
        self.assertTrue(all(
            (state.planet_q[planet], state.planet_r[planet]) == expected[planet]
            for planet in expected
        ))
        self.assertEqual(state.planet_source_q, donor.planet_q)
        self.assertEqual(state.planet_source_r, donor.planet_r)

    def test_manual_planet_positions_reject_overlap_and_outside_cells(self) -> None:
        donor = GaiaState.initial(2, seed=31)
        positions = [
            (planet, donor.planet_q[planet], donor.planet_r[planet])
            for planet, active in enumerate(donor.active_planets)
            if active
        ]
        positions[1] = (positions[1][0], positions[0][1], positions[0][2])
        with self.assertRaisesRegex(ValueError, "must not overlap"):
            GaiaState.initial(
                2,
                seed=31,
                sector_tiles=donor.sector_tiles,
                sector_rotations=donor.sector_rotations,
                planet_positions=tuple(positions),
                map_mode="manual",
            )

        positions[1] = (positions[1][0], 100, 100)
        with self.assertRaisesRegex(ValueError, "outside the assembled map"):
            GaiaState.initial(
                2,
                seed=31,
                sector_tiles=donor.sector_tiles,
                sector_rotations=donor.sector_rotations,
                planet_positions=tuple(positions),
                map_mode="manual",
            )

    def test_manual_layout_can_add_and_delete_individual_planets(self) -> None:
        donor = GaiaState.initial(2, seed=37, faction_indices=(0, 2))
        deleted = next(
            planet
            for planet, active in enumerate(donor.active_planets)
            if active and donor.terrains[planet] == Terrain.GAIA
        )
        source = next(
            planet
            for planet, active in enumerate(donor.active_planets)
            if active and donor.terrains[planet] == Terrain.TRANSDIM
        )
        added = next(
            planet for planet, active in enumerate(donor.active_planets) if not active
        )
        occupied = {
            (donor.planet_q[planet], donor.planet_r[planet])
            for planet, active in enumerate(donor.active_planets)
            if active
        }
        board_spaces = {
            (center_q + q, center_r + r)
            for center_q, center_r in donor.sector_centers
            for q in range(-2, 3)
            for r in range(-2, 3)
            if max(abs(q), abs(r), abs(q + r)) <= 2
        }
        destination = next(iter(board_spaces - occupied))
        layout = tuple(
            (
                planet,
                donor.planet_q[planet],
                donor.planet_r[planet],
                donor.planet_source_ids[planet],
            )
            for planet, active in enumerate(donor.active_planets)
            if active and planet != deleted
        ) + ((added, destination[0], destination[1], source),)

        state = GaiaState.initial(
            2,
            seed=37,
            faction_indices=(0, 2),
            sector_tiles=donor.sector_tiles,
            sector_rotations=donor.sector_rotations,
            planet_layout=layout,
            map_mode="manual",
        )

        self.assertFalse(state.active_planets[deleted])
        self.assertTrue(state.active_planets[added])
        self.assertEqual((state.planet_q[added], state.planet_r[added]), destination)
        self.assertEqual(state.terrains[added], Terrain.TRANSDIM)
        self.assertEqual(state.planet_source_ids[added], source)
        self.assertEqual(sum(state.active_planets), sum(donor.active_planets))

    def test_sector_artwork_footprints_do_not_overlap(self) -> None:
        local_spaces = {
            (q, r)
            for q in range(-2, 3)
            for r in range(-2, 3)
            if max(abs(q), abs(r), abs(q + r)) <= 2
        }
        self.assertEqual(len(local_spaces), 19)
        for players in (2, 3, 4):
            state = GaiaState.initial(players, seed=9)
            occupied: set[tuple[int, int]] = set()
            footprints: list[set[tuple[int, int]]] = []
            for center_q, center_r in state.sector_centers:
                footprint = {
                    (center_q + q, center_r + r)
                    for q, r in local_spaces
                }
                self.assertTrue(occupied.isdisjoint(footprint))
                occupied.update(footprint)
                footprints.append(footprint)

            seams = 0
            for left_index, left in enumerate(footprints):
                for right in footprints[left_index + 1:]:
                    contacts = sum(
                        hex_distance(left_q, left_r, right_q, right_r) == 1
                        for left_q, left_r in left
                        for right_q, right_r in right
                    )
                    if contacts:
                        self.assertGreaterEqual(contacts, 3)
                        seams += 1
            self.assertEqual(seams, 12 if players == 2 else 19)

    def test_power_charges_bowl_one_before_bowl_two(self) -> None:
        info = PlayerState(faction=0, bowl_one=1, bowl_two=2, bowl_three=0)

        charged, amount = GaiaState._charge_power(info, 4)

        self.assertEqual(amount, 4)
        self.assertEqual((charged.bowl_one, charged.bowl_two, charged.bowl_three), (0, 0, 3))
        spent = GaiaState._spend_power(charged, 2)
        self.assertEqual((spent.bowl_one, spent.bowl_two, spent.bowl_three), (2, 0, 1))

    def test_gaia_project_transforms_then_returns_gaiaformer(self) -> None:
        state = finish_starting_placement(GaiaState.initial(2))
        targets = [
            planet
            for planet, active in enumerate(state.active_planets)
            if active and state.terrains[planet] == Terrain.TRANSDIM
        ]
        target = targets[0]
        source = min(
            (
                planet
                for planet, active in enumerate(state.active_planets)
                if active and planet != target
            ),
            key=lambda planet: state._distance(planet, target),
        )
        owners = list(state.owners)
        buildings = list(state.buildings)
        owners[source] = 0
        buildings[source] = Building.MINE
        players = list(state.players)
        tracks = list(players[0].tracks)
        tracks[Track.GAIA_PROJECT] = 1
        tracks[Track.NAVIGATION] = 5
        players[0] = replace(
            players[0],
            faction=0,
            tracks=tuple(tracks),
            gaiaformers=1,
            bowl_one=6,
            bowl_two=0,
            bowl_three=0,
        )
        state = replace(
            state,
            player_to_move=0,
            owners=tuple(owners),
            buildings=tuple(int(value) for value in buildings),
            players=tuple(players),
        )
        action = state.gaia_action(target)
        self.assertIn(action, state.legal_actions())

        started = state.apply(action)
        self.assertEqual(started.gaiaformer_owner[target], 0)
        self.assertEqual(started.players[0].gaia_power, 6)
        transformed = started._gaia_phase()
        self.assertEqual(transformed.terrains[target], Terrain.GAIA)
        self.assertEqual(transformed.players[0].gaia_power, 0)
        self.assertEqual(transformed.players[0].bowl_two, 6)

        transformed = replace(transformed, player_to_move=0)
        built = transformed.apply(transformed.build_action(target))
        self.assertEqual(built.owners[target], 0)
        self.assertEqual(built.gaiaformer_owner[target], -1)
        self.assertEqual(built.players[0].gaiaformers, 1)

    def test_terrans_pi_converts_gaia_power_before_it_returns_to_bowl_two(self) -> None:
        state = GaiaState.initial(
            2,
            faction_indices=(0, 2),
            first_player=1,
        )
        pi_planet = next(
            planet for planet, active in enumerate(state.active_planets) if active
        )
        owners = list(state.owners)
        buildings = list(state.buildings)
        owners[pi_planet] = 0
        buildings[pi_planet] = Building.PLANETARY_INSTITUTE
        players = list(state.players)
        players[0] = replace(
            players[0],
            credits=20,
            ore=4,
            knowledge=3,
            qic=1,
            bowl_two=1,
            gaia_power=8,
        )
        state = replace(
            state,
            round_number=2,
            player_to_move=1,
            owners=tuple(owners),
            buildings=tuple(int(building) for building in buildings),
            players=tuple(players),
        )

        pending = state._gaia_phase()

        self.assertEqual(pending.player_to_move, 0)
        self.assertEqual(pending.pending_gaia_conversion_player, 0)
        self.assertEqual(pending.pending_gaia_conversion_power, 8)
        self.assertEqual(pending.players[0].gaia_power, 8)
        self.assertEqual(pending.players[0].bowl_two, 1)
        self.assertEqual(pending.snapshot()["phase"], "gaia_conversion")
        self.assertEqual(
            set(pending.legal_actions()),
            {
                TERRANS_GAIA_CREDIT_ACTION,
                TERRANS_GAIA_ORE_ACTION,
                TERRANS_GAIA_KNOWLEDGE_ACTION,
                TERRANS_GAIA_QIC_ACTION,
                TERRANS_GAIA_FINISH_ACTION,
            },
        )

        converted = pending.apply(TERRANS_GAIA_ORE_ACTION)
        self.assertEqual(converted.players[0].ore, 5)
        self.assertEqual(converted.pending_gaia_conversion_power, 5)
        self.assertEqual(converted.players[0].gaia_power, 8)
        converted = converted.apply(TERRANS_GAIA_QIC_ACTION)
        self.assertEqual(converted.players[0].qic, 2)
        self.assertEqual(converted.pending_gaia_conversion_power, 1)
        converted = converted.apply(TERRANS_GAIA_CREDIT_ACTION)
        self.assertEqual(converted.players[0].credits, 21)
        self.assertEqual(converted.pending_gaia_conversion_power, 0)
        self.assertEqual(converted.legal_actions(), (TERRANS_GAIA_FINISH_ACTION,))

        finished = converted.apply(TERRANS_GAIA_FINISH_ACTION)
        self.assertEqual(finished.pending_gaia_conversion_player, -1)
        self.assertEqual(finished.pending_gaia_conversion_power, 0)
        self.assertEqual(finished.players[0].gaia_power, 0)
        self.assertEqual(finished.players[0].bowl_two, 9)
        self.assertEqual(finished.player_to_move, 1)
        self.assertEqual(finished.snapshot()["phase"], "round")

    def test_terrans_without_pi_returns_gaia_power_directly_to_bowl_two(self) -> None:
        state = GaiaState.initial(
            2,
            faction_indices=(0, 2),
            first_player=0,
        )
        players = list(state.players)
        players[0] = replace(players[0], bowl_two=2, gaia_power=4)

        resolved = replace(state, players=tuple(players))._gaia_phase()

        self.assertEqual(resolved.pending_gaia_conversion_player, -1)
        self.assertEqual(resolved.players[0].gaia_power, 0)
        self.assertEqual(resolved.players[0].bowl_two, 6)

    def test_research_lab_requires_immediate_tech_choice(self) -> None:
        state = finish_starting_placement(GaiaState.initial(2))
        mine = next(
            planet
            for planet, owner in enumerate(state.owners)
            if owner == 0 and state.buildings[planet] == Building.MINE
        )
        state = replace(state, player_to_move=0)
        state = state.apply(state.upgrade_trading_action(mine))
        players = list(state.players)
        players[0] = replace(players[0], credits=30, ore=15)
        state = replace(state, player_to_move=0, players=tuple(players))

        pending = state.apply(state.upgrade_lab_action(mine))

        self.assertEqual(pending.current_player, 0)
        self.assertEqual(pending.pending_tech_player, 0)
        self.assertTrue(pending.legal_actions())
        self.assertTrue(all(action >= pending.tech_action(Track.TERRAFORMING) for action in pending.legal_actions()))
        resolved = pending.apply(pending.tech_action(Track.TERRAFORMING))
        self.assertEqual(resolved.pending_tech_player, -1)
        self.assertEqual(resolved.current_player, 1)
        self.assertEqual(resolved.players[0].tracks[Track.TERRAFORMING], 1)

    def test_upgrade_requiring_tech_is_hidden_when_no_choice_remains(self) -> None:
        state = finish_starting_placement(GaiaState.initial(2))
        planet = next(index for index, active in enumerate(state.active_planets) if active)
        owners = list(state.owners)
        buildings = list(state.buildings)
        owners[planet] = 0
        buildings[planet] = Building.TRADING_STATION
        players = list(state.players)
        players[0] = replace(
            players[0],
            credits=30,
            ore=15,
            tracks=(5, 5, 5, 5, 5, 5),
            tech_tiles=(1 << STANDARD_TECH_COUNT) - 1,
        )
        state = replace(
            state,
            player_to_move=0,
            players=tuple(players),
            owners=tuple(owners),
            buildings=tuple(int(value) for value in buildings),
        )

        self.assertNotIn(state.upgrade_lab_action(planet), state.legal_actions())

    def test_all_nine_standard_technology_spaces_are_available(self) -> None:
        state = replace(
            finish_starting_placement(
                GaiaState.initial(
                    2,
                    faction_indices=(0, 2),
                    standard_tech_tiles=tuple(range(STANDARD_TECH_COUNT)),
                )
            ),
            player_to_move=0,
            pending_tech_player=0,
        )
        players = list(state.players)
        players[0] = replace(players[0], tech_tiles=0, federation_keys=0)
        state = replace(state, players=tuple(players))

        self.assertEqual(
            state.legal_actions(),
            tuple(state.tech_action(space) for space in range(STANDARD_TECH_COUNT)),
        )

    def test_advanced_technology_covers_standard_tile_and_uses_keys(self) -> None:
        state = finish_starting_placement(
            GaiaState.initial(
                2,
                faction_indices=(0, 2),
                standard_tech_tiles=tuple(range(STANDARD_TECH_COUNT)),
                advanced_tech_tiles=(0, 1, 2, 3, 4, 5),
            )
        )
        players = list(state.players)
        players[0] = replace(
            players[0],
            tracks=(4, 0, 0, 0, 0, 0),
            tech_tiles=1 << 8,
            federation_keys=2,
        )
        state = replace(
            state,
            player_to_move=0,
            players=tuple(players),
            pending_tech_player=0,
        )

        selected = state.apply(state.tech_action(STANDARD_TECH_COUNT))
        self.assertEqual(selected.pending_advanced_tech, 0)
        self.assertEqual(selected.legal_actions(), (state.tech_action(8),))
        covered = selected.apply(state.tech_action(8))
        self.assertTrue(covered.players[0].advanced_tech_tiles & 1)
        self.assertTrue(covered.players[0].covered_tech_tiles & (1 << 8))
        self.assertEqual(covered.players[0].federation_keys, 1)
        self.assertEqual(covered.pending_research_player, 0)

        resolved = covered.apply(state.research_action(Track.TERRAFORMING))
        self.assertEqual(resolved.players[0].tracks[Track.TERRAFORMING], 5)
        self.assertEqual(
            resolved.players[0].federation_keys,
            int(state.terraforming_federation_tile != 5),
        )

    def test_advanced_technology_tile_is_unique_across_players(self) -> None:
        state = finish_starting_placement(
            GaiaState.initial(2, faction_indices=(0, 2), first_player=0)
        )
        advanced_tile = state.advanced_tech_tiles[Track.TERRAFORMING]
        players = list(state.players)
        tracks = list(players[0].tracks)
        tracks[Track.TERRAFORMING] = 4
        players[0] = replace(
            players[0],
            tracks=tuple(tracks),
            federation_keys=1,
            tech_tiles=1,
        )
        players[1] = replace(players[1], advanced_tech_tiles=1 << advanced_tile)
        blocked = replace(state, player_to_move=0, players=tuple(players))

        self.assertNotIn(
            blocked.tech_action(STANDARD_TECH_COUNT + Track.TERRAFORMING),
            blocked._legal_technology_actions(0),
        )

    def test_research_level_five_is_exclusive(self) -> None:
        state = finish_starting_placement(
            GaiaState.initial(2, faction_indices=(0, 2), first_player=0)
        )
        players = list(state.players)
        players[0] = replace(players[0], tracks=(5, 0, 0, 0, 0, 0))
        players[1] = replace(
            players[1],
            tracks=(4, 0, 0, 0, 0, 0),
            knowledge=4,
            federation_keys=1,
        )
        state = replace(state, player_to_move=1, players=tuple(players))

        self.assertFalse(state._can_player_advance(1, Track.TERRAFORMING))
        self.assertNotIn(
            state.research_action(Track.TERRAFORMING),
            state.legal_actions(),
        )

    def test_bal_taks_navigation_unlocks_after_planetary_institute(self) -> None:
        state = finish_starting_placement(
            GaiaState.initial(2, faction_indices=(9, 0), first_player=0)
        )
        self.assertFalse(state._can_player_advance(0, Track.NAVIGATION))

        planet = state.starting_planets[0][0]
        buildings = list(state.buildings)
        buildings[planet] = Building.PLANETARY_INSTITUTE
        unlocked = replace(state, buildings=tuple(int(value) for value in buildings))
        self.assertTrue(unlocked._can_player_advance(0, Track.NAVIGATION))

    def test_xenos_federation_threshold_drops_only_after_pi(self) -> None:
        state = finish_starting_placement(
            GaiaState.initial(2, faction_indices=(2, 0), first_player=0)
        )
        planets = [index for index, active in enumerate(state.active_planets) if active][:3]
        owners = [-1] * len(state.owners)
        buildings = [Building.EMPTY] * len(state.buildings)
        federated = [False] * len(state.federated)
        for planet in planets:
            owners[planet] = 0
            federated[planet] = False
        buildings[planets[0]] = Building.ACADEMY
        buildings[planets[1]] = Building.TRADING_STATION
        buildings[planets[2]] = Building.MINE
        players = list(state.players)
        players[0] = replace(players[0], bowl_one=30, bowl_two=0, bowl_three=0)
        state = replace(
            state,
            players=tuple(players),
            owners=tuple(owners),
            buildings=tuple(int(value) for value in buildings),
            federated=tuple(federated),
        )
        self.assertEqual(state._federation_threshold(0), 7)
        self.assertEqual(state.snapshot()["players"][0]["federation_threshold"], 7)
        self.assertIsNone(state._federation_plan(0))

        buildings[planets[0]] = Building.PLANETARY_INSTITUTE
        with_pi = replace(state, buildings=tuple(int(value) for value in buildings))
        self.assertEqual(with_pi._federation_threshold(0), 6)
        self.assertEqual(
            with_pi.snapshot()["players"][0]["federation_threshold"],
            6,
        )
        self.assertIsNotNone(with_pi._federation_plan(0))

    def test_xenos_pi_income_replaces_power_token_with_qic(self) -> None:
        state = GaiaState.initial(
            2,
            faction_indices=(2, 0),
            first_player=0,
        )
        pi_planet = next(
            planet for planet, active in enumerate(state.active_planets) if active
        )
        owners = [-1] * len(state.owners)
        buildings = [Building.EMPTY] * len(state.buildings)
        owners[pi_planet] = 0
        buildings[pi_planet] = Building.PLANETARY_INSTITUTE
        players = list(state.players)
        players[0] = replace(
            players[0],
            credits=0,
            ore=0,
            knowledge=0,
            qic=0,
            bowl_one=2,
            bowl_two=4,
            bowl_three=0,
        )
        state = replace(
            state,
            players=tuple(players),
            owners=tuple(owners),
            buildings=tuple(int(building) for building in buildings),
        )

        income = state._income_preview(0)

        self.assertEqual(income["qic"], 1)
        self.assertEqual(income["power_tokens"], 0)
        self.assertEqual(income["power_charge"], 4)
        after = state._grant_income().players[0]
        self.assertEqual(after.qic, 1)
        self.assertEqual(
            (after.bowl_one, after.bowl_two, after.bowl_three),
            (0, 4, 2),
        )

    def test_faction_specific_power_and_gaia_rules(self) -> None:
        nevlas = finish_starting_placement(
            GaiaState.initial(2, faction_indices=(12, 0), first_player=0)
        )
        planet = nevlas.starting_planets[0][0]
        buildings = list(nevlas.buildings)
        buildings[planet] = Building.PLANETARY_INSTITUTE
        nevlas = replace(nevlas, buildings=tuple(int(value) for value in buildings))
        self.assertEqual(
            tuple(nevlas._power_action_cost(0, action) for action in PowerAction),
            (4, 3, 2, 2, 2, 2, 2),
        )

        gleens = finish_starting_placement(
            GaiaState.initial(2, faction_indices=(3, 0), first_player=0)
        )
        gaia = next(
            planet
            for planet, active in enumerate(gleens.active_planets)
            if active and gleens.terrains[planet] == Terrain.GAIA
        )
        self.assertEqual(gleens._build_cost(0, gaia), (2, 2, 0))
        players = list(gleens.players)
        players[0] = replace(players[0], credits=30, ore=15, vp=10)
        gleens = replace(
            gleens,
            player_to_move=0,
            players=tuple(players),
            round_scoring_tiles=(1, 0, 2, 3, 4, 5),
        )
        built = gleens._apply_build(gaia)
        self.assertEqual(built.players[0].vp, 12)

        qic_academy = replace(
            gleens,
            players=(replace(gleens.players[0], qic_academies=1), gleens.players[1]),
        )
        self.assertEqual(qic_academy._build_cost(0, gaia), (2, 2, 0))

    def test_gleens_qic_conversion_stops_after_qic_academy(self) -> None:
        state = finish_starting_placement(
            GaiaState.initial(2, faction_indices=(3, 0), first_player=0, seed=31)
        )
        info = replace(state.players[0], ore=5, qic=0)

        converted = state._gain_qic(info, 2)
        enabled = state._gain_qic(replace(info, qic_academies=1), 2)

        self.assertEqual((converted.ore, converted.qic), (7, 0))
        self.assertEqual((enabled.ore, enabled.qic), (5, 2))

        boosters = [owner if owner != 0 else -1 for owner in state.booster_owner]
        without_booster = replace(state, booster_owner=tuple(boosters))
        base_income = without_booster._income_preview(0)
        boosters[4] = 0
        qic_booster = replace(state, booster_owner=tuple(boosters))
        converted_income = qic_booster._income_preview(0)
        self.assertEqual(converted_income["qic"], 0)
        self.assertEqual(converted_income["ore"], base_income["ore"] + 1)

        players = list(qic_booster.players)
        players[0] = replace(players[0], qic_academies=1)
        academy_income = replace(qic_booster, players=tuple(players))._income_preview(0)
        self.assertEqual(academy_income["qic"], 1)
        self.assertEqual(academy_income["ore"], base_income["ore"])

    def test_gleens_pi_grants_special_federation_tile_and_income(self) -> None:
        state = finish_starting_placement(
            GaiaState.initial(2, faction_indices=(3, 0), first_player=0, seed=37)
        )
        planet = state.starting_planets[0][0]
        buildings = list(state.buildings)
        buildings[planet] = Building.TRADING_STATION
        players = list(state.players)
        players[0] = replace(
            players[0],
            credits=10,
            ore=10,
            knowledge=0,
            vp=10,
            federation_tokens=0,
            federation_keys=0,
            gleens_federation_tokens=0,
        )
        state = replace(
            state,
            player_to_move=0,
            players=tuple(players),
            buildings=tuple(int(building) for building in buildings),
            round_scoring_tiles=(3, 0, 1, 2, 4, 5),
        )
        income_before = state._income_preview(0)
        action = state.upgrade_pi_action(planet)

        self.assertIn(action, state.legal_actions())
        built = state.apply(action)
        gleens = built.players[0]

        self.assertEqual((gleens.credits, gleens.ore, gleens.knowledge), (6, 7, 1))
        self.assertEqual(gleens.vp, 15)
        self.assertEqual(gleens.federation_tokens, 1)
        self.assertEqual(gleens.federation_keys, 1)
        self.assertEqual(gleens.gleens_federation_tokens, 1)
        self.assertFalse(built.federated[planet])
        income = built._income_preview(0)
        self.assertEqual(income["ore"], income_before["ore"] + 1)
        self.assertEqual(income["power_tokens"], 0)
        self.assertEqual(
            income["power_charge"],
            income_before["power_charge"] + 4,
        )

        players = list(built.players)
        players[0] = replace(
            players[0],
            credits=0,
            ore=0,
            knowledge=0,
            qic=3,
            qic_academies=1,
        )
        repeat_state = replace(
            built,
            player_to_move=0,
            players=tuple(players),
            used_qic_actions=0,
        )
        repeat_action = QIC_FEDERATION_ACTION_OFFSET + len(FEDERATION_TILES)
        self.assertIn(repeat_action, repeat_state.legal_actions())
        repeated = repeat_state.apply(repeat_action).players[0]
        self.assertEqual((repeated.credits, repeated.ore, repeated.knowledge), (2, 1, 1))
        self.assertEqual(repeated.qic, 0)
        self.assertEqual(repeated.federation_tokens, 1)
        self.assertEqual(repeated.gleens_federation_tokens, 1)

    def test_all_seven_public_power_actions_use_bga_costs_and_rewards(self) -> None:
        state = finish_starting_placement(
            GaiaState.initial(2, faction_indices=(0, 2), first_player=0)
        )
        players = list(state.players)
        players[0] = replace(
            players[0],
            credits=10,
            ore=5,
            knowledge=5,
            bowl_one=0,
            bowl_two=0,
            bowl_three=20,
        )
        state = replace(state, player_to_move=0, players=tuple(players))
        self.assertEqual(
            tuple(state._power_action_cost(0, action) for action in PowerAction),
            (7, 5, 4, 4, 4, 3, 3),
        )

        knowledge = state._apply_power_action(PowerAction.KNOWLEDGE_THREE)
        self.assertEqual(knowledge.players[0].knowledge, 8)
        ore = state._apply_power_action(PowerAction.ORE_TWO)
        self.assertEqual(ore.players[0].ore, 7)
        credits = state._apply_power_action(PowerAction.CREDITS_SEVEN)
        self.assertEqual(credits.players[0].credits, 17)
        tokens = state._apply_power_action(PowerAction.POWER_TOKENS_TWO)
        self.assertEqual(tokens.players[0].bowl_one, 5)

    def test_power_terraform_action_keeps_turn_until_mine_is_built(self) -> None:
        state = finish_starting_placement(
            GaiaState.initial(2, faction_indices=(0, 2), first_player=0)
        )
        players = list(state.players)
        tracks = list(players[0].tracks)
        tracks[Track.NAVIGATION] = 5
        players[0] = replace(
            players[0],
            credits=30,
            ore=15,
            qic=10,
            bowl_one=0,
            bowl_two=0,
            bowl_three=20,
            tracks=tuple(tracks),
        )
        state = replace(state, player_to_move=0, players=tuple(players))

        pending = state.apply(state.power_action(PowerAction.TERRAFORM_TWO))
        self.assertEqual(pending.current_player, 0)
        self.assertEqual(pending.pending_power_terraform_player, 0)
        self.assertTrue(pending.legal_actions())
        self.assertTrue(all(action < pending.gaia_action(0) for action in pending.legal_actions()))
        resolved = pending.apply(pending.legal_actions()[0])
        self.assertEqual(resolved.pending_power_terraform_player, -1)
        self.assertEqual(resolved.current_player, 1)

    def test_terraforming_booster_builds_with_one_free_step(self) -> None:
        state = finish_starting_placement(
            GaiaState.initial(2, faction_indices=(0, 2), first_player=0)
        )
        players = list(state.players)
        tracks = list(players[0].tracks)
        tracks[Track.NAVIGATION] = 5
        players[0] = replace(
            players[0],
            credits=30,
            ore=15,
            qic=10,
            tracks=tuple(tracks),
        )
        boosters = list(state.booster_owner)
        for index, owner in enumerate(boosters):
            if owner == 0:
                boosters[index] = -1
        boosters[0] = 0
        state = replace(
            state,
            player_to_move=0,
            players=tuple(players),
            booster_owner=tuple(boosters),
        )
        target = next(
            planet
            for planet in range(len(state.active_planets))
            if state.active_planets[planet]
            and state.owners[planet] == -1
            and state.terrains[planet]
            not in (Terrain.TRANSDIM, Terrain.GAIA, Terrain.TERRA)
            and state._can_build_mine(0, planet, free_steps=1)
        )
        normal_cost = state._build_cost(0, target)
        boosted_cost = state._build_cost(0, target, free_steps=1)
        self.assertLess(boosted_cost[1], normal_cost[1])
        self.assertIn(BOOSTER_TERRAFORM_ACTION, state.legal_actions())

        pending = state.apply(BOOSTER_TERRAFORM_ACTION)
        self.assertEqual(pending.pending_booster_terraform_player, 0)
        self.assertIn(pending.build_action(target), pending.legal_actions())
        resolved = pending.apply(pending.build_action(target))
        self.assertEqual(resolved.pending_booster_terraform_player, -1)
        self.assertEqual(resolved.players[0].ore, 15 - boosted_cost[1])

    def test_range_booster_extends_a_build_by_three_hexes(self) -> None:
        state = finish_starting_placement(
            GaiaState.initial(2, faction_indices=(0, 2), first_player=0)
        )
        players = list(state.players)
        players[0] = replace(players[0], credits=30, ore=15, qic=10)
        boosters = list(state.booster_owner)
        for index, owner in enumerate(boosters):
            if owner == 0:
                boosters[index] = -1
        boosters[1] = 0
        state = replace(
            state,
            player_to_move=0,
            players=tuple(players),
            booster_owner=tuple(boosters),
        )
        target = next(
            planet
            for planet in range(len(state.active_planets))
            if state.active_planets[planet]
            and state.terrains[planet] != Terrain.TRANSDIM
            and not state._can_build_mine(0, planet)
            and state._can_build_mine(0, planet, range_bonus=3)
        )
        self.assertIn(BOOSTER_RANGE_ACTION, state.legal_actions())

        pending = state.apply(BOOSTER_RANGE_ACTION)
        self.assertEqual(pending.pending_booster_range_player, 0)
        self.assertIn(pending.build_action(target), pending.legal_actions())
        resolved = pending.apply(pending.build_action(target))
        self.assertEqual(resolved.pending_booster_range_player, -1)
        self.assertEqual(resolved.current_player, 1)

    def test_qic_and_knowledge_academies_are_distinct(self) -> None:
        state = finish_starting_placement(
            GaiaState.initial(
                2,
                faction_indices=(3, 0),
                first_player=0,
                standard_tech_tiles=tuple(range(STANDARD_TECH_COUNT)),
            )
        )
        planet = state.starting_planets[0][0]
        buildings = list(state.buildings)
        buildings[planet] = Building.RESEARCH_LAB
        players = list(state.players)
        players[0] = replace(players[0], credits=30, ore=15, tech_tiles=0)
        state = replace(
            state,
            player_to_move=0,
            buildings=tuple(int(value) for value in buildings),
            players=tuple(players),
        )
        self.assertIn(state.upgrade_academy_action(planet), state.legal_actions())
        self.assertIn(state.upgrade_qic_academy_action(planet), state.legal_actions())

        pending = state.apply(state.upgrade_qic_academy_action(planet))
        self.assertEqual(pending.players[0].qic_academies, 1)
        resolved = pending.apply(pending.tech_action(0))
        resolved = replace(resolved, player_to_move=0)
        self.assertIn(QIC_ACADEMY_ACTION, resolved.legal_actions())
        ore_before = resolved.players[0].ore
        qic_before = resolved.players[0].qic
        used = resolved.apply(QIC_ACADEMY_ACTION)
        self.assertEqual(used.players[0].qic, qic_before + 1)
        self.assertEqual(used.players[0].ore, ore_before)
        self.assertTrue(used.players[0].used_qic_academy_action)

    def test_advanced_technology_special_action_is_once_per_round(self) -> None:
        state = finish_starting_placement(
            GaiaState.initial(2, faction_indices=(0, 2), first_player=0)
        )
        players = list(state.players)
        players[0] = replace(
            players[0],
            credits=10,
            qic=1,
            advanced_tech_tiles=1,
        )
        state = replace(state, player_to_move=0, players=tuple(players))
        action = ADVANCED_TECH_ACTION_OFFSET
        self.assertIn(action, state.legal_actions())
        used = state.apply(action)
        self.assertEqual((used.players[0].credits, used.players[0].qic), (15, 2))
        self.assertTrue(used.players[0].used_advanced_tech_actions & 1)
        used = replace(used, player_to_move=0)
        self.assertNotIn(action, used.legal_actions())

    def test_standard_technology_charge_action_is_once_per_round(self) -> None:
        state = finish_starting_placement(
            GaiaState.initial(2, faction_indices=(0, 2), first_player=0)
        )
        players = list(state.players)
        players[0] = replace(
            players[0],
            tech_tiles=1 << 8,
            bowl_one=2,
            bowl_two=2,
            bowl_three=0,
        )
        state = replace(state, player_to_move=0, players=tuple(players))
        self.assertIn(STANDARD_TECH_ACTION, state.legal_actions())
        used = state.apply(STANDARD_TECH_ACTION)
        self.assertEqual(
            (used.players[0].bowl_one, used.players[0].bowl_two, used.players[0].bowl_three),
            (0, 2, 2),
        )
        self.assertTrue(used.players[0].used_standard_tech_action)
        used = replace(used, player_to_move=0)
        self.assertNotIn(STANDARD_TECH_ACTION, used.legal_actions())

        players = list(state.players)
        players[0] = replace(players[0], covered_tech_tiles=1 << 8)
        covered = replace(state, players=tuple(players))
        self.assertNotIn(STANDARD_TECH_ACTION, covered.legal_actions())

    def test_qic_actions_cover_tech_federation_and_planet_type_spaces(self) -> None:
        state = finish_starting_placement(
            GaiaState.initial(
                2,
                faction_indices=(0, 2),
                first_player=0,
                standard_tech_tiles=tuple(range(STANDARD_TECH_COUNT)),
            )
        )
        players = list(state.players)
        players[0] = replace(players[0], qic=7, vp=10)
        state = replace(state, player_to_move=0, players=tuple(players))
        self.assertIn(QIC_TECH_ACTION, state.legal_actions())
        self.assertIn(QIC_PLANET_TYPES_ACTION, state.legal_actions())

        scored = state.apply(QIC_PLANET_TYPES_ACTION)
        self.assertEqual(scored.players[0].qic, 5)
        self.assertEqual(scored.players[0].vp, 14)
        self.assertNotIn(QIC_PLANET_TYPES_ACTION, replace(scored, player_to_move=0).legal_actions())

        tech = replace(scored, player_to_move=0, players=(replace(scored.players[0], qic=5), scored.players[1]))
        pending = tech.apply(QIC_TECH_ACTION)
        self.assertEqual(pending.players[0].qic, 1)
        self.assertEqual(pending.pending_tech_player, 0)
        resolved = pending.apply(pending.tech_action(0))
        self.assertEqual(resolved.pending_tech_player, -1)

        players = list(resolved.players)
        players[0] = replace(
            players[0],
            qic=3,
            federation_tile_counts=(1, 0, 0, 0, 0, 0),
        )
        repeat = replace(
            resolved,
            player_to_move=0,
            players=tuple(players),
            used_qic_actions=0,
        )
        self.assertIn(QIC_FEDERATION_ACTION_OFFSET, repeat.legal_actions())
        repeated = repeat.apply(QIC_FEDERATION_ACTION_OFFSET)
        self.assertEqual(repeated.players[0].qic, 0)
        self.assertEqual(repeated.players[0].vp, players[0].vp + 6)

    def test_action_descriptions_keep_federation_and_qic_ranges_distinct(self) -> None:
        state = finish_starting_placement(GaiaState.initial(2, seed=41))

        self.assertEqual(
            state.describe_action(FEDERATION_OFFSET),
            "form federation and take tile 0",
        )
        self.assertEqual(
            state.describe_action(QIC_FEDERATION_ACTION_OFFSET),
            "Q.I.C. action: repeat federation tile 0",
        )

    def test_bescods_titanium_structure_power_bonus_stacks(self) -> None:
        state = finish_starting_placement(
            GaiaState.initial(2, faction_indices=(11, 0), first_player=0)
        )
        home_planets = state.starting_planets[0]
        buildings = list(state.buildings)
        buildings[home_planets[0]] = Building.PLANETARY_INSTITUTE
        buildings[home_planets[1]] = Building.ACADEMY
        state = replace(state, buildings=tuple(int(value) for value in buildings))
        self.assertEqual(
            state._structure_power(0, Building.ACADEMY, home_planets[1]),
            4,
        )
        players = list(state.players)
        players[0] = replace(players[0], tech_tiles=1 << 4)
        state = replace(state, players=tuple(players))
        self.assertEqual(
            state._structure_power(0, Building.ACADEMY, home_planets[1]),
            5,
        )

    def test_gray_federation_tile_has_no_research_key(self) -> None:
        state = finish_starting_placement(GaiaState.initial(2))
        planets = [index for index, active in enumerate(state.active_planets) if active][:3]
        owners = list(state.owners)
        buildings = list(state.buildings)
        for planet in planets:
            owners[planet] = 0
        buildings[planets[0]] = Building.PLANETARY_INSTITUTE
        buildings[planets[1]] = Building.ACADEMY
        buildings[planets[2]] = Building.MINE
        players = list(state.players)
        players[0] = replace(players[0], bowl_one=30, bowl_two=0, bowl_three=0)
        state = replace(
            state,
            player_to_move=0,
            owners=tuple(owners),
            buildings=tuple(int(value) for value in buildings),
            players=tuple(players),
            round_scoring_tiles=(0, 1, 2, 3, 4, 5),
        )
        gray_action = FEDERATION_OFFSET + 5
        self.assertIn(gray_action, state.legal_actions())
        score_before = state.players[0].vp
        formed = state.apply(gray_action)
        self.assertEqual(formed.players[0].vp, score_before + 12)
        self.assertEqual(formed.players[0].federation_keys, 0)

    def test_two_player_final_ranking_includes_neutral_marker(self) -> None:
        state = GaiaState.initial(2)
        self.assertEqual(state._ranking_awards([9, 8], 10), [12.0, 6.0])
        self.assertEqual(state._ranking_awards([10, 5], 10), [15.0, 6.0])

    def test_final_resource_scoring_excludes_qic(self) -> None:
        state = GaiaState.initial(2, seed=53)
        players = list(state.players)
        players[0] = replace(players[0], credits=0, ore=0, knowledge=0, qic=0)
        empty = replace(state, players=tuple(players))
        base_score = empty.final_scores()[0]

        players[0] = replace(players[0], qic=9)
        qic_only = replace(state, players=tuple(players))
        self.assertEqual(qic_only.final_scores()[0], base_score)

        players[0] = replace(players[0], credits=2, ore=1, knowledge=0, qic=0)
        resources = replace(state, players=tuple(players))
        self.assertEqual(resources.final_scores()[0], base_score + 1)

    def test_canonical_federation_awards_token(self) -> None:
        state = finish_starting_placement(GaiaState.initial(2))
        planets = [index for index, active in enumerate(state.active_planets) if active][:3]
        owners = list(state.owners)
        buildings = list(state.buildings)
        for planet in planets:
            owners[planet] = 0
        buildings[planets[0]] = Building.PLANETARY_INSTITUTE
        buildings[planets[1]] = Building.ACADEMY
        buildings[planets[2]] = Building.MINE
        players = list(state.players)
        players[0] = replace(players[0], bowl_one=15, bowl_two=10, bowl_three=0)
        state = replace(
            state,
            player_to_move=0,
            owners=tuple(owners),
            buildings=tuple(int(value) for value in buildings),
            players=tuple(players),
            round_scoring_tiles=(3, 0, 1, 2, 4, 5),
        )

        self.assertIn(FEDERATION_ACTION, state.legal_actions())
        score_before = state.players[0].vp
        formed = state.apply(FEDERATION_ACTION)
        self.assertEqual(formed.players[0].federation_tokens, 1)
        self.assertEqual(formed.players[0].federation_keys, 1)
        self.assertEqual(formed.players[0].vp, score_before + 11)
        self.assertTrue(all(formed.federated[planet] for planet in planets))

    def test_passing_returns_old_booster_and_takes_new_one(self) -> None:
        state = finish_starting_placement(GaiaState.initial(2))
        player = state.current_player
        old_booster = state._player_booster(player)
        new_booster = next(
            booster for booster, owner in enumerate(state.booster_owner) if owner == -1
        )
        passed = state.apply(state.pass_booster_action(new_booster))

        self.assertEqual(passed.booster_owner[old_booster], -1)
        self.assertEqual(passed.booster_owner[new_booster], player)
        self.assertTrue(passed.players[player].passed)

    def test_random_legal_playouts_reach_terminal_state(self) -> None:
        for seed in range(12):
            rng = np.random.default_rng(seed)
            state = GaiaState.initial(2 + seed % 3, seed)
            moves = 0
            while not state.is_terminal and moves < 500:
                legal = state.legal_actions()
                self.assertTrue(legal)
                state = state.apply(int(rng.choice(legal)))
                moves += 1
            self.assertTrue(state.is_terminal)
            self.assertLess(moves, 500)
            self.assertAlmostEqual(float(state.returns().sum()), 0.0, places=5)

    def test_puct_operates_on_standard_rules(self) -> None:
        state = GaiaState.initial(3)
        result = PUCTSearch(
            GaiaHeuristicEvaluator(),
            SearchConfig(simulations=12, seed=4),
        ).run(state)

        self.assertEqual(int(result.visits.sum()), 12)
        self.assertAlmostEqual(float(result.policy.sum()), 1.0, places=6)
        self.assertEqual(result.root_value.shape, (3,))
        self.assertTrue(np.all(result.policy[~state.legal_action_mask()] == 0))


if __name__ == "__main__":
    unittest.main()
