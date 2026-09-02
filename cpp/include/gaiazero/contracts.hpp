#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string_view>

namespace gaiazero {

inline constexpr std::string_view kRulesVersion = "standard-v22";
inline constexpr std::string_view kActionTupleSchemaVersion = "action-tuple-v1";
inline constexpr std::string_view kStateHashVersion = "state-hash-v1";
inline constexpr std::size_t kMaxActionArguments = 8;

// The order is a stable audit key only. Inference uses parameterized action
// logits and the rules engine remains responsible for legal tuple generation.
enum class ActionType : std::uint16_t {
    build_mine = 0,
    place_starting_structure,
    gaia_project,
    upgrade_trading,
    upgrade_lab,
    upgrade_planetary_institute,
    upgrade_academy,
    upgrade_qic_academy,
    research,
    power_action,
    tech_take,
    federation,
    qic_academy,
    standard_tech,
    advanced_tech,
    qic_tech,
    qic_federation,
    qic_planet_types,
    booster_terraform,
    booster_range,
    pass_booster,
    pass_final,
    skip_tech_research,
    brainstone,
    terrans_gaia_credit,
    terrans_gaia_ore,
    terrans_gaia_knowledge,
    terrans_gaia_qic,
    terrans_gaia_finish,
    taklons_passive_before,
    taklons_passive_after,
    ivits_space_station,
    bal_taks_gaiaformer_qic,
    bescods_research,
    itars_burn_power,
    itars_gaia_technology,
    itars_gaia_finish,
    nevlas_power_to_gaia,
    nevlas_credits,
    nevlas_credit_ore,
    nevlas_ore,
    nevlas_qic,
    nevlas_knowledge,
    lost_planet,
    passive_charge_accept,
    passive_charge_decline,
    power_to_credit,
    power_to_ore,
    power_to_knowledge,
    power_to_qic,
    qic_to_ore,
    ore_to_credit,
    knowledge_to_credit,
    legacy_action,
};

inline constexpr std::size_t kActionTypeCount =
    static_cast<std::size_t>(ActionType::legacy_action) + 1;

constexpr std::uint16_t action_type_id(ActionType type) noexcept {
    return static_cast<std::uint16_t>(type);
}

inline constexpr std::array<std::string_view, kActionTypeCount> kActionTypeNames{
    "build_mine", "place_starting_structure", "gaia_project",
    "upgrade_trading", "upgrade_lab", "upgrade_planetary_institute",
    "upgrade_academy", "upgrade_qic_academy", "research", "power_action",
    "tech_take", "federation", "qic_academy", "standard_tech",
    "advanced_tech", "qic_tech", "qic_federation", "qic_planet_types",
    "booster_terraform", "booster_range", "pass_booster", "pass_final",
    "skip_tech_research", "brainstone", "terrans_gaia_credit",
    "terrans_gaia_ore", "terrans_gaia_knowledge", "terrans_gaia_qic",
    "terrans_gaia_finish", "taklons_passive_before", "taklons_passive_after",
    "ivits_space_station", "bal_taks_gaiaformer_qic", "bescods_research",
    "itars_burn_power", "itars_gaia_technology", "itars_gaia_finish",
    "nevlas_power_to_gaia", "nevlas_credits", "nevlas_credit_ore",
    "nevlas_ore", "nevlas_qic", "nevlas_knowledge", "lost_planet",
    "passive_charge_accept", "passive_charge_decline", "power_to_credit",
    "power_to_ore", "power_to_knowledge", "power_to_qic", "qic_to_ore",
    "ore_to_credit", "knowledge_to_credit", "legacy_action"};

constexpr bool is_valid_action_type(ActionType type) noexcept {
    return static_cast<std::size_t>(action_type_id(type)) < kActionTypeCount;
}

constexpr std::string_view action_type_name(ActionType type) noexcept {
    return is_valid_action_type(type) ? kActionTypeNames[action_type_id(type)]
                                      : std::string_view{};
}

struct ActionTuple {
    ActionType action_type{ActionType::legacy_action};
    std::array<std::int32_t, kMaxActionArguments> arguments{
        {-1, -1, -1, -1, -1, -1, -1, -1}};
    std::uint8_t argument_count{0};

    static ActionTuple create(
        ActionType type,
        std::initializer_list<std::int32_t> arguments);

    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] bool operator==(const ActionTuple& other) const noexcept;
    [[nodiscard]] bool operator!=(const ActionTuple& other) const noexcept {
        return !(*this == other);
    }
};

}  // namespace gaiazero
