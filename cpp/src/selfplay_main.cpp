#include "gaiazero/gaia_state.hpp"
#include "gaiazero/graph_encoder.hpp"
#include "gaiazero/onnxruntime_backend.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <cstdlib>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <process.h>
#else
#include <unistd.h>
#endif

namespace fs = std::filesystem;
using gaiazero::ActionTuple;
using gaiazero::ActionType;
using gaiazero::GaiaState;

namespace {

unsigned long long process_id() noexcept {
#ifdef _WIN32
    return static_cast<unsigned long long>(::_getpid());
#else
    return static_cast<unsigned long long>(::getpid());
#endif
}

constexpr std::size_t kPlayersMax = gaiazero::kMaxPlayers;
constexpr std::size_t kActionSize = gaiazero::kActionSize;
constexpr std::size_t kMaxArgs = gaiazero::kMaxActionArguments;
constexpr int kN = gaiazero::kMaxPlanets;
constexpr int kActionBuild = 0;
constexpr int kActionGaia = kActionBuild + kN;
constexpr int kActionUpgradeTrading = kActionGaia + kN;
constexpr int kActionUpgradeLab = kActionUpgradeTrading + kN;
constexpr int kActionUpgradePi = kActionUpgradeLab + kN;
constexpr int kActionUpgradeAcademy = kActionUpgradePi + kN;
constexpr int kActionUpgradeQicAcademy = kActionUpgradeAcademy + kN;
constexpr int kActionResearch = kActionUpgradeQicAcademy + kN;
constexpr int kActionPower = kActionResearch + 6;
constexpr int kActionTech = kActionPower + 7;
constexpr int kActionFederation = kActionTech + 15;
constexpr int kActionQicAcademy = kActionFederation + 6;
constexpr int kActionStandardTech = kActionQicAcademy + 1;
constexpr int kActionAdvancedTech = kActionStandardTech + 1;
constexpr int kActionQicTech = kActionAdvancedTech + 3;
constexpr int kActionQicFederation = kActionQicTech + 1;
constexpr int kActionQicPlanetTypes = kActionQicFederation + 7;
constexpr int kActionBoosterTerraform = kActionQicPlanetTypes + 1;
constexpr int kActionBoosterRange = kActionBoosterTerraform + 1;
constexpr int kActionPassBooster = kActionBoosterRange + 1;
constexpr int kActionPassFinal = kActionPassBooster + 10;
constexpr int kActionSkipResearch = kActionPassFinal;
constexpr int kActionBrainstone = kActionPassFinal + 1;
constexpr int kActionTerransCredit = kActionBrainstone + 1;
constexpr int kActionTerransOre = kActionTerransCredit + 1;
constexpr int kActionTerransKnowledge = kActionTerransOre + 1;
constexpr int kActionTerransQic = kActionTerransKnowledge + 1;
constexpr int kActionTerransFinish = kActionTerransQic + 1;
constexpr int kActionTaklonsBefore = kActionTerransFinish + 1;
constexpr int kActionTaklonsAfter = kActionTaklonsBefore + 1;
constexpr int kActionIvitsStation = kActionTaklonsAfter + 1;
constexpr int kActionBalTaks = kActionIvitsStation + gaiazero::kMaxBoardSpaces;
constexpr int kActionBescodsResearch = kActionBalTaks + 1;
constexpr int kActionItarsBurn = kActionBescodsResearch + 6;
constexpr int kActionItarsTech = kActionItarsBurn + 1;
constexpr int kActionItarsFinish = kActionItarsTech + 1;
constexpr int kActionNevlasGaia = kActionItarsFinish + 1;
constexpr int kActionNevlasCredits = kActionNevlasGaia + 1;
constexpr int kActionNevlasCreditOre = kActionNevlasCredits + 1;
constexpr int kActionNevlasOre = kActionNevlasCreditOre + 1;
constexpr int kActionNevlasQic = kActionNevlasOre + 1;
constexpr int kActionNevlasKnowledge = kActionNevlasQic + 1;
constexpr int kActionLostPlanet = kActionNevlasKnowledge + 1;
constexpr int kActionPassiveAccept = kActionLostPlanet + gaiazero::kMaxBoardSpaces;
constexpr int kActionPassiveDecline = kActionPassiveAccept + 1;
constexpr int kActionPowerCredit = kActionPassiveDecline + 1;
constexpr int kActionPowerOre = kActionPowerCredit + 1;
constexpr int kActionPowerKnowledge = kActionPowerOre + 1;
constexpr int kActionPowerQic = kActionPowerKnowledge + 1;
constexpr int kActionQicOre = kActionPowerQic + 1;
constexpr int kActionOreCredit = kActionQicOre + 1;
constexpr int kActionKnowledgeCredit = kActionOreCredit + 1;

static_assert(kActionKnowledgeCredit + 1 == static_cast<int>(kActionSize));

struct Config {
    int players{3};
    std::int64_t seed{0};
    int games{1};
    int simulations{128};
    int max_moves{512};
    int temperature_moves{24};
    double temperature{1.0};
    double c_puct{1.5};
    double dirichlet_alpha{0.3};
    double root_noise_fraction{0.25};
    double beta_vp{0.10};
    double vp_scale{20.0};
    bool root_noise{true};
    int poll_ms{1000};
    fs::path output_dir{"runs/cpp-selfplay/raw"};
    fs::path status_file{};
    fs::path model_path{};
    fs::path stop_file{};
    bool allow_uniform{true};
    bool once{false};
};

void usage() {
    std::cout << "gaiazero_selfplay [options]\n"
              << "  --players 2|3|4       player count (default 3)\n"
              << "  --games N             games per invocation (default 1)\n"
              << "  --seed N              root seed\n"
              << "  --simulations N       MCTS simulations\n"
              << "  --max-moves N         safety move limit\n"
              << "  --output DIR          raw NPZ directory\n"
              << "  --status-file FILE    atomic worker status JSON\n"
              << "  --model FILE          graph ONNX model (CPU ORT when enabled)\n"
              << "  --beta-vp N           bounded VP utility weight (default 0.10)\n"
              << "  --vp-scale N          VP utility scale (default 20)\n"
              << "  --no-root-noise       disable Dirichlet root noise\n"
              << "  --once                run the requested games and exit\n"
              << "  --poll-ms N           wait between model polls\n";
}

template <typename T>
T parse_value(const char* value, const char* name) {
    try {
        if constexpr (std::is_same_v<T, int>) return std::stoi(value);
        if constexpr (std::is_same_v<T, std::int64_t>) return std::stoll(value);
        if constexpr (std::is_same_v<T, double>) return std::stod(value);
    } catch (const std::exception&) {
        throw std::invalid_argument(std::string("invalid ") + name + ": " + value);
    }
    throw std::invalid_argument(std::string("unsupported option type: ") + name);
}

Config parse_args(int argc, char** argv) {
    Config config;
    for (int index = 1; index < argc; ++index) {
        const std::string option = argv[index];
        if (option == "--help" || option == "-h") { usage(); std::exit(0); }
        auto require = [&](const char* name) -> const char* {
            if (index + 1 >= argc) throw std::invalid_argument(std::string(name) + " requires a value");
            return argv[++index];
        };
        if (option == "--players") config.players = parse_value<int>(require("--players"), "players");
        else if (option == "--games") config.games = parse_value<int>(require("--games"), "games");
        else if (option == "--seed") config.seed = parse_value<std::int64_t>(require("--seed"), "seed");
        else if (option == "--simulations") config.simulations = parse_value<int>(require("--simulations"), "simulations");
        else if (option == "--max-moves") config.max_moves = parse_value<int>(require("--max-moves"), "max-moves");
        else if (option == "--temperature-moves") config.temperature_moves = parse_value<int>(require("--temperature-moves"), "temperature-moves");
        else if (option == "--temperature") config.temperature = parse_value<double>(require("--temperature"), "temperature");
        else if (option == "--c-puct") config.c_puct = parse_value<double>(require("--c-puct"), "c-puct");
        else if (option == "--dirichlet-alpha") config.dirichlet_alpha = parse_value<double>(require("--dirichlet-alpha"), "dirichlet-alpha");
        else if (option == "--root-noise-fraction") config.root_noise_fraction = parse_value<double>(require("--root-noise-fraction"), "root-noise-fraction");
        else if (option == "--beta-vp") config.beta_vp = parse_value<double>(require("--beta-vp"), "beta-vp");
        else if (option == "--vp-scale") config.vp_scale = parse_value<double>(require("--vp-scale"), "vp-scale");
        else if (option == "--poll-ms") config.poll_ms = parse_value<int>(require("--poll-ms"), "poll-ms");
        else if (option == "--output") config.output_dir = require("--output");
        else if (option == "--status-file") config.status_file = require("--status-file");
        else if (option == "--model") config.model_path = require("--model");
        else if (option == "--stop-file") config.stop_file = require("--stop-file");
        else if (option == "--no-root-noise") config.root_noise = false;
        else if (option == "--no-uniform") config.allow_uniform = false;
        else if (option == "--once") config.once = true;
        else throw std::invalid_argument("unknown option: " + option);
    }
    if (config.players < 2 || config.players > 4 || config.games < 1 ||
        config.simulations < 1 || config.max_moves < 1 || config.temperature_moves < 0 ||
        config.poll_ms < 1 || config.c_puct <= 0 || config.dirichlet_alpha <= 0 ||
        config.root_noise_fraction < 0 || config.root_noise_fraction > 1 ||
        config.temperature < 0 || config.beta_vp < 0 || config.vp_scale <= 0)
        throw std::invalid_argument("invalid selfplay configuration");
    if (config.stop_file.empty()) config.stop_file = config.output_dir.parent_path() / "STOP";
    if (config.status_file.empty()) config.status_file = config.output_dir.parent_path() / "status.json";
    return config;
}

std::string json_quote(std::string_view value) {
    std::ostringstream out;
    out << '"';
    for (const char ch : value) {
        switch (ch) {
        case '"': out << "\\\""; break;
        case '\\': out << "\\\\"; break;
        case '\n': out << "\\n"; break;
        case '\r': out << "\\r"; break;
        case '\t': out << "\\t"; break;
        default: out << ch; break;
        }
    }
    out << '"';
    return out.str();
}

void write_status(const Config& config, std::string_view phase, int games,
                  int moves, const fs::path& last_shard = {},
                  std::string_view error = {}) {
    if (config.status_file.empty()) return;
    std::ostringstream payload;
    payload << "{\"schema_version\":\"cpp-selfplay-status-v1\",\"worker\":\"selfplay\",\"pid\":"
            << process_id()
            << ",\"phase\":" << json_quote(phase)
            << ",\"players\":" << config.players
            << ",\"games\":" << games
            << ",\"moves\":" << moves
            << ",\"model\":" << json_quote(config.model_path.empty() ? "" : config.model_path.string())
            << ",\"last_shard\":" << json_quote(last_shard.empty() ? "" : last_shard.string())
            << ",\"error\":" << json_quote(error)
            << ",\"updated_at_unix_ms\":"
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::system_clock::now().time_since_epoch()).count()
            << '}';
    if (!config.status_file.parent_path().empty())
        fs::create_directories(config.status_file.parent_path());
    const fs::path temporary = config.status_file.string() + ".tmp-" +
        std::to_string(process_id());
    {
        std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
        if (!out) return;
        const auto text = payload.str();
        out.write(text.data(), static_cast<std::streamsize>(text.size()));
        out.flush();
    }
#ifdef _WIN32
    const auto temporary_wide = temporary.wstring();
    const auto target_wide = config.status_file.wstring();
    if (!::MoveFileExW(temporary_wide.c_str(), target_wide.c_str(),
                       MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        std::error_code ignored;
        fs::remove(temporary, ignored);
    }
#else
    std::error_code ignored;
    fs::rename(temporary, config.status_file, ignored);
    if (ignored) fs::remove(temporary, ignored);
#endif
}

std::string action_json(const ActionTuple& action) {
    std::ostringstream out;
    out << "{\"action_type\":" << json_quote(gaiazero::action_type_name(action.action_type))
        << ",\"action_type_id\":" << gaiazero::action_type_id(action.action_type)
        << ",\"args\":[";
    for (std::size_t index = 0; index < action.argument_count; ++index) {
        if (index) out << ',';
        out << action.arguments[index];
    }
    out << "],\"parameter_types\":[";
    for (std::size_t index = 0; index < action.argument_count; ++index) {
        if (index) out << ',';
        out << "\"int\"";
    }
    out << "],\"schema_version\":\"action-tuple-v1\"}";
    return out.str();
}

int legacy_action_id(const ActionTuple& action) {
    const int arg = action.argument_count ? action.arguments[0] : 0;
    switch (action.action_type) {
    case ActionType::build_mine:
    case ActionType::place_starting_structure: return kActionBuild + arg;
    case ActionType::gaia_project: return kActionGaia + arg;
    case ActionType::upgrade_trading: return kActionUpgradeTrading + arg;
    case ActionType::upgrade_lab: return kActionUpgradeLab + arg;
    case ActionType::upgrade_planetary_institute: return kActionUpgradePi + arg;
    case ActionType::upgrade_academy: return kActionUpgradeAcademy + arg;
    case ActionType::upgrade_qic_academy: return kActionUpgradeQicAcademy + arg;
    case ActionType::research: return kActionResearch + arg;
    case ActionType::power_action: return kActionPower + arg;
    case ActionType::tech_take: return kActionTech + arg;
    case ActionType::federation: return kActionFederation + arg;
    case ActionType::qic_academy: return kActionQicAcademy;
    case ActionType::standard_tech: return kActionStandardTech;
    case ActionType::advanced_tech: return kActionAdvancedTech + arg;
    case ActionType::qic_tech: return kActionQicTech;
    case ActionType::qic_federation: return kActionQicFederation + arg;
    case ActionType::qic_planet_types: return kActionQicPlanetTypes;
    case ActionType::booster_terraform: return kActionBoosterTerraform;
    case ActionType::booster_range: return kActionBoosterRange;
    case ActionType::pass_booster: return kActionPassBooster + arg;
    case ActionType::pass_final: return kActionPassFinal;
    case ActionType::skip_tech_research: return kActionSkipResearch;
    case ActionType::brainstone: return kActionBrainstone;
    case ActionType::terrans_gaia_credit: return kActionTerransCredit;
    case ActionType::terrans_gaia_ore: return kActionTerransOre;
    case ActionType::terrans_gaia_knowledge: return kActionTerransKnowledge;
    case ActionType::terrans_gaia_qic: return kActionTerransQic;
    case ActionType::terrans_gaia_finish: return kActionTerransFinish;
    case ActionType::taklons_passive_before: return kActionTaklonsBefore;
    case ActionType::taklons_passive_after: return kActionTaklonsAfter;
    case ActionType::ivits_space_station: return kActionIvitsStation + arg;
    case ActionType::bal_taks_gaiaformer_qic: return kActionBalTaks;
    case ActionType::bescods_research: return kActionBescodsResearch + arg;
    case ActionType::itars_burn_power: return kActionItarsBurn;
    case ActionType::itars_gaia_technology: return kActionItarsTech;
    case ActionType::itars_gaia_finish: return kActionItarsFinish;
    case ActionType::nevlas_power_to_gaia: return kActionNevlasGaia;
    case ActionType::nevlas_credits: return kActionNevlasCredits;
    case ActionType::nevlas_credit_ore: return kActionNevlasCreditOre;
    case ActionType::nevlas_ore: return kActionNevlasOre;
    case ActionType::nevlas_qic: return kActionNevlasQic;
    case ActionType::nevlas_knowledge: return kActionNevlasKnowledge;
    case ActionType::lost_planet: return kActionLostPlanet + arg;
    case ActionType::passive_charge_accept: return kActionPassiveAccept;
    case ActionType::passive_charge_decline: return kActionPassiveDecline;
    case ActionType::power_to_credit: return kActionPowerCredit;
    case ActionType::power_to_ore: return kActionPowerOre;
    case ActionType::power_to_knowledge: return kActionPowerKnowledge;
    case ActionType::power_to_qic: return kActionPowerQic;
    case ActionType::qic_to_ore: return kActionQicOre;
    case ActionType::ore_to_credit: return kActionOreCredit;
    case ActionType::knowledge_to_credit: return kActionKnowledgeCredit;
    default: throw std::invalid_argument("unsupported ActionTuple for legacy NPZ mapping");
    }
}

std::string action_list_json(const std::vector<ActionTuple>& actions) {
    std::ostringstream out;
    out << '[';
    for (std::size_t index = 0; index < actions.size(); ++index) {
        if (index) out << ',';
        out << action_json(actions[index]);
    }
    out << ']';
    return out.str();
}

std::string policy_targets_json(const std::vector<ActionTuple>& actions,
                                const std::vector<int>& visits,
                                const std::vector<float>& policy) {
    std::ostringstream out;
    out << '[';
    for (std::size_t index = 0; index < actions.size(); ++index) {
        if (index) out << ',';
        out << "{\"action_tuple\":" << action_json(actions[index])
            << ",\"visit_count\":" << visits[index]
            << ",\"visit_probability\":" << std::setprecision(9) << policy[index] << '}';
    }
    out << ']';
    return out.str();
}

std::string policy_type_targets_json(const std::vector<ActionTuple>& actions,
                                    const std::vector<float>& policy) {
    std::map<std::string, double> totals;
    for (std::size_t index = 0; index < actions.size() && index < policy.size(); ++index)
        totals[std::string(gaiazero::action_type_name(actions[index].action_type))] += policy[index];
    std::ostringstream out;
    out << '{';
    bool first = true;
    for (const auto& [type, value] : totals) {
        if (!first) out << ',';
        first = false;
        out << json_quote(type) << ':' << std::setprecision(9) << value;
    }
    out << '}';
    return out.str();
}

std::string policy_argument_targets_json(const std::vector<ActionTuple>& actions,
                                         const std::vector<float>& policy) {
    std::map<int, std::map<int, double>> totals;
    for (std::size_t index = 0; index < actions.size() && index < policy.size(); ++index)
        for (std::size_t slot = 0; slot < actions[index].argument_count; ++slot)
            totals[static_cast<int>(slot)][actions[index].arguments[slot]] += policy[index];
    std::ostringstream out;
    out << '{';
    bool first_slot = true;
    for (const auto& [slot, values] : totals) {
        if (!first_slot) out << ',';
        first_slot = false;
        out << json_quote(std::to_string(slot)) << ':' << '{';
        bool first_value = true;
        for (const auto& [argument, value] : values) {
            if (!first_value) out << ',';
            first_value = false;
            out << json_quote(std::to_string(argument)) << ':' << std::setprecision(9) << value;
        }
        out << '}';
    }
    out << '}';
    return out.str();
}

std::string root_visit_counts_json(const std::vector<ActionTuple>& actions,
                                   const std::vector<int>& visits) {
    std::ostringstream out;
    out << '[';
    for (std::size_t index = 0; index < actions.size(); ++index) {
        if (index) out << ',';
        out << "{\"action_tuple\":" << action_json(actions[index])
            << ",\"visit_count\":" << (index < visits.size() ? visits[index] : 0) << '}';
    }
    out << ']';
    return out.str();
}

std::string root_policy_priors_json(const std::vector<ActionTuple>& actions,
                                    const std::vector<float>& policy) {
    std::ostringstream out;
    out << '[';
    for (std::size_t index = 0; index < actions.size(); ++index) {
        if (index) out << ',';
        out << "{\"action_tuple\":" << action_json(actions[index])
            << ",\"prior_probability\":"
            << std::setprecision(9) << (index < policy.size() ? policy[index] : 0.0F) << '}';
    }
    out << ']';
    return out.str();
}

struct TraceRow {
    GaiaState state;
    std::vector<ActionTuple> legal;
    std::vector<int> visits;
    std::vector<float> policy;
    int action_id{-1};
    ActionTuple action{};
};

struct GameResult {
    int moves{0};
    fs::path shard;
};

struct SearchResult {
    std::vector<int> visits;
    std::vector<float> policy;
    std::array<float, gaiazero::kMaxPlayers> value{};
};

struct Evaluation {
    std::vector<float> priors;
    std::array<float, gaiazero::kMaxPlayers> values{};
    std::array<float, gaiazero::kMaxPlayers> vp_mean{};
    bool has_pairwise{false};
    bool has_vp_mean{false};
};

struct SearchNode;
struct Edge {
    ActionTuple action;
    float prior{0.0F};
    int visits{0};
    std::array<float, gaiazero::kMaxPlayers> value_sum{};
    std::unique_ptr<SearchNode> child;
};

struct SearchNode {
    GaiaState state;
    std::vector<Edge> edges;
    bool expanded{false};
};

class Evaluator {
public:
    explicit Evaluator(const Config& config) : config_(config) {
        reload_if_changed();
        if (!backend_ && !config_.allow_uniform)
            throw std::invalid_argument("--model is required when --no-uniform is set");
    }

    void reload_if_changed() {
        if (config_.model_path.empty()) return;
        std::error_code error;
        if (!fs::is_regular_file(config_.model_path, error)) {
            if (!config_.allow_uniform)
                throw std::runtime_error("model file is not available: " + config_.model_path.string());
            // Keep the last valid backend while a publisher atomically
            // replaces the model file.
            if (!backend_) model_mtime_.reset();
            return;
        }
        const auto mtime = fs::last_write_time(config_.model_path, error);
        if (error || (model_mtime_ && *model_mtime_ == mtime)) return;
        // Construct into a temporary first.  A partially copied or invalid
        // model never replaces the backend serving the current game.
        try {
            auto candidate = std::make_unique<gaiazero::OnnxRuntimeCpuBackend>(
                config_.model_path, gaiazero::OnnxRuntimeCpuConfig{});
            backend_ = std::move(candidate);
            model_mtime_ = mtime;
            std::cout << "[cpp-selfplay] loaded model " << config_.model_path.string() << "\n" << std::flush;
        } catch (const std::exception& error) {
            if (!backend_ && !config_.allow_uniform) throw;
            std::cerr << "[cpp-selfplay] keeping previous model; reload failed: "
                      << error.what() << "\n" << std::flush;
        }
    }

    Evaluation evaluate(const GaiaState& state,
                        const std::array<float, gaiazero::kMaxPlayers>* root_center = nullptr) {
        const auto legal = state.legal_action_tuples();
        Evaluation evaluation;
        evaluation.priors.assign(legal.size(), 1.0F);
        auto& priors = evaluation.priors;
        auto& values = evaluation.values;
        if (backend_) {
            const auto batch = gaiazero::encode_graph_batch(state);
            const auto output = backend_->infer(batch);
            if (output.action_type_logits.size() >= gaiazero::kActionTypeCount &&
                output.action_argument_logits.size() >= kMaxArgs * 128U) {
                std::vector<float> scores(legal.size(), -std::numeric_limits<float>::infinity());
                for (std::size_t i = 0; i < legal.size(); ++i) {
                    const auto& action = legal[i];
                    float score = output.action_type_logits[gaiazero::action_type_id(action.action_type)];
                    for (std::size_t slot = 0; slot < action.argument_count; ++slot) {
                        const int arg = action.arguments[slot];
                        if (arg < 0 || arg >= 128) { score = -std::numeric_limits<float>::infinity(); break; }
                        score += output.action_argument_logits[slot * 128U + static_cast<std::size_t>(arg)];
                    }
                    scores[i] = score;
                }
                const float maximum = *std::max_element(scores.begin(), scores.end());
                float total = 0.0F;
                for (std::size_t i = 0; i < scores.size(); ++i) {
                    priors[i] = std::isfinite(scores[i]) ? std::exp(scores[i] - maximum) : 0.0F;
                    total += priors[i];
                }
                if (total > 0.0F) for (float& value : priors) value /= total;
            }
            const int players = state.player_count;
            const int pair_count = players * (players - 1) / 2;
            if (static_cast<int>(output.pairwise_wdl_logits.size()) >= pair_count * 3) {
                std::array<float, gaiazero::kMaxPlayers> utility{};
                int pair = 0;
                for (int left = 0; left < players; ++left) {
                    for (int right = left + 1; right < players; ++right, ++pair) {
                        const float a = output.pairwise_wdl_logits[static_cast<std::size_t>(pair * 3)];
                        const float b = output.pairwise_wdl_logits[static_cast<std::size_t>(pair * 3 + 1)];
                        const float c = output.pairwise_wdl_logits[static_cast<std::size_t>(pair * 3 + 2)];
                        const float maximum = std::max({a, b, c});
                        const float wa = std::exp(a - maximum);
                        const float wb = std::exp(b - maximum);
                        const float wc = std::exp(c - maximum);
                        const float total = std::max(1e-12F, wa + wb + wc);
                        const float margin = (wa - wc) / total;
                        utility[static_cast<std::size_t>(left)] += margin;
                        utility[static_cast<std::size_t>(right)] -= margin;
                    }
                }
                for (int player = 0; player < players; ++player)
                    values[static_cast<std::size_t>(player)] = utility[static_cast<std::size_t>(player)] /
                        static_cast<float>(std::max(1, players - 1));
                evaluation.has_pairwise = true;
            }
            if (static_cast<int>(output.vp_belief_logits.size()) >= players * 403) {
                constexpr int bucket_count = 403;
                for (int player = 0; player < players; ++player) {
                    const auto begin = output.vp_belief_logits.begin() + player * bucket_count;
                    const auto end = begin + bucket_count;
                    const float maximum = *std::max_element(begin, end);
                    double denominator = 0.0;
                    double numerator = 0.0;
                    for (auto cursor = begin; cursor != end; ++cursor) {
                        const double weight = std::exp(static_cast<double>(*cursor - maximum));
                        denominator += weight;
                        const auto index = static_cast<int>(cursor - begin);
                        const int representative = index == 0 ? -201 : index == 402 ? 201 : index - 201;
                        numerator += weight * representative;
                    }
                    evaluation.vp_mean[static_cast<std::size_t>(player)] =
                        static_cast<float>(numerator / std::max(1e-12, denominator));
                }
                evaluation.has_vp_mean = true;
            }
        }
        const auto scores = state.final_scores();
        if (!evaluation.has_pairwise) {
            double mean = 0.0;
            for (int p = 0; p < state.player_count; ++p) mean += scores[static_cast<std::size_t>(p)];
            mean /= static_cast<double>(state.player_count);
            double scale = 18.0;
            for (int p = 0; p < state.player_count; ++p)
                scale = std::max(scale, std::abs(scores[static_cast<std::size_t>(p)] - mean));
            for (int p = 0; p < state.player_count; ++p)
                values[static_cast<std::size_t>(p)] = static_cast<float>(std::tanh((scores[static_cast<std::size_t>(p)] - mean) / scale));
        }
        if (evaluation.has_vp_mean && root_center != nullptr) {
            for (int p = 0; p < state.player_count; ++p)
                values[static_cast<std::size_t>(p)] += static_cast<float>(config_.beta_vp *
                    std::tanh((evaluation.vp_mean[static_cast<std::size_t>(p)] - (*root_center)[static_cast<std::size_t>(p)]) /
                              config_.vp_scale));
        }
        return evaluation;
    }

    [[nodiscard]] fs::path active_model_path() const {
        return backend_ ? config_.model_path : fs::path{};
    }

private:
    const Config& config_;
    std::unique_ptr<gaiazero::InferenceBackend> backend_;
    std::optional<fs::file_time_type> model_mtime_;
};

class Search {
public:
    Search(Evaluator& evaluator, const Config& config, std::uint64_t seed)
        : evaluator_(evaluator), config_(config), rng_(seed) {}

    SearchResult run(const GaiaState& root_state, double temperature) {
        const auto legal = root_state.legal_action_tuples();
        if (legal.empty()) throw std::invalid_argument("cannot search a state without legal actions");
        SearchNode root{root_state};
        const auto root_evaluation = evaluator_.evaluate(root_state);
        std::array<float, gaiazero::kMaxPlayers> root_center{};
        if (root_evaluation.has_vp_mean) {
            root_center = root_evaluation.vp_mean;
        } else {
            const auto scores = root_state.final_scores();
            for (int p = 0; p < root_state.player_count; ++p)
                root_center[static_cast<std::size_t>(p)] = static_cast<float>(scores[static_cast<std::size_t>(p)]);
        }
        expand(root, &root_center, &root_evaluation);
        if (config_.root_noise && config_.root_noise_fraction > 0.0)
            add_root_noise(root);
        for (int simulation = 0; simulation < config_.simulations; ++simulation) {
            std::vector<Edge*> path;
            SearchNode* node = &root;
            std::array<float, gaiazero::kMaxPlayers> value{};
            while (true) {
                if (node->state.is_terminal()) { value = terminal_value(node->state, root_center); break; }
                if (!node->expanded) { value = expand(*node, &root_center); break; }
                Edge* edge = select(*node);
                path.push_back(edge);
                if (!edge->child) edge->child = std::make_unique<SearchNode>(SearchNode{node->state.apply(edge->action)});
                node = edge->child.get();
            }
            for (Edge* edge : path) {
                ++edge->visits;
                for (int p = 0; p < gaiazero::kMaxPlayers; ++p) edge->value_sum[static_cast<std::size_t>(p)] += value[static_cast<std::size_t>(p)];
            }
        }
        SearchResult result;
        result.visits.assign(legal.size(), 0);
        result.policy.assign(legal.size(), 0.0F);
        for (std::size_t i = 0; i < root.edges.size(); ++i) result.visits[i] = root.edges[i].visits;
        if (temperature <= 1e-6) {
            const auto best = std::max_element(result.visits.begin(), result.visits.end());
            result.policy[static_cast<std::size_t>(std::distance(result.visits.begin(), best))] = 1.0F;
        } else {
            double total = 0.0;
            for (std::size_t i = 0; i < result.visits.size(); ++i) {
                result.policy[i] = static_cast<float>(std::pow(static_cast<double>(result.visits[i]), 1.0 / temperature));
                total += result.policy[i];
            }
            if (total <= 0.0) for (float& value : result.policy) value = 1.0F / static_cast<float>(result.policy.size());
            else for (float& value : result.policy) value = static_cast<float>(value / total);
        }
        for (const auto& edge : root.edges) for (int p = 0; p < gaiazero::kMaxPlayers; ++p)
            result.value[static_cast<std::size_t>(p)] += edge.value_sum[static_cast<std::size_t>(p)];
        const float visits = static_cast<float>(std::max(1, config_.simulations));
        for (int p = 0; p < gaiazero::kMaxPlayers; ++p) result.value[static_cast<std::size_t>(p)] /= visits;
        return result;
    }

    std::size_t sample(const std::vector<float>& policy) {
        std::discrete_distribution<std::size_t> distribution(policy.begin(), policy.end());
        return distribution(rng_);
    }

private:
    std::array<float, gaiazero::kMaxPlayers> terminal_value(
        const GaiaState& state,
        const std::array<float, gaiazero::kMaxPlayers>& root_center) const {
        std::array<float, gaiazero::kMaxPlayers> result{};
        const auto scores = state.final_scores();
        for (int p = 0; p < state.player_count; ++p) for (int q = 0; q < state.player_count; ++q) {
            if (p == q) continue;
            result[static_cast<std::size_t>(p)] += scores[static_cast<std::size_t>(p)] > scores[static_cast<std::size_t>(q)] ? 1.0F : scores[static_cast<std::size_t>(p)] < scores[static_cast<std::size_t>(q)] ? -1.0F : 0.0F;
        }
        for (int p = 0; p < state.player_count; ++p) result[static_cast<std::size_t>(p)] /= static_cast<float>(std::max(1, state.player_count - 1));
        for (int p = 0; p < state.player_count; ++p)
            result[static_cast<std::size_t>(p)] += static_cast<float>(config_.beta_vp *
                std::tanh((static_cast<float>(scores[static_cast<std::size_t>(p)]) - root_center[static_cast<std::size_t>(p)]) /
                          config_.vp_scale));
        return result;
    }

    std::array<float, gaiazero::kMaxPlayers> expand(
        SearchNode& node,
        const std::array<float, gaiazero::kMaxPlayers>* root_center,
        const Evaluation* provided = nullptr) {
        const auto legal = node.state.legal_action_tuples();
        if (legal.empty()) {
            std::array<float, gaiazero::kMaxPlayers> center{};
            if (root_center) center = *root_center;
            return terminal_value(node.state, center);
        }
        const auto evaluation = provided != nullptr
            ? *provided
            : evaluator_.evaluate(node.state, root_center);
        const auto& priors = evaluation.priors;
        const auto& value = evaluation.values;
        node.edges.clear();
        node.edges.reserve(legal.size());
        for (std::size_t i = 0; i < legal.size(); ++i) {
            Edge edge;
            edge.action = legal[i];
            edge.prior = i < priors.size() ? priors[i] : 1.0F / static_cast<float>(legal.size());
            node.edges.push_back(std::move(edge));
        }
        float total = 0.0F;
        for (const Edge& edge : node.edges) total += std::max(0.0F, edge.prior);
        if (total <= 0.0F) for (Edge& edge : node.edges) edge.prior = 1.0F / static_cast<float>(node.edges.size());
        else for (Edge& edge : node.edges) edge.prior = std::max(0.0F, edge.prior) / total;
        node.expanded = true;
        return value;
    }

    Edge* select(SearchNode& node) const {
        Edge* best = nullptr;
        float score = -std::numeric_limits<float>::infinity();
        const float exploration = static_cast<float>(config_.c_puct * std::sqrt(static_cast<double>(1 + total_visits(node))));
        for (Edge& edge : node.edges) {
            const float q = edge.visits == 0 ? 0.0F : edge.value_sum[static_cast<std::size_t>(node.state.player_to_move)] / static_cast<float>(edge.visits);
            const float candidate = q + exploration * edge.prior / static_cast<float>(1 + edge.visits);
            if (candidate > score) { score = candidate; best = &edge; }
        }
        if (!best) throw std::logic_error("MCTS selection found no edge");
        return best;
    }

    static int total_visits(const SearchNode& node) {
        int total = 0;
        for (const Edge& edge : node.edges) total += edge.visits;
        return total;
    }

    void add_root_noise(SearchNode& root) {
        std::gamma_distribution<double> gamma(config_.dirichlet_alpha, 1.0);
        std::vector<double> noise(root.edges.size());
        double total = 0.0;
        for (double& value : noise) { value = gamma(rng_); total += value; }
        if (total <= 0.0) return;
        for (std::size_t i = 0; i < root.edges.size(); ++i)
            root.edges[i].prior = static_cast<float>((1.0 - config_.root_noise_fraction) * root.edges[i].prior + config_.root_noise_fraction * noise[i] / total);
    }

    Evaluator& evaluator_;
    const Config& config_;
    std::mt19937_64 rng_;
};

// Minimal ZIP writer with stored members. NumPy accepts stored .npy members,
// and avoiding a third-party zip dependency keeps the C++ worker portable.
std::uint32_t crc32(const std::vector<std::uint8_t>& data) {
    std::uint32_t crc = 0xFFFFFFFFU;
    for (const std::uint8_t byte : data) {
        crc ^= byte;
        for (int bit = 0; bit < 8; ++bit) crc = (crc >> 1U) ^ (0xEDB88320U & static_cast<std::uint32_t>(-(static_cast<int>(crc & 1U))));
    }
    return ~crc;
}

void write_u16(std::ofstream& out, std::uint16_t value) { out.put(static_cast<char>(value & 0xFFU)); out.put(static_cast<char>((value >> 8U) & 0xFFU)); }
void write_u32(std::ofstream& out, std::uint32_t value) { for (int i = 0; i < 4; ++i) out.put(static_cast<char>((value >> (8 * i)) & 0xFFU)); }

struct ZipMember { std::string name; std::vector<std::uint8_t> data; std::uint32_t crc{0}; std::uint32_t offset{0}; };

void write_zip(const fs::path& path, std::vector<ZipMember> members) {
    fs::create_directories(path.parent_path());
    const fs::path temporary = path.string() + ".tmp-" + std::to_string(std::uint64_t(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("cannot create NPZ temporary file: " + temporary.string());
    for (auto& member : members) {
        member.crc = crc32(member.data);
        member.offset = static_cast<std::uint32_t>(out.tellp());
        write_u32(out, 0x04034B50U); write_u16(out, 20); write_u16(out, 0); write_u16(out, 0);
        write_u16(out, 0); write_u16(out, 0); write_u32(out, member.crc);
        write_u32(out, static_cast<std::uint32_t>(member.data.size())); write_u32(out, static_cast<std::uint32_t>(member.data.size()));
        write_u16(out, static_cast<std::uint16_t>(member.name.size())); write_u16(out, 0);
        out.write(member.name.data(), static_cast<std::streamsize>(member.name.size()));
        if (!member.data.empty()) out.write(reinterpret_cast<const char*>(member.data.data()), static_cast<std::streamsize>(member.data.size()));
    }
    const auto directory_offset = static_cast<std::uint32_t>(out.tellp());
    for (const auto& member : members) {
        write_u32(out, 0x02014B50U); write_u16(out, 20); write_u16(out, 20); write_u16(out, 0); write_u16(out, 0);
        write_u16(out, 0); write_u16(out, 0); write_u32(out, member.crc);
        write_u32(out, static_cast<std::uint32_t>(member.data.size())); write_u32(out, static_cast<std::uint32_t>(member.data.size()));
        write_u16(out, static_cast<std::uint16_t>(member.name.size())); write_u16(out, 0); write_u16(out, 0);
        write_u16(out, 0); write_u16(out, 0); write_u32(out, 0); write_u32(out, member.offset);
        out.write(member.name.data(), static_cast<std::streamsize>(member.name.size()));
    }
    const auto directory_size = static_cast<std::uint32_t>(out.tellp()) - directory_offset;
    write_u32(out, 0x06054B50U); write_u16(out, 0); write_u16(out, 0);
    write_u16(out, static_cast<std::uint16_t>(members.size())); write_u16(out, static_cast<std::uint16_t>(members.size()));
    write_u32(out, directory_size); write_u32(out, directory_offset); write_u16(out, 0);
    out.flush(); out.close();
    if (!out) throw std::runtime_error("failed writing NPZ: " + path.string());
    fs::rename(temporary, path);
}

template <typename T>
std::vector<std::uint8_t> bytes_of(const std::vector<T>& values) {
    std::vector<std::uint8_t> bytes(values.size() * sizeof(T));
    if (!bytes.empty()) std::memcpy(bytes.data(), values.data(), bytes.size());
    return bytes;
}

std::vector<std::uint8_t> npy_header(std::string descr, std::string shape) {
    std::string header = "{'descr': '" + descr + "', 'fortran_order': False, 'shape': " + shape + ", }";
    const std::size_t preamble = 10;
    const std::size_t padding = 16 - ((preamble + header.size() + 1) % 16);
    header.append(padding, ' '); header.push_back('\n');
    std::vector<std::uint8_t> result;
    result.insert(result.end(), {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0});
    const auto length = static_cast<std::uint16_t>(header.size());
    result.push_back(static_cast<std::uint8_t>(length & 0xFFU)); result.push_back(static_cast<std::uint8_t>(length >> 8U));
    result.insert(result.end(), header.begin(), header.end());
    return result;
}

template <typename T>
std::vector<std::uint8_t> npy_numeric(const std::vector<T>& values, std::string descr, std::string shape) {
    auto result = npy_header(std::move(descr), std::move(shape));
    const auto payload = bytes_of(values); result.insert(result.end(), payload.begin(), payload.end()); return result;
}

std::vector<std::uint8_t> npy_strings(const std::vector<std::string>& values) {
    std::size_t width = 1;
    for (const auto& value : values) width = std::max(width, value.size());
    auto result = npy_header("<U" + std::to_string(width), "(" + std::to_string(values.size()) + ",)");
    for (const auto& value : values) {
        for (unsigned char ch : value) {
            const std::uint32_t code = ch;
            result.push_back(static_cast<std::uint8_t>(code & 0xFFU)); result.push_back(static_cast<std::uint8_t>((code >> 8U) & 0xFFU));
            result.push_back(static_cast<std::uint8_t>((code >> 16U) & 0xFFU)); result.push_back(static_cast<std::uint8_t>((code >> 24U) & 0xFFU));
        }
        for (std::size_t pad = value.size(); pad < width; ++pad) result.insert(result.end(), 4, 0);
    }
    return result;
}

std::vector<std::uint8_t> npy_bool(const std::vector<std::uint8_t>& values, std::string shape) {
    auto result = npy_header("|b1", std::move(shape)); result.insert(result.end(), values.begin(), values.end()); return result;
}

std::string metadata_json(const GaiaState& initial, const fs::path& model, std::string_view game_id,
                          int moves, const std::array<double, kPlayersMax>& scores,
                          const std::array<float, kPlayersMax>& returns) {
    std::ostringstream out;
    out << "{\"action_schema_version\":\"action-tuple-v1\",\"game_id\":" << json_quote(game_id)
        << ",\"kind\":\"selfplay-game\",\"model\":" << json_quote(model.empty() ? "uniform-bootstrap" : model.string())
        << ",\"player_count\":" << initial.player_count << ",\"players\":" << initial.player_count
        << ",\"rules_version\":\"standard-v22\",\"schema_version\":\"npz-trajectory-v1\",\"seed\":" << initial.setup_seed
        << ",\"setup_hash\":" << json_quote(initial.setup_hash)
        << ",\"setup_seed_stream_version\":" << json_quote(initial.setup_seed_stream_version)
        << ",\"state_hash_version\":\"state-hash-v1\",\"terminal_valid\":true,\"trace_alignment\":\"pre_action_positions_plus_terminal\",\"trace_length\":" << (moves + 1)
        << ",\"moves\":" << moves << ",\"scores\":[";
    for (int p = 0; p < initial.player_count; ++p) { if (p) out << ','; out << scores[static_cast<std::size_t>(p)]; }
    out << "],\"returns\":[";
    for (int p = 0; p < initial.player_count; ++p) { if (p) out << ','; out << returns[static_cast<std::size_t>(p)]; }
    out << "]}";
    return out.str();
}

std::vector<std::array<int, 2>> snapshot_board_spaces(const GaiaState& state) {
    std::vector<std::array<int, 2>> result;
    for (int sector = 0; sector < state.sector_count; ++sector) {
        const auto center = state.sector_centers[static_cast<std::size_t>(sector)];
        for (int q = -2; q <= 2; ++q) for (int r = -2; r <= 2; ++r)
            if (std::max({std::abs(q), std::abs(r), std::abs(q + r)}) <= 2)
                result.push_back({center[0] + q, center[1] + r});
    }
    std::sort(result.begin(), result.end());
    return result;
}

std::string terrain_name(int terrain) {
    static constexpr std::array<std::string_view, 10> names{
        "terra", "desert", "swamp", "volcanic", "oxide", "titanium", "ice", "transdim", "gaia", "lost"};
    return terrain >= 0 && terrain < static_cast<int>(names.size()) ? std::string(names[static_cast<std::size_t>(terrain)]) : "empty";
}

std::string building_name(int building) {
    static constexpr std::array<std::string_view, 6> names{
        "empty", "mine", "trading_station", "research_lab", "planetary_institute", "academy"};
    return building >= 0 && building < static_cast<int>(names.size()) ? std::string(names[static_cast<std::size_t>(building)]) : "empty";
}

std::string snapshot_json(const GaiaState& state) {
    static constexpr std::array<std::string_view, 14> faction_names{
        "Terrans", "Lantids", "Xenos", "Gleens", "Taklons", "Ambas", "Hadsch Hallas",
        "Ivits", "Geodens", "Bal T'aks", "Firaks", "Bescods", "Nevlas", "Itars"};
    std::ostringstream out;
    out << "{\"ruleset\":\"standard-v22\",\"rules_version\":\"standard-v22\",\"state_hash_version\":\"state-hash-v1\",\"state_hash\":"
        << json_quote(state.state_hash()) << ",\"round\":" << std::max(0, std::min(state.round_number, 6))
        << ",\"max_rounds\":6,\"phase\":"
        << json_quote(state.is_starting_placement() ? "starting_placement" : state.is_booster_selection() ? "booster_selection" : state.is_terminal() ? "terminal" : "round")
        << ",\"current_player\":" << (state.is_terminal() ? -1 : state.player_to_move)
        << ",\"first_player\":" << state.first_player << ",\"terminal\":" << (state.is_terminal() ? "true" : "false")
        << ",\"scores\":[";
    const auto scores = state.final_scores();
    for (int p = 0; p < state.player_count; ++p) { if (p) out << ','; out << scores[static_cast<std::size_t>(p)]; }
    out << "],\"players\":[";
    for (int p = 0; p < state.player_count; ++p) {
        if (p) out << ',';
        const auto& info = state.players[static_cast<std::size_t>(p)];
        const int faction = info.faction;
        out << "{\"id\":" << p << ",\"faction_id\":" << faction << ",\"faction\":"
            << json_quote(faction >= 0 && faction < static_cast<int>(faction_names.size()) ? faction_names[static_cast<std::size_t>(faction)] : "Unknown")
            << ",\"credits\":" << info.credits << ",\"ore\":" << info.ore << ",\"knowledge\":" << info.knowledge
            << ",\"qic\":" << info.qic << ",\"vp\":" << info.vp << ",\"power\":[" << info.bowl_one << ',' << info.bowl_two << ',' << info.bowl_three
            << "],\"gaia_power\":" << info.gaia_power << ",\"brainstone_bowl\":" << info.brainstone_bowl
            << ",\"tracks\":[";
        for (int track = 0; track < 6; ++track) { if (track) out << ','; out << info.tracks[static_cast<std::size_t>(track)]; }
        out << "],\"satellites\":" << info.satellites << ",\"federations\":" << info.federation_tokens
            << ",\"federation_keys\":" << info.federation_keys << ",\"federation_unused\":" << info.federation_keys
            << ",\"federation_used\":" << std::max(0, info.federation_tokens - info.federation_keys)
            << ",\"colonized_types\":" << info.colonized_types << ",\"passed\":" << (info.passed ? "true" : "false")
            << ",\"tech_tiles\":[";
        bool first = true;
        for (int tile = 0; tile < 9; ++tile) if (info.tech_tiles & (1U << tile)) { if (!first) out << ','; first = false; out << tile; }
        out << "],\"covered_tech_tiles\":[],\"advanced_tech_tiles\":["; first = true;
        for (int tile = 0; tile < 15; ++tile) if (info.advanced_tech_tiles & (1U << tile)) { if (!first) out << ','; first = false; out << tile; }
        out << "]}";
    }
    out << "],\"planets\":[";
    bool first_planet = true;
    for (int planet = 0; planet < kN; ++planet) if (state.active_planets[static_cast<std::size_t>(planet)]) {
        if (!first_planet) out << ','; first_planet = false;
        const auto index = static_cast<std::size_t>(planet);
        out << "{\"id\":" << planet << ",\"q\":" << state.planet_q[index] << ",\"r\":" << state.planet_r[index]
            << ",\"source_q\":" << state.planet_source_q[index] << ",\"source_r\":" << state.planet_source_r[index]
            << ",\"source_id\":" << state.planet_source_ids[index] << ",\"sector\":" << state.planet_sectors[index]
            << ",\"terrain\":" << state.terrains[index] << ",\"terrain_name\":" << json_quote(terrain_name(state.terrains[index]))
            << ",\"owner\":" << state.owners[index] << ",\"building\":" << json_quote(building_name(state.buildings[index]))
            << ",\"coexisting_mine_owner\":" << state.coexisting_mine_owner[index]
            << ",\"coexisting_mine_federated\":" << (state.coexisting_mine_federated[index] ? "true" : "false")
            << ",\"gaiaformer\":" << state.gaiaformer_owner[index] << ",\"federated\":" << (state.federated[index] ? "true" : "false") << '}';
    }
    out << "],\"satellites\":[";
    const auto spaces = snapshot_board_spaces(state);
    bool first_piece = true;
    for (std::size_t space = 0; space < spaces.size() && space < state.satellite_owners.size(); ++space) {
        const auto owners = state.satellite_owners[space];
        if (!owners) continue;
        if (!first_piece) out << ','; first_piece = false;
        out << "{\"id\":" << space << ",\"q\":" << spaces[space][0] << ",\"r\":" << spaces[space][1] << ",\"owners\":[";
        bool first_owner = true;
        for (int p = 0; p < state.player_count; ++p) if (owners & (1 << p)) { if (!first_owner) out << ','; first_owner = false; out << p; }
        out << "]}";
    }
    out << "],\"space_stations\":["; first_piece = true;
    for (std::size_t space = 0; space < spaces.size() && space < state.space_station_owner.size(); ++space) {
        const int owner = state.space_station_owner[space]; if (owner < 0) continue;
        if (!first_piece) out << ','; first_piece = false;
        out << "{\"id\":" << space << ",\"q\":" << spaces[space][0] << ",\"r\":" << spaces[space][1] << ",\"owner\":" << owner
            << ",\"federated\":" << (state.space_station_federated[space] ? "true" : "false") << '}';
    }
    out << "],\"setup\":{\"seed\":" << state.setup_seed << ",\"hash\":" << json_quote(state.setup_hash)
        << ",\"map\":{\"method\":" << json_quote(state.map_mode) << ",\"size\":" << json_quote(state.sector_count == 7 ? "reduced" : "normal") << ",\"sector_count\":" << state.sector_count << ",\"sectors\":[";
    for (int sector = 0; sector < state.sector_count; ++sector) {
        if (sector) out << ',';
        out << "{\"position\":" << sector << ",\"tile\":" << state.sector_tiles[static_cast<std::size_t>(sector)] + 1
            << ",\"rotation\":" << state.sector_rotations[static_cast<std::size_t>(sector)] * 60
            << ",\"q\":" << state.sector_centers[static_cast<std::size_t>(sector)][0]
            << ",\"r\":" << state.sector_centers[static_cast<std::size_t>(sector)][1] << '}';
    }
    out << "],\"planet_sources\":[";
    for (int i = 0; i < state.planet_source_catalog_length; ++i) {
        if (i) out << ',';
        const auto& source = state.planet_source_catalog[static_cast<std::size_t>(i)];
        out << "{\"id\":" << source[0] << ",\"q\":" << source[1] << ",\"r\":" << source[2] << ",\"terrain\":" << source[3] << ",\"sector\":" << source[4] << '}';
    }
    out << "]},\"factions\":[";
    for (int p = 0; p < state.player_count; ++p) {
        if (p) out << ',';
        const int faction = state.players[static_cast<std::size_t>(p)].faction;
        out << "{\"player\":" << p << ",\"id\":" << faction << ",\"name\":"
            << json_quote(faction >= 0 && faction < static_cast<int>(faction_names.size()) ? faction_names[static_cast<std::size_t>(faction)] : "Unknown") << '}';
    }
    out << "],\"faction_catalog\":[],\"round_scoring\":[";
    static constexpr std::array<std::string_view, 10> round_keys{
        "terraform-2", "research-2", "mine-2", "federation-5", "trading-3",
        "trading-4", "gaia-mine-3", "gaia-mine-4", "big-5a", "big-5b"};
    static constexpr std::array<int, 10> round_points{{2, 2, 2, 5, 3, 4, 3, 4, 5, 5}};
    for (std::size_t round = 0; round < state.round_scoring_tiles.size(); ++round) {
        if (round) out << ',';
        const int tile = state.round_scoring_tiles[round];
        out << "{\"round\":" << round + 1 << ",\"id\":" << tile
            << ",\"key\":" << json_quote(tile >= 0 && tile < static_cast<int>(round_keys.size()) ? round_keys[static_cast<std::size_t>(tile)] : "round")
            << ",\"label\":\"round scoring\",\"points\":" << (tile >= 0 && tile < static_cast<int>(round_points.size()) ? round_points[static_cast<std::size_t>(tile)] : 0) << '}';
    }
    out << "],\"final_scoring\":[";
    static constexpr std::array<std::string_view, 6> final_keys{
        "federation-structures", "structures", "planet-types", "gaia-planets", "sectors", "satellites"};
    for (std::size_t index = 0; index < state.final_scoring_tiles.size(); ++index) {
        if (index) out << ',';
        const int tile = state.final_scoring_tiles[index];
        out << "{\"id\":" << tile << ",\"key\":"
            << json_quote(tile >= 0 && tile < static_cast<int>(final_keys.size()) ? final_keys[static_cast<std::size_t>(tile)] : "final-scoring")
            << ",\"label\":\"final scoring\"}";
    }
    out << "],\"standard_tech\":[";
    static constexpr std::array<std::string_view, 6> track_names{
        "terraforming", "navigation", "artificial_intelligence", "gaia_project", "economy", "science"};
    for (int space = 0; space < 9; ++space) {
        if (space) out << ',';
        const int tile = state.standard_tech_tiles[static_cast<std::size_t>(space)];
        out << "{\"space\":" << space << ",\"track\":"
            << (space < 6 ? json_quote(track_names[static_cast<std::size_t>(space)]) : "null")
            << ",\"id\":" << tile << ",\"key\":\"standard-tech\",\"label\":\"standard technology\"}";
    }
    out << "],\"advanced_tech\":[";
    for (int track = 0; track < 6; ++track) {
        if (track) out << ',';
        const int tile = state.advanced_tech_tiles[static_cast<std::size_t>(track)];
        out << "{\"track\":" << json_quote(track_names[static_cast<std::size_t>(track)]) << ",\"id\":" << tile
            << ",\"key\":\"advanced-tech\",\"label\":\"advanced technology\"}";
    }
    out << "],\"terraforming_federation\":{\"id\":" << state.terraforming_federation_tile
        << ",\"key\":\"terraforming-federation\",\"label\":\"terraforming federation\"},\"federation_supply\":[";
    for (std::size_t index = 0; index < state.federation_tile_supply.size(); ++index) {
        if (index) out << ',';
        out << state.federation_tile_supply[index];
    }
    out << "],\"boosters\":[";
    bool first_booster = true;
    for (int booster = 0; booster < gaiazero::kBoosterCount; ++booster) if (state.booster_owner[static_cast<std::size_t>(booster)] != -2) {
        if (!first_booster) out << ','; first_booster = false;
        out << "{\"id\":" << booster << ",\"owner\":" << state.booster_owner[static_cast<std::size_t>(booster)] << '}';
    }
    out << "]}}";
    return out.str();
}

void write_game_npz(const fs::path& destination, const GaiaState& initial,
                    const std::vector<TraceRow>& trace, const fs::path& model) {
    const std::size_t positions = trace.size() - 1;
    const std::size_t obs_size = initial.observation_size();
    std::vector<float> observations(positions * obs_size);
    std::vector<std::uint8_t> legal_masks(positions * kActionSize, 0);
    std::vector<float> policy_targets(positions * kActionSize, 0.0F);
    std::vector<float> value_targets(positions * initial.player_count, 0.0F);
    std::vector<float> pairwise_wdl_targets(
        positions * static_cast<std::size_t>(initial.player_count) *
        static_cast<std::size_t>(initial.player_count) * 3U, 0.0F);
    std::vector<float> final_utility_targets(
        positions * static_cast<std::size_t>(initial.player_count), 0.0F);
    std::vector<float> final_vp_belief_targets(
        positions * static_cast<std::size_t>(initial.player_count) * 403U, 0.0F);
    std::vector<std::string> state_trace, state_snapshots, state_hashes, action_tuples, legal_json, policy_json;
    std::vector<std::string> policy_type, policy_argument, root_visits, root_priors;
    std::vector<std::int16_t> action_type_ids(trace.size(), -1), action_args(trace.size() * kMaxArgs, -1);
    std::vector<std::uint8_t> action_arg_mask(trace.size() * kMaxArgs, 0);
    std::vector<std::int64_t> action_ids(trace.size(), -1), position_index(trace.size()), semantic_index(trace.size());
    std::vector<std::int16_t> rounds(trace.size()); std::vector<std::int8_t> players(trace.size());
    const auto scores = trace.back().state.final_scores();
    std::array<float, kPlayersMax> returns{};
    for (int p = 0; p < initial.player_count; ++p) for (int q = 0; q < initial.player_count; ++q) if (p != q)
        returns[static_cast<std::size_t>(p)] += scores[static_cast<std::size_t>(p)] > scores[static_cast<std::size_t>(q)] ? 1.0F : scores[static_cast<std::size_t>(p)] < scores[static_cast<std::size_t>(q)] ? -1.0F : 0.0F;
    for (int p = 0; p < initial.player_count; ++p) returns[static_cast<std::size_t>(p)] /= static_cast<float>(std::max(1, initial.player_count - 1));
    for (std::size_t i = 0; i < positions; ++i)
        for (int p = 0; p < initial.player_count; ++p)
            value_targets[i * static_cast<std::size_t>(initial.player_count) + static_cast<std::size_t>(p)] = returns[static_cast<std::size_t>(p)];
    for (std::size_t i = 0; i < positions; ++i) {
        for (int player = 0; player < initial.player_count; ++player) {
            final_utility_targets[i * static_cast<std::size_t>(initial.player_count) +
                                  static_cast<std::size_t>(player)] = returns[static_cast<std::size_t>(player)];
            const auto score = static_cast<int>(std::llround(scores[static_cast<std::size_t>(player)]));
            const int bucket = score < -200 ? 0 : score > 200 ? 402 : score + 201;
            final_vp_belief_targets[
                (i * static_cast<std::size_t>(initial.player_count) + static_cast<std::size_t>(player)) * 403U +
                static_cast<std::size_t>(bucket)] = 1.0F;
            for (int opponent = 0; opponent < initial.player_count; ++opponent) {
                if (player == opponent) continue;
                const float result = scores[static_cast<std::size_t>(player)] > scores[static_cast<std::size_t>(opponent)]
                    ? 1.0F : scores[static_cast<std::size_t>(player)] < scores[static_cast<std::size_t>(opponent)] ? -1.0F : 0.0F;
                const std::size_t base = ((i * static_cast<std::size_t>(initial.player_count) +
                                           static_cast<std::size_t>(player)) * static_cast<std::size_t>(initial.player_count) +
                                          static_cast<std::size_t>(opponent)) * 3U;
                pairwise_wdl_targets[base + static_cast<std::size_t>(result > 0.0F ? 0 : result < 0.0F ? 2 : 1)] = 1.0F;
            }
        }
    }
    for (std::size_t i = 0; i < trace.size(); ++i) {
        const auto& row = trace[i];
        const auto canonical = row.state.canonical_json();
        state_trace.push_back(canonical); state_snapshots.push_back(snapshot_json(row.state)); state_hashes.push_back(row.state.state_hash());
        action_tuples.push_back(i < positions ? action_json(row.action) : std::string{});
        legal_json.push_back(action_list_json(row.legal)); policy_json.push_back(policy_targets_json(row.legal, row.visits, row.policy));
        policy_type.push_back(policy_type_targets_json(row.legal, row.policy));
        policy_argument.push_back(policy_argument_targets_json(row.legal, row.policy));
        root_visits.push_back(root_visit_counts_json(row.legal, row.visits));
        root_priors.push_back(root_policy_priors_json(row.legal, row.policy));
        position_index[i] = static_cast<std::int64_t>(i); semantic_index[i] = static_cast<std::int64_t>(i); rounds[i] = static_cast<std::int16_t>(row.state.round_number); players[i] = static_cast<std::int8_t>(row.state.player_to_move);
        if (i < positions) {
            const auto observation = row.state.observation(); std::copy(observation.begin(), observation.end(), observations.begin() + static_cast<std::ptrdiff_t>(i * obs_size));
            for (std::size_t j = 0; j < row.legal.size(); ++j) {
                const int id = legacy_action_id(row.legal[j]);
                if (id < 0 || id >= static_cast<int>(kActionSize)) throw std::logic_error("legacy action id outside NPZ action space");
                legal_masks[i * kActionSize + static_cast<std::size_t>(id)] = 1;
                policy_targets[i * kActionSize + static_cast<std::size_t>(id)] = row.policy[j];
            }
            action_ids[i] = row.action_id; action_type_ids[i] = static_cast<std::int16_t>(gaiazero::action_type_id(row.action.action_type));
            for (std::size_t arg = 0; arg < row.action.argument_count; ++arg) { action_args[i * kMaxArgs + arg] = static_cast<std::int16_t>(row.action.arguments[arg]); action_arg_mask[i * kMaxArgs + arg] = 1; }
        }
    }
    std::ostringstream meta; meta << metadata_json(initial, model, destination.stem().string(), static_cast<int>(positions), scores, returns);
    std::vector<ZipMember> members;
    members.push_back({"observations.npy", npy_numeric(observations, "<f4", "(" + std::to_string(positions) + "," + std::to_string(obs_size) + ")")});
    members.push_back({"legal_masks.npy", npy_bool(legal_masks, "(" + std::to_string(positions) + "," + std::to_string(kActionSize) + ")")});
    members.push_back({"policy_targets.npy", npy_numeric(policy_targets, "<f4", "(" + std::to_string(positions) + "," + std::to_string(kActionSize) + ")")});
    members.push_back({"value_targets.npy", npy_numeric(value_targets, "<f4", "(" + std::to_string(positions) + "," + std::to_string(initial.player_count) + ")")});
    members.push_back({"pairwise_wdl_targets.npy", npy_numeric(pairwise_wdl_targets, "<f4", "(" + std::to_string(positions) + "," + std::to_string(initial.player_count) + "," + std::to_string(initial.player_count) + ",3)")});
    members.push_back({"final_utility_targets.npy", npy_numeric(final_utility_targets, "<f4", "(" + std::to_string(positions) + "," + std::to_string(initial.player_count) + ")")});
    members.push_back({"final_vp_belief_targets.npy", npy_numeric(final_vp_belief_targets, "<f4", "(" + std::to_string(positions) + "," + std::to_string(initial.player_count) + ",403)")});
    members.push_back({"state_trace_json.npy", npy_strings(state_trace)}); members.push_back({"state_snapshot_json.npy", npy_strings(state_snapshots)}); members.push_back({"state_hashes.npy", npy_strings(state_hashes)}); members.push_back({"action_tuples_json.npy", npy_strings(action_tuples)}); members.push_back({"legal_action_tuples_json.npy", npy_strings(legal_json)}); members.push_back({"policy_visit_targets_by_tuple_json.npy", npy_strings(policy_json)}); members.push_back({"policy_type_targets_json.npy", npy_strings(policy_type)}); members.push_back({"policy_argument_targets_json.npy", npy_strings(policy_argument)}); members.push_back({"root_visit_counts_by_tuple_json.npy", npy_strings(root_visits)}); members.push_back({"root_policy_priors_by_tuple_json.npy", npy_strings(root_priors)});
    members.push_back({"action_type_ids.npy", npy_numeric(action_type_ids, "<i2", "(" + std::to_string(trace.size()) + ",)")}); members.push_back({"action_args.npy", npy_numeric(action_args, "<i2", "(" + std::to_string(trace.size()) + ",8)")}); members.push_back({"action_arg_mask.npy", npy_bool(action_arg_mask, "(" + std::to_string(trace.size()) + ",8)")}); members.push_back({"action_ids.npy", npy_numeric(action_ids, "<i8", "(" + std::to_string(trace.size()) + ",)")}); members.push_back({"position_index.npy", npy_numeric(position_index, "<i8", "(" + std::to_string(trace.size()) + ",)")}); members.push_back({"semantic_turn_index.npy", npy_numeric(semantic_index, "<i8", "(" + std::to_string(trace.size()) + ",)")}); members.push_back({"round.npy", npy_numeric(rounds, "<i2", "(" + std::to_string(trace.size()) + ",)")}); members.push_back({"player_to_move.npy", npy_numeric(players, "|i1", "(" + std::to_string(trace.size()) + ",)")}); members.push_back({"terminal_valid.npy", npy_bool(std::vector<std::uint8_t>{1}, "()")}); members.push_back({"metadata.npy", npy_strings({meta.str()})});
    write_zip(destination, std::move(members));
}

GameResult play_game(const Config& config, int game_index, Evaluator& evaluator) {
    const auto seed = config.seed + game_index;
    GaiaState initial = GaiaState::initial(config.players, seed);
    GaiaState state = initial;
    std::vector<TraceRow> trace;
    std::mt19937_64 game_rng(static_cast<std::uint64_t>(seed) ^ UINT64_C(0x9e3779b97f4a7c15));
    int move = 0;
    while (!state.is_terminal()) {
        if (move >= config.max_moves) throw std::runtime_error("self-play exceeded max_moves");
        Search search(evaluator, config, game_rng());
        const double temperature = move < config.temperature_moves ? config.temperature : 0.0;
        const auto result = search.run(state, temperature);
        const auto legal = state.legal_action_tuples();
        std::vector<int> visits(legal.size());
        for (std::size_t i = 0; i < legal.size(); ++i) {
            const int id = legacy_action_id(legal[i]);
            (void)id;
            visits[i] = result.visits[i];
        }
        const auto chosen = search.sample(result.policy);
        TraceRow row{state, legal, std::move(visits), result.policy, legacy_action_id(legal[chosen]), legal[chosen]};
        trace.push_back(std::move(row));
        state = state.apply(legal[chosen]); ++move;
    }
    trace.push_back(TraceRow{state, {}, {}, {}, -1, ActionTuple{}});
    fs::create_directories(config.output_dir);
    const auto timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    const auto destination = config.output_dir / ("game-" + std::to_string(timestamp) + "-" +
        std::to_string(process_id()) + "-" + std::to_string(game_index) + ".npz");
    write_game_npz(destination, initial, trace, evaluator.active_model_path());
    std::cout << "[cpp-selfplay] wrote " << destination.string() << " moves=" << move << " states=" << trace.size() << "\n" << std::flush;
    return GameResult{move, destination};
}

} // namespace

int main(int argc, char** argv) {
    std::optional<Config> active_config;
    int completed_games = 0;
    int completed_moves = 0;
    fs::path last_shard;
    try {
        active_config = parse_args(argc, argv);
        const Config& config = *active_config;
        Evaluator evaluator(config);
        int game = 0;
        write_status(config, "starting", completed_games, completed_moves);
        for (;;) {
            if (!config.once && fs::exists(config.stop_file)) break;
            for (int cycle = 0; cycle < config.games; ++cycle) {
                if (!config.once && fs::exists(config.stop_file)) break;
                evaluator.reload_if_changed();
                write_status(config, "running", completed_games, completed_moves);
                const auto result = play_game(config, game++, evaluator);
                ++completed_games;
                completed_moves += result.moves;
                last_shard = result.shard;
                write_status(config, "running", completed_games, completed_moves, last_shard);
            }
            if (config.once) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(config.poll_ms));
        }
        write_status(config, "stopped", completed_games, completed_moves, last_shard);
        return 0;
    } catch (const std::exception& error) {
        if (active_config)
            write_status(*active_config, "error", completed_games, completed_moves, last_shard, error.what());
        std::cerr << "[cpp-selfplay] error: " << error.what() << '\n';
        return 2;
    }
}
