#include "gaiazero/contracts.hpp"
#include "gaiazero/inference.hpp"
#include "gaiazero/onnxruntime_backend.hpp"

#include <cassert>
#include <iostream>
#include <stdexcept>

int main() {
    using namespace gaiazero;

    static_assert(kActionTypeCount == 54);
    assert(kRulesVersion == "standard-v22");
    assert(action_type_id(ActionType::build_mine) == 0);
    assert(action_type_id(ActionType::legacy_action) == 53);
    assert(action_type_name(ActionType::research) == "research");
    assert(action_type_name(static_cast<ActionType>(99)).empty());

    const auto mine = ActionTuple::create(ActionType::build_mine, {3, 17});
    const auto same_mine = ActionTuple::create(ActionType::build_mine, {3, 17});
    const auto other_mine = ActionTuple::create(ActionType::build_mine, {3, 18});
    assert(mine.valid());
    assert(mine.arguments[2] == -1);
    assert(mine == same_mine);
    assert(mine != other_mine);

    bool rejected_action = false;
    try {
        (void)ActionTuple::create(ActionType::legacy_action,
                                  {0, 1, 2, 3, 4, 5, 6, 7, 8});
    } catch (const std::invalid_argument&) {
        rejected_action = true;
    }
    assert(rejected_action);

    GraphBatch batch;
    batch.shape = GraphShape{2, 8, 12, 3, 4, 5, 6};
    batch.node_features.resize(2 * 8 * 4);
    batch.edge_index.resize(2 * 12 * 2);
    batch.edge_type.resize(2 * 12);
    batch.edge_mask.resize(2 * 12);
    batch.node_mask.resize(2 * 8);
    batch.global_features.resize(2 * 5);
    batch.player_features.resize(2 * 3 * 6);
    batch.player_mask.resize(2 * 3);
    validate_graph_batch(batch);

    batch.player_mask.pop_back();
    bool rejected = false;
    try {
        validate_graph_batch(batch);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    assert(rejected);

    bool backend_unavailable = false;
    try {
        OnnxRuntimeCpuBackend backend(std::filesystem::path{});
        assert(backend.name() == "onnxruntime-cpu");
    } catch (const std::exception&) {
        backend_unavailable = true;
    }
    assert(backend_unavailable);

    std::cout << "gaiazero_cpp_smoke ok\n";
    return 0;
}
