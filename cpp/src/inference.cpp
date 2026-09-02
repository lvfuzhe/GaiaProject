#include "gaiazero/inference.hpp"

#include <string>
#include <stdexcept>

namespace gaiazero {

bool GraphShape::valid() const noexcept {
    return batch > 0 && nodes > 0 && edges > 0 && players > 0 &&
           node_features > 0 && global_features > 0 && player_features > 0;
}

namespace {

void require_size(const char* name, std::size_t actual, std::size_t expected) {
    if (actual != expected) {
        throw std::invalid_argument(
            std::string(name) + " has size " + std::to_string(actual) +
            ", expected " + std::to_string(expected));
    }
}

}  // namespace

void validate_graph_batch(const GraphBatch& batch) {
    const auto& s = batch.shape;
    if (!s.valid()) {
        throw std::invalid_argument("GraphShape dimensions must all be positive");
    }
    const auto b = static_cast<std::size_t>(s.batch);
    const auto n = static_cast<std::size_t>(s.nodes);
    const auto e = static_cast<std::size_t>(s.edges);
    const auto p = static_cast<std::size_t>(s.players);
    const auto nf = static_cast<std::size_t>(s.node_features);
    const auto gf = static_cast<std::size_t>(s.global_features);
    const auto pf = static_cast<std::size_t>(s.player_features);

    require_size("node_features", batch.node_features.size(), b * n * nf);
    require_size("edge_index", batch.edge_index.size(), b * e * 2);
    require_size("edge_type", batch.edge_type.size(), b * e);
    require_size("edge_mask", batch.edge_mask.size(), b * e);
    require_size("node_mask", batch.node_mask.size(), b * n);
    require_size("global_features", batch.global_features.size(), b * gf);
    require_size("player_features", batch.player_features.size(), b * p * pf);
    require_size("player_mask", batch.player_mask.size(), b * p);
}

}  // namespace gaiazero
