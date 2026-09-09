#pragma once

#include "gaiazero/gaia_state.hpp"
#include "gaiazero/inference.hpp"

#include <cstdint>
#include <vector>

namespace gaiazero {

// Fixed graph input layout consumed by the Python GNN and exported ONNX model.
// Player count is intentionally read from GaiaState: the 2/3/4-player models
// are independent and must not silently pad across player counts.
struct GraphEncoderConfig {
    std::int64_t max_nodes{128};
    std::int64_t max_edges{512};
    std::int64_t node_features{16};
    std::int64_t global_features{16};
    std::int64_t player_features{16};
    std::int64_t relation_types{16};

    [[nodiscard]] bool valid() const noexcept;
};

// Encodes one state as a batch of one. Padded entries remain zero and are
// excluded by the explicit masks. Ordering and normalization match
// src/gaiazero/gnn.py::graph_inputs_from_state.
[[nodiscard]] GraphBatch encode_graph_batch(
    const GaiaState& state,
    const GraphEncoderConfig& config = {});

// Encodes several states into one fixed-shape batch.  Every state must use
// the same graph/player capacities; edge indices remain local to each batch
// item (the ONNX/TensorRT contract is [B, E, 2], not a flattened COO graph).
[[nodiscard]] GraphBatch encode_graph_batch(
    const std::vector<const GaiaState*>& states,
    const GraphEncoderConfig& config = {});

}  // namespace gaiazero
