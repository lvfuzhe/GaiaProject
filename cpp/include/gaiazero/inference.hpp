#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

namespace gaiazero {

inline constexpr std::string_view kActionTypeLogitsName = "action_type_logits";
inline constexpr std::string_view kActionArgumentLogitsName = "action_argument_logits";
inline constexpr std::string_view kPairwiseWdlLogitsName = "pairwise_wdl_logits";
inline constexpr std::string_view kVpBeliefLogitsName = "vp_belief_logits";

struct GraphShape {
    std::int64_t batch{0};
    std::int64_t nodes{0};
    std::int64_t edges{0};
    std::int64_t players{0};
    std::int64_t node_features{0};
    std::int64_t global_features{0};
    std::int64_t player_features{0};

    [[nodiscard]] bool valid() const noexcept;
};

// Fixed padded graph tensors match the Python GNN/ONNX boundary. A future
// rules encoder fills these vectors; backends never inspect Python objects.
struct GraphBatch {
    GraphShape shape;
    std::vector<float> node_features;
    std::vector<std::int64_t> edge_index;
    std::vector<std::int64_t> edge_type;
    std::vector<float> edge_mask;
    std::vector<float> node_mask;
    std::vector<float> global_features;
    std::vector<float> player_features;
    std::vector<float> player_mask;
};

struct NetworkOutput {
    std::vector<float> action_type_logits;
    std::vector<float> action_argument_logits;
    std::vector<float> pairwise_wdl_logits;
    std::vector<float> vp_belief_logits;
};

void validate_graph_batch(const GraphBatch& batch);

class InferenceBackend {
public:
    virtual ~InferenceBackend() = default;
    [[nodiscard]] virtual std::string_view name() const noexcept = 0;
    [[nodiscard]] virtual NetworkOutput infer(const GraphBatch& batch) = 0;
};

}  // namespace gaiazero

