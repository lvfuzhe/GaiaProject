#include "gaiazero/gaia_state.hpp"
#include "gaiazero/graph_encoder.hpp"
#include "gaiazero/sha256.hpp"

#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void append_u64_le(std::string& output, std::uint64_t value) {
    for (int byte = 0; byte < 8; ++byte) {
        output.push_back(static_cast<char>((value >> (byte * 8)) & 0xffU));
    }
}

void append_shape(std::string& output,
                  std::initializer_list<std::int64_t> shape) {
    for (const auto value : shape) {
        append_u64_le(output, static_cast<std::uint64_t>(value));
    }
}

template <typename T>
void append_values(std::string& output, const std::vector<T>& values) {
    if (values.empty()) return;
    const auto* bytes = reinterpret_cast<const char*>(values.data());
    output.append(bytes, values.size() * sizeof(T));
}

void append_float_array(std::string& output,
                        const std::vector<float>& values,
                        std::initializer_list<std::int64_t> shape) {
    output.append("float32");
    append_shape(output, shape);
    append_values(output, values);
}

void append_int64_array(std::string& output,
                        const std::vector<std::int64_t>& values,
                        std::initializer_list<std::int64_t> shape) {
    output.append("int64");
    append_shape(output, shape);
    append_values(output, values);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 3) {
            throw std::invalid_argument(
                "usage: gaiazero_graph_probe <players> <seed>");
        }
        const auto players = std::stoi(argv[1]);
        const auto seed = std::stoll(argv[2]);
        const auto state = gaiazero::GaiaState::initial(players, seed);
        const auto batch = gaiazero::encode_graph_batch(state);
        const auto& shape = batch.shape;
        std::string payload;
        payload.reserve((batch.node_features.size() * sizeof(float)) +
                        (batch.edge_index.size() * sizeof(std::int64_t)) +
                        256);
        append_float_array(payload, batch.node_features,
                           {shape.batch, shape.nodes, shape.node_features});
        append_int64_array(payload, batch.edge_index,
                           {shape.batch, shape.edges, 2});
        append_int64_array(payload, batch.edge_type,
                           {shape.batch, shape.edges});
        append_float_array(payload, batch.edge_mask,
                           {shape.batch, shape.edges});
        append_float_array(payload, batch.node_mask,
                           {shape.batch, shape.nodes});
        append_float_array(payload, batch.global_features,
                           {shape.batch, shape.global_features});
        append_float_array(payload, batch.player_features,
                           {shape.batch, shape.players, shape.player_features});
        append_float_array(payload, batch.player_mask,
                           {shape.batch, shape.players});
        std::cout << "digest=" << gaiazero::sha256_hex(payload) << '\n';
        std::cout << "shape=" << shape.batch << ',' << shape.nodes << ','
                  << shape.edges << ',' << shape.players << ','
                  << shape.node_features << ',' << shape.global_features << ','
                  << shape.player_features << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
