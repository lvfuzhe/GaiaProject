#include "gaiazero/onnxruntime_backend.hpp"

#include <array>
#include <stdexcept>
#include <string>
#include <utility>

#if defined(GAIA_HAS_ONNXRUNTIME)
#include <onnxruntime_cxx_api.h>
#endif

namespace gaiazero {

struct OnnxRuntimeCpuBackend::Impl {
#if defined(GAIA_HAS_ONNXRUNTIME)
    Ort::Env environment{ORT_LOGGING_LEVEL_WARNING, "gaiazero"};
    Ort::Session session{nullptr};
    std::array<std::string, 8> input_names;
    std::array<std::string, 4> output_names;
    std::array<std::vector<std::int64_t>, 8> input_shapes;
    OnnxRuntimeCpuBackendInfo metadata;
#else
    std::filesystem::path model_path;
    OnnxRuntimeCpuBackendInfo metadata;
#endif
};

namespace {

#if defined(GAIA_HAS_ONNXRUNTIME)
constexpr std::array<std::string_view, 8> kInputNames{
    "node_features", "edge_index", "edge_type", "edge_mask",
    "node_mask", "global_features", "player_features", "player_mask"};
constexpr std::array<std::string_view, 4> kOutputNames{
    kActionTypeLogitsName, kActionArgumentLogitsName, kPairwiseWdlLogitsName,
    kVpBeliefLogitsName};

std::string allocated_name_to_string(const Ort::AllocatedStringPtr& value) {
    if (!value) {
        throw std::runtime_error("ONNX Runtime returned a null tensor name");
    }
    return std::string(value.get());
}

std::size_t tensor_element_count(const Ort::Value& tensor) {
    const auto shape = tensor.GetTensorTypeAndShapeInfo().GetShape();
    std::size_t count = 1;
    for (const auto dimension : shape) {
        if (dimension < 0) {
            throw std::runtime_error("ONNX output has an unresolved dynamic dimension");
        }
        count *= static_cast<std::size_t>(dimension);
    }
    return count;
}

template <typename T>
Ort::Value make_tensor(
    Ort::MemoryInfo& memory,
    const std::vector<T>& values,
    const std::vector<std::int64_t>& shape) {
    return Ort::Value::CreateTensor<T>(
        memory, const_cast<T*>(values.data()), values.size(), shape.data(), shape.size());
}

void validate_tensor_signature(
    const Ort::TypeInfo& type_info,
    ONNXTensorElementDataType expected_type,
    std::size_t expected_rank,
    const char* tensor_name) {
    const auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    if (tensor_info.GetElementType() != expected_type) {
        throw std::runtime_error(
            std::string("unexpected ONNX element type for ") + tensor_name);
    }
    const auto shape = tensor_info.GetShape();
    if (shape.size() != expected_rank) {
        throw std::runtime_error(
            std::string("unexpected ONNX rank for ") + tensor_name);
    }
}

void validate_runtime_shape(
    const std::vector<std::int64_t>& expected,
    const std::vector<std::int64_t>& actual,
    const char* tensor_name) {
    if (expected.size() != actual.size()) {
        throw std::invalid_argument(
            std::string("runtime rank does not match ONNX model for ") + tensor_name);
    }
    for (std::size_t index = 0; index < expected.size(); ++index) {
        // ONNX uses -1 for a dynamic dimension (the exported graph only has
        // a dynamic batch axis).
        if (expected[index] >= 0 && expected[index] != actual[index]) {
            throw std::invalid_argument(
                std::string("runtime shape does not match ONNX model for ") +
                tensor_name);
        }
    }
}
#endif

}  // namespace

OnnxRuntimeCpuBackend::OnnxRuntimeCpuBackend(
    const std::filesystem::path& model_path,
    OnnxRuntimeCpuConfig config)
    : impl_(std::make_unique<Impl>()) {
#if defined(GAIA_HAS_ONNXRUNTIME)
    if (model_path.empty() || !std::filesystem::exists(model_path)) {
        throw std::invalid_argument("ONNX model path does not exist: " + model_path.string());
    }
    if (config.intra_op_threads < 1 || config.inter_op_threads < 1) {
        throw std::invalid_argument("ONNX Runtime thread counts must be positive");
    }
    impl_->metadata.model_path = model_path.string();
    impl_->metadata.runtime_version = Ort::GetVersionString();
    impl_->metadata.intra_op_threads = config.intra_op_threads;
    impl_->metadata.inter_op_threads = config.inter_op_threads;

    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(config.intra_op_threads);
    options.SetInterOpNumThreads(config.inter_op_threads);
    options.SetGraphOptimizationLevel(
        config.enable_graph_optimizations ? ORT_ENABLE_ALL : ORT_ENABLE_BASIC);
#ifdef _WIN32
    const auto wide_path = model_path.wstring();
    impl_->session = Ort::Session(impl_->environment, wide_path.c_str(), options);
#else
    impl_->session = Ort::Session(impl_->environment, model_path.string().c_str(), options);
#endif

    Ort::AllocatorWithDefaultOptions allocator;
    if (impl_->session.GetInputCount() != kInputNames.size()) {
        throw std::runtime_error("ONNX model input count does not match graph contract");
    }
    if (impl_->session.GetOutputCount() != kOutputNames.size()) {
        throw std::runtime_error("ONNX model output count does not match graph contract");
    }
    for (std::size_t index = 0; index < kInputNames.size(); ++index) {
        impl_->input_names[index] = allocated_name_to_string(
            impl_->session.GetInputNameAllocated(index, allocator));
        if (impl_->input_names[index] != kInputNames[index]) {
            throw std::runtime_error(
                "unexpected ONNX input name at index " + std::to_string(index) +
                ": " + impl_->input_names[index]);
        }
        static constexpr std::array<ONNXTensorElementDataType, 8> kInputTypes{
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT};
        static constexpr std::array<std::size_t, 8> kInputRanks{3, 3, 2, 2, 2, 2, 3, 2};
        validate_tensor_signature(
            impl_->session.GetInputTypeInfo(index), kInputTypes[index],
            kInputRanks[index], impl_->input_names[index].c_str());
        impl_->input_shapes[index] = impl_->session.GetInputTypeInfo(index)
                                         .GetTensorTypeAndShapeInfo()
                                         .GetShape();
    }
    for (std::size_t index = 0; index < kOutputNames.size(); ++index) {
        impl_->output_names[index] = allocated_name_to_string(
            impl_->session.GetOutputNameAllocated(index, allocator));
        if (impl_->output_names[index] != kOutputNames[index]) {
            throw std::runtime_error(
                "unexpected ONNX output name at index " + std::to_string(index) +
                ": " + impl_->output_names[index]);
        }
        static constexpr std::array<std::size_t, 4> kOutputRanks{2, 3, 3, 3};
        validate_tensor_signature(
            impl_->session.GetOutputTypeInfo(index),
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, kOutputRanks[index],
            impl_->output_names[index].c_str());
    }
#else
    (void)config;
    impl_->model_path = model_path;
    impl_->metadata.model_path = model_path.string();
    throw std::runtime_error(
        "ONNX Runtime CPU backend is disabled; configure with "
        "-DGAIA_ENABLE_ORT_CPU=ON and provide ONNXRUNTIME_ROOT");
#endif
}

OnnxRuntimeCpuBackend::~OnnxRuntimeCpuBackend() = default;
OnnxRuntimeCpuBackend::OnnxRuntimeCpuBackend(OnnxRuntimeCpuBackend&&) noexcept = default;
OnnxRuntimeCpuBackend& OnnxRuntimeCpuBackend::operator=(OnnxRuntimeCpuBackend&&) noexcept = default;

std::string_view OnnxRuntimeCpuBackend::name() const noexcept {
    return "onnxruntime-cpu";
}

OnnxRuntimeCpuBackendInfo OnnxRuntimeCpuBackend::info() const {
    return impl_->metadata;
}

NetworkOutput OnnxRuntimeCpuBackend::infer(const GraphBatch& batch) {
    validate_graph_batch(batch);
#if defined(GAIA_HAS_ONNXRUNTIME)
    const auto& s = batch.shape;

    const std::vector<std::int64_t> node_shape{
        s.batch, s.nodes, s.node_features};
    const std::vector<std::int64_t> edge_index_shape{s.batch, s.edges, 2};
    const std::vector<std::int64_t> edge_shape{s.batch, s.edges};
    const std::vector<std::int64_t> global_shape{s.batch, s.global_features};
    const std::vector<std::int64_t> player_shape{
        s.batch, s.players, s.player_features};
    const std::vector<std::int64_t> player_mask_shape{s.batch, s.players};
    const std::array<std::vector<std::int64_t>, 8> runtime_shapes{
        node_shape, edge_index_shape, edge_shape, edge_shape,
        std::vector<std::int64_t>{s.batch, s.nodes}, global_shape,
        player_shape, player_mask_shape};
    static constexpr std::array<const char*, 8> input_names_for_errors{
        "node_features", "edge_index", "edge_type", "edge_mask",
        "node_mask", "global_features", "player_features", "player_mask"};
    for (std::size_t index = 0; index < runtime_shapes.size(); ++index) {
        validate_runtime_shape(
            impl_->input_shapes[index], runtime_shapes[index],
            input_names_for_errors[index]);
    }
    Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);
    std::array<Ort::Value, 8> inputs{
        make_tensor(memory, batch.node_features, node_shape),
        make_tensor(memory, batch.edge_index, edge_index_shape),
        make_tensor(memory, batch.edge_type, edge_shape),
        make_tensor(memory, batch.edge_mask, edge_shape),
        make_tensor(memory, batch.node_mask, std::vector<std::int64_t>{s.batch, s.nodes}),
        make_tensor(memory, batch.global_features, global_shape),
        make_tensor(memory, batch.player_features, player_shape),
        make_tensor(memory, batch.player_mask, player_mask_shape)};
    std::array<const char*, 8> input_names{};
    std::array<const char*, 4> output_names{};
    for (std::size_t index = 0; index < input_names.size(); ++index) {
        input_names[index] = impl_->input_names[index].c_str();
    }
    for (std::size_t index = 0; index < output_names.size(); ++index) {
        output_names[index] = impl_->output_names[index].c_str();
    }
    auto outputs = impl_->session.Run(
        Ort::RunOptions{nullptr}, input_names.data(), inputs.data(), inputs.size(),
        output_names.data(), output_names.size());
    if (outputs.size() != 4) {
        throw std::runtime_error("ONNX Runtime returned an unexpected output count");
    }

    NetworkOutput result;
    auto copy_output = [](const Ort::Value& value, std::vector<float>& destination) {
        if (!value.IsTensor() || value.GetTensorTypeAndShapeInfo().GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            throw std::runtime_error("ONNX output must be a float tensor");
        }
        const auto count = tensor_element_count(value);
        const auto* data = value.GetTensorData<float>();
        destination.assign(data, data + count);
    };
    copy_output(outputs[0], result.action_type_logits);
    copy_output(outputs[1], result.action_argument_logits);
    copy_output(outputs[2], result.pairwise_wdl_logits);
    copy_output(outputs[3], result.vp_belief_logits);
    return result;
#else
    throw std::runtime_error(
        "ONNX Runtime CPU backend is disabled; rebuild with GAIA_ENABLE_ORT_CPU=ON");
#endif
}

}  // namespace gaiazero
