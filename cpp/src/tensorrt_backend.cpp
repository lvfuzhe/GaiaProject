#include "gaiazero/tensorrt_backend.hpp"

#include <array>
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(GAIA_HAS_TENSORRT)
#include <NvInfer.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#endif

namespace gaiazero {

struct TensorRtBackend::Impl {
    TensorRtBackendInfo metadata;
#if defined(GAIA_HAS_TENSORRT)
    struct Logger final : nvinfer1::ILogger {
        void log(Severity severity, const char* message) noexcept override {
            if (severity <= Severity::kWARNING) {
                // TensorRT diagnostics are intentionally sent to stderr; the
                // worker status file remains machine-readable.
                std::fprintf(stderr, "[TensorRT] %s\n", message ? message : "");
            }
        }
    } logger;
    std::unique_ptr<nvinfer1::IRuntime, void(*)(nvinfer1::IRuntime*)> runtime{
        nullptr, [](nvinfer1::IRuntime* value) { if (value) value->destroy(); }};
    std::unique_ptr<nvinfer1::ICudaEngine, void(*)(nvinfer1::ICudaEngine*)> engine{
        nullptr, [](nvinfer1::ICudaEngine* value) { if (value) value->destroy(); }};
    std::unique_ptr<nvinfer1::IExecutionContext, void(*)(nvinfer1::IExecutionContext*)> context{
        nullptr, [](nvinfer1::IExecutionContext* value) { if (value) value->destroy(); }};
#endif
};

namespace {

#if defined(GAIA_HAS_TENSORRT)
constexpr std::array<const char*, 8> kInputs{
    "node_features", "edge_index", "edge_type", "edge_mask",
    "node_mask", "global_features", "player_features", "player_mask"};
constexpr std::array<const char*, 4> kOutputs{
    "action_type_logits", "action_argument_logits",
    "pairwise_wdl_logits", "vp_belief_logits"};

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

std::size_t volume(const nvinfer1::Dims& dims) {
    std::size_t result = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] < 0) throw std::runtime_error("TensorRT output has unresolved dynamic dimension");
        result *= static_cast<std::size_t>(dims.d[i]);
    }
    return result;
}

std::size_t element_size(nvinfer1::DataType type) {
    switch (type) {
    case nvinfer1::DataType::kFLOAT: return sizeof(float);
    case nvinfer1::DataType::kHALF: return sizeof(__half);
    case nvinfer1::DataType::kINT32: return sizeof(std::int32_t);
    case nvinfer1::DataType::kINT64: return sizeof(std::int64_t);
    default: throw std::runtime_error("unsupported TensorRT binding data type");
    }
}

std::vector<__half> to_half(const std::vector<float>& values) {
    std::vector<__half> converted(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) converted[i] = __float2half(values[i]);
    return converted;
}

std::vector<std::int32_t> to_int32(const std::vector<std::int64_t>& values) {
    std::vector<std::int32_t> converted;
    converted.reserve(values.size());
    for (const auto value : values) {
        if (value < std::numeric_limits<std::int32_t>::min() ||
            value > std::numeric_limits<std::int32_t>::max())
            throw std::invalid_argument("TensorRT INT32 input cannot represent graph index/type");
        converted.push_back(static_cast<std::int32_t>(value));
    }
    return converted;
}

#endif

} // namespace

TensorRtBackend::TensorRtBackend(const std::filesystem::path& engine_path,
                                 TensorRtConfig config)
    : impl_(std::make_unique<Impl>()) {
    if (engine_path.empty() || !std::filesystem::is_regular_file(engine_path))
        throw std::invalid_argument("TensorRT engine path does not exist: " + engine_path.string());
    if (config.device_id < 0 || config.optimization_profile < 0)
        throw std::invalid_argument("TensorRT device/profile must be non-negative");
    impl_->metadata.engine_path = engine_path.string();
    impl_->metadata.device_id = config.device_id;
    impl_->metadata.optimization_profile = config.optimization_profile;
#if defined(GAIA_HAS_TENSORRT)
    check_cuda(cudaSetDevice(config.device_id), "cudaSetDevice");
    std::ifstream input(engine_path, std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("cannot open TensorRT engine: " + engine_path.string());
    const auto size = input.tellg();
    if (size <= 0) throw std::runtime_error("TensorRT engine is empty");
    std::vector<char> serialized(static_cast<std::size_t>(size));
    input.seekg(0);
    input.read(serialized.data(), size);
    if (!input) throw std::runtime_error("cannot read TensorRT engine");
    impl_->runtime.reset(nvinfer1::createInferRuntime(impl_->logger));
    if (!impl_->runtime) throw std::runtime_error("TensorRT createInferRuntime failed");
    impl_->engine.reset(impl_->runtime->deserializeCudaEngine(serialized.data(), serialized.size()));
    if (!impl_->engine) throw std::runtime_error("TensorRT deserializeCudaEngine failed");
    impl_->context.reset(impl_->engine->createExecutionContext());
    if (!impl_->context) throw std::runtime_error("TensorRT createExecutionContext failed");
    if (config.optimization_profile > 0 && !impl_->context->setOptimizationProfile(config.optimization_profile))
        throw std::runtime_error("TensorRT optimization profile is not available");
    if (impl_->engine->getNbBindings() != static_cast<int>(kInputs.size() + kOutputs.size()))
        throw std::runtime_error("TensorRT binding count does not match graph contract");
    for (int binding = 0; binding < impl_->engine->getNbBindings(); ++binding) {
        const char* name = impl_->engine->getBindingName(binding);
        if (!name) throw std::runtime_error("TensorRT binding has no name");
        bool known = false;
        for (const auto* expected : kInputs) if (std::string(name) == expected) known = true;
        for (const auto* expected : kOutputs) if (std::string(name) == expected) known = true;
        if (!known) throw std::runtime_error(std::string("unexpected TensorRT binding: ") + name);
    }
    // The runtime does not need to query a profile's max shape here: each
    // infer() call sets concrete batch dimensions and TensorRT validates them
    // against the selected optimization profile.
    impl_->metadata.max_batch_size = 0;
#else
    (void)config;
    throw std::runtime_error(
        "TensorRT backend is disabled; configure with -DGAIA_ENABLE_TENSORRT=ON "
        "and provide TENSORRT_ROOT (CUDA/TensorRT SDK)");
#endif
}

TensorRtBackend::~TensorRtBackend() = default;
TensorRtBackend::TensorRtBackend(TensorRtBackend&&) noexcept = default;
TensorRtBackend& TensorRtBackend::operator=(TensorRtBackend&&) noexcept = default;

std::string_view TensorRtBackend::name() const noexcept { return "tensorrt"; }
TensorRtBackendInfo TensorRtBackend::info() const { return impl_->metadata; }

NetworkOutput TensorRtBackend::infer(const GraphBatch& batch) {
    validate_graph_batch(batch);
#if defined(GAIA_HAS_TENSORRT)
    const auto& s = batch.shape;
    std::array<std::vector<std::int64_t>, 8> shapes{
        std::vector<std::int64_t>{s.batch, s.nodes, s.node_features},
        std::vector<std::int64_t>{s.batch, s.edges, 2},
        std::vector<std::int64_t>{s.batch, s.edges},
        std::vector<std::int64_t>{s.batch, s.edges},
        std::vector<std::int64_t>{s.batch, s.nodes},
        std::vector<std::int64_t>{s.batch, s.global_features},
        std::vector<std::int64_t>{s.batch, s.players, s.player_features},
        std::vector<std::int64_t>{s.batch, s.players}};
    std::array<int, 12> binding_index{};
    for (std::size_t i = 0; i < kInputs.size(); ++i) {
        const int binding = impl_->engine->getBindingIndex(kInputs[i]);
        if (binding < 0) throw std::runtime_error(std::string("TensorRT input is missing: ") + kInputs[i]);
        binding_index[i] = binding;
        nvinfer1::Dims dims;
        dims.nbDims = static_cast<int>(shapes[i].size());
        for (int d = 0; d < dims.nbDims; ++d) dims.d[d] = shapes[i][static_cast<std::size_t>(d)];
        if (!impl_->context->setBindingDimensions(binding, dims))
            throw std::runtime_error(std::string("TensorRT rejected input shape: ") + kInputs[i]);
    }
    if (!impl_->context->allInputDimensionsSpecified())
        throw std::runtime_error("TensorRT input dimensions are incomplete");
    std::array<std::vector<float>, 4> host_outputs;
    std::array<void*, 12> device{};
    struct DeviceCleanup {
        std::array<void*, 12>& pointers;
        ~DeviceCleanup() { for (void* pointer : pointers) if (pointer) cudaFree(pointer); }
    } cleanup{device};
    (void)cleanup;
    auto upload = [&](std::size_t input_index, const void* source, std::size_t bytes) {
        const auto binding = binding_index[input_index];
        check_cuda(cudaMalloc(&device[static_cast<std::size_t>(binding)], bytes), "cudaMalloc input");
        check_cuda(cudaMemcpy(device[static_cast<std::size_t>(binding)], source, bytes, cudaMemcpyHostToDevice), "cudaMemcpy input");
    };
    std::array<std::vector<__half>, 8> half_inputs;
    std::array<std::vector<std::int32_t>, 8> int32_inputs;
    const std::array<const std::vector<float>*, 8> float_inputs{
        &batch.node_features, nullptr, nullptr, &batch.edge_mask,
        &batch.node_mask, &batch.global_features, &batch.player_features,
        &batch.player_mask};
    const std::array<const std::vector<std::int64_t>*, 8> integer_inputs{
        nullptr, &batch.edge_index, &batch.edge_type, nullptr,
        nullptr, nullptr, nullptr, nullptr};
    for (std::size_t input_index = 0; input_index < kInputs.size(); ++input_index) {
        const auto binding = binding_index[input_index];
        const auto type = impl_->engine->getBindingDataType(binding);
        if (const auto* values = float_inputs[input_index]) {
            if (type == nvinfer1::DataType::kFLOAT) {
                upload(input_index, values->data(), values->size() * sizeof(float));
            } else if (type == nvinfer1::DataType::kHALF) {
                half_inputs[input_index] = to_half(*values);
                upload(input_index, half_inputs[input_index].data(),
                       half_inputs[input_index].size() * sizeof(__half));
            } else {
                throw std::runtime_error(std::string("unsupported TensorRT floating input type: ") + kInputs[input_index]);
            }
        } else if (const auto* values = integer_inputs[input_index]) {
            if (type == nvinfer1::DataType::kINT64) {
                upload(input_index, values->data(), values->size() * sizeof(std::int64_t));
            } else if (type == nvinfer1::DataType::kINT32) {
                int32_inputs[input_index] = to_int32(*values);
                upload(input_index, int32_inputs[input_index].data(),
                       int32_inputs[input_index].size() * sizeof(std::int32_t));
            } else {
                throw std::runtime_error(std::string("unsupported TensorRT integer input type: ") + kInputs[input_index]);
            }
        }
    }
    for (std::size_t output_index = 0; output_index < kOutputs.size(); ++output_index) {
        const int binding = impl_->engine->getBindingIndex(kOutputs[output_index]);
        if (binding < 0) throw std::runtime_error(std::string("TensorRT output is missing: ") + kOutputs[output_index]);
        binding_index[8 + output_index] = binding;
        const auto dims = impl_->context->getBindingDimensions(binding);
        if (dims.nbDims < 1 || dims.d[0] != s.batch)
            throw std::runtime_error(std::string("TensorRT output batch dimension mismatch: ") + kOutputs[output_index]);
        const auto elements = volume(dims);
        const auto type = impl_->engine->getBindingDataType(binding);
        if (type != nvinfer1::DataType::kFLOAT && type != nvinfer1::DataType::kHALF)
            throw std::runtime_error(std::string("TensorRT output must be FP32/FP16: ") + kOutputs[output_index]);
        host_outputs[output_index].resize(elements);
        check_cuda(cudaMalloc(&device[static_cast<std::size_t>(binding)], elements * element_size(type)), "cudaMalloc output");
    }
    if (!impl_->context->enqueueV2(device.data(), nullptr, nullptr))
        throw std::runtime_error("TensorRT enqueueV2 failed");
    for (std::size_t output_index = 0; output_index < kOutputs.size(); ++output_index) {
        const auto binding = binding_index[8 + output_index];
        const auto type = impl_->engine->getBindingDataType(binding);
        if (type == nvinfer1::DataType::kFLOAT) {
            check_cuda(cudaMemcpy(host_outputs[output_index].data(), device[static_cast<std::size_t>(binding)],
                                  host_outputs[output_index].size() * sizeof(float), cudaMemcpyDeviceToHost),
                      "cudaMemcpy output");
        } else {
            std::vector<__half> half_output(host_outputs[output_index].size());
            check_cuda(cudaMemcpy(half_output.data(), device[static_cast<std::size_t>(binding)],
                                  half_output.size() * sizeof(__half), cudaMemcpyDeviceToHost),
                      "cudaMemcpy output");
            for (std::size_t index = 0; index < half_output.size(); ++index)
                host_outputs[output_index][index] = __half2float(half_output[index]);
        }
    }
    return NetworkOutput{std::move(host_outputs[0]), std::move(host_outputs[1]),
                         std::move(host_outputs[2]), std::move(host_outputs[3])};
#else
    (void)batch;
    throw std::runtime_error("TensorRT backend is disabled; rebuild with GAIA_ENABLE_TENSORRT=ON");
#endif
}

} // namespace gaiazero
