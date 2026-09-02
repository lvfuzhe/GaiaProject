#pragma once

#include "gaiazero/inference.hpp"

#include <filesystem>
#include <memory>
#include <string>

namespace gaiazero {

struct OnnxRuntimeCpuConfig {
    int intra_op_threads{1};
    int inter_op_threads{1};
    bool enable_graph_optimizations{true};
};

struct OnnxRuntimeCpuBackendInfo {
    std::string model_path;
    std::string runtime_version;
    int intra_op_threads{1};
    int inter_op_threads{1};
};

// ONNX Runtime CPU reference backend. The class is always available at the
// source level; builds without the optional SDK produce a clear runtime error
// when this backend is constructed. This keeps the CPU contract testable on
// machines that do not have the native ONNX Runtime distribution installed.
class OnnxRuntimeCpuBackend final : public InferenceBackend {
public:
    explicit OnnxRuntimeCpuBackend(
        const std::filesystem::path& model_path,
        OnnxRuntimeCpuConfig config = {});
    ~OnnxRuntimeCpuBackend() override;

    OnnxRuntimeCpuBackend(const OnnxRuntimeCpuBackend&) = delete;
    OnnxRuntimeCpuBackend& operator=(const OnnxRuntimeCpuBackend&) = delete;
    OnnxRuntimeCpuBackend(OnnxRuntimeCpuBackend&&) noexcept;
    OnnxRuntimeCpuBackend& operator=(OnnxRuntimeCpuBackend&&) noexcept;

    [[nodiscard]] std::string_view name() const noexcept override;
    [[nodiscard]] NetworkOutput infer(const GraphBatch& batch) override;
    [[nodiscard]] OnnxRuntimeCpuBackendInfo info() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace gaiazero
