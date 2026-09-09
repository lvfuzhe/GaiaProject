#pragma once

#include "gaiazero/inference.hpp"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

namespace gaiazero {

struct TensorRtConfig {
    int device_id{0};
    int optimization_profile{0};
    bool enable_fp16{false}; // The engine's precision is authoritative; this is metadata only.
};

struct TensorRtBackendInfo {
    std::string engine_path;
    int device_id{0};
    int optimization_profile{0};
    std::int64_t max_batch_size{0};
};

// TensorRT backend for the fixed graph contract.  The implementation uses a
// single enqueue for the whole GraphBatch, so MCTS leaf waves amortize kernel
// launch and host/device transfer overhead.  Builds without the optional SDK
// retain this class at the source level and fail with an actionable message at
// construction time.
class TensorRtBackend final : public InferenceBackend {
public:
    explicit TensorRtBackend(const std::filesystem::path& engine_path,
                             TensorRtConfig config = {});
    ~TensorRtBackend() override;

    TensorRtBackend(const TensorRtBackend&) = delete;
    TensorRtBackend& operator=(const TensorRtBackend&) = delete;
    TensorRtBackend(TensorRtBackend&&) noexcept;
    TensorRtBackend& operator=(TensorRtBackend&&) noexcept;

    [[nodiscard]] std::string_view name() const noexcept override;
    [[nodiscard]] NetworkOutput infer(const GraphBatch& batch) override;
    [[nodiscard]] TensorRtBackendInfo info() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace gaiazero
