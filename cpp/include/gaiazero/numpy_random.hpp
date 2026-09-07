#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace gaiazero {

// Minimal, frozen implementation of NumPy 2.x default_rng (PCG64) behavior
// used by gaia_setup.py. It intentionally exposes only the operations needed
// by setup generation so the setup-seed-stream-v1 contract is portable to C++.
class NumpyRandom final {
public:
    explicit NumpyRandom(std::uint64_t seed);

    [[nodiscard]] std::uint64_t next_uint64();
    [[nodiscard]] std::uint32_t next_uint32();
    [[nodiscard]] std::uint64_t bounded_uint64(std::uint64_t inclusive_max);
    [[nodiscard]] std::int64_t integers(std::int64_t exclusive_high);
    [[nodiscard]] std::vector<int> permutation(int size);
    [[nodiscard]] std::vector<int> choice_without_replacement(int population,
                                                               int size);

private:
    struct UInt128 {
        std::uint64_t high{0};
        std::uint64_t low{0};
    };

    UInt128 state_{};
    UInt128 increment_{};
    bool has_uint32_{false};
    std::uint32_t buffered_uint32_{0};

    void step();
    [[nodiscard]] std::uint64_t random_interval(std::uint64_t inclusive_max);
};

}  // namespace gaiazero
