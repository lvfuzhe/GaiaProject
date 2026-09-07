#include "gaiazero/numpy_random.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <limits>
#include <stdexcept>

#if defined(_MSC_VER) && defined(_M_X64)
#include <intrin.h>
#endif

namespace gaiazero {
namespace {

constexpr std::uint32_t kInitA = 0x43b0d7e5U;
constexpr std::uint32_t kMultA = 0x931e8875U;
constexpr std::uint32_t kInitB = 0x8b51f9ddU;
constexpr std::uint32_t kMultB = 0x58f38dedU;
constexpr std::uint32_t kMixMultLeft = 0xca01f9ddU;
constexpr std::uint32_t kMixMultRight = 0x4973f715U;
constexpr std::uint64_t kPcgMultiplierHigh = 2549297995355413924ULL;
constexpr std::uint64_t kPcgMultiplierLow = 4865540595714422341ULL;

std::uint64_t multiply_high(std::uint64_t left, std::uint64_t right) {
#if defined(_MSC_VER) && defined(_M_X64)
    return __umulh(left, right);
#elif defined(__SIZEOF_INT128__)
    return static_cast<std::uint64_t>(
        (static_cast<unsigned __int128>(left) * right) >> 64U);
#else
    const std::uint64_t left_low = static_cast<std::uint32_t>(left);
    const std::uint64_t left_high = left >> 32U;
    const std::uint64_t right_low = static_cast<std::uint32_t>(right);
    const std::uint64_t right_high = right >> 32U;
    const std::uint64_t word_zero = left_low * right_low;
    const std::uint64_t middle = left_high * right_low + (word_zero >> 32U);
    std::uint64_t word_one = middle & 0xffffffffULL;
    const std::uint64_t word_two = middle >> 32U;
    word_one += left_low * right_high;
    return left_high * right_high + word_two + (word_one >> 32U);
#endif
}

std::array<std::uint64_t, 4> seed_sequence(std::uint64_t seed) {
    std::vector<std::uint32_t> entropy;
    entropy.push_back(static_cast<std::uint32_t>(seed));
    const auto high = static_cast<std::uint32_t>(seed >> 32U);
    if (high != 0) entropy.push_back(high);

    std::array<std::uint32_t, 4> pool{};
    std::uint32_t hash_constant = kInitA;
    auto hashmix = [&hash_constant](std::uint32_t value) {
        value ^= hash_constant;
        hash_constant *= kMultA;
        value *= hash_constant;
        value ^= value >> 16U;
        return value;
    };
    auto mix = [](std::uint32_t left, std::uint32_t right) {
        std::uint32_t result = kMixMultLeft * left - kMixMultRight * right;
        result ^= result >> 16U;
        return result;
    };
    for (std::size_t index = 0; index < pool.size(); ++index) {
        pool[index] = hashmix(index < entropy.size() ? entropy[index] : 0U);
    }
    for (std::size_t source = 0; source < pool.size(); ++source) {
        for (std::size_t destination = 0; destination < pool.size(); ++destination) {
            if (source != destination) {
                pool[destination] = mix(pool[destination], hashmix(pool[source]));
            }
        }
    }
    for (std::size_t source = pool.size(); source < entropy.size(); ++source) {
        for (auto& destination : pool) {
            destination = mix(destination, hashmix(entropy[source]));
        }
    }

    std::array<std::uint32_t, 8> words{};
    hash_constant = kInitB;
    for (std::size_t index = 0; index < words.size(); ++index) {
        std::uint32_t value = pool[index % pool.size()];
        value ^= hash_constant;
        hash_constant *= kMultB;
        value *= hash_constant;
        value ^= value >> 16U;
        words[index] = value;
    }
    std::array<std::uint64_t, 4> result{};
    for (std::size_t index = 0; index < result.size(); ++index) {
        result[index] = static_cast<std::uint64_t>(words[index * 2]) |
                        (static_cast<std::uint64_t>(words[index * 2 + 1]) << 32U);
    }
    return result;
}

}  // namespace

NumpyRandom::NumpyRandom(std::uint64_t seed) {
    const auto generated = seed_sequence(seed);
    const UInt128 initial_state{generated[0], generated[1]};
    const UInt128 initial_sequence{generated[2], generated[3]};
    increment_.high = (initial_sequence.high << 1U) |
                      (initial_sequence.low >> 63U);
    increment_.low = (initial_sequence.low << 1U) | 1U;
    step();
    const auto previous_low = state_.low;
    state_.low += initial_state.low;
    state_.high += initial_state.high + (state_.low < previous_low ? 1ULL : 0ULL);
    step();
}

void NumpyRandom::step() {
    const std::uint64_t low = state_.low * kPcgMultiplierLow;
    std::uint64_t high = multiply_high(state_.low, kPcgMultiplierLow);
    high += state_.high * kPcgMultiplierLow;
    high += state_.low * kPcgMultiplierHigh;
    const auto before_add = low;
    state_.low = low + increment_.low;
    state_.high = high + increment_.high + (state_.low < before_add ? 1ULL : 0ULL);
}

std::uint64_t NumpyRandom::next_uint64() {
    step();
    return std::rotr(state_.high ^ state_.low,
                     static_cast<int>(state_.high >> 58U));
}

std::uint32_t NumpyRandom::next_uint32() {
    if (has_uint32_) {
        has_uint32_ = false;
        return buffered_uint32_;
    }
    const auto value = next_uint64();
    has_uint32_ = true;
    buffered_uint32_ = static_cast<std::uint32_t>(value >> 32U);
    return static_cast<std::uint32_t>(value);
}

std::uint64_t NumpyRandom::bounded_uint64(std::uint64_t inclusive_max) {
    if (inclusive_max == 0) return 0;
    if (inclusive_max <= std::numeric_limits<std::uint32_t>::max()) {
        if (inclusive_max == std::numeric_limits<std::uint32_t>::max()) {
            return next_uint32();
        }
        const auto exclusive = static_cast<std::uint32_t>(inclusive_max + 1U);
        std::uint64_t product = static_cast<std::uint64_t>(next_uint32()) * exclusive;
        auto leftover = static_cast<std::uint32_t>(product);
        if (leftover < exclusive) {
            const auto threshold = static_cast<std::uint32_t>(
                (std::numeric_limits<std::uint32_t>::max() - inclusive_max) %
                exclusive);
            while (leftover < threshold) {
                product = static_cast<std::uint64_t>(next_uint32()) * exclusive;
                leftover = static_cast<std::uint32_t>(product);
            }
        }
        return product >> 32U;
    }
    if (inclusive_max == std::numeric_limits<std::uint64_t>::max()) {
        return next_uint64();
    }
    const auto exclusive = inclusive_max + 1U;
    auto value = next_uint64();
    auto leftover = value * exclusive;
    if (leftover < exclusive) {
        const auto threshold =
            (std::numeric_limits<std::uint64_t>::max() - inclusive_max) % exclusive;
        while (leftover < threshold) {
            value = next_uint64();
            leftover = value * exclusive;
        }
    }
    return multiply_high(value, exclusive);
}

std::int64_t NumpyRandom::integers(std::int64_t exclusive_high) {
    if (exclusive_high <= 0) {
        throw std::invalid_argument("integers exclusive_high must be positive");
    }
    return static_cast<std::int64_t>(
        bounded_uint64(static_cast<std::uint64_t>(exclusive_high - 1)));
}

std::uint64_t NumpyRandom::random_interval(std::uint64_t inclusive_max) {
    if (inclusive_max == 0) return 0;
    auto mask = inclusive_max;
    mask |= mask >> 1U;
    mask |= mask >> 2U;
    mask |= mask >> 4U;
    mask |= mask >> 8U;
    mask |= mask >> 16U;
    mask |= mask >> 32U;
    std::uint64_t value = 0;
    if (inclusive_max <= std::numeric_limits<std::uint32_t>::max()) {
        do {
            value = next_uint32() & static_cast<std::uint32_t>(mask);
        } while (value > inclusive_max);
    } else {
        do {
            value = next_uint64() & mask;
        } while (value > inclusive_max);
    }
    return value;
}

std::vector<int> NumpyRandom::permutation(int size) {
    if (size < 0) throw std::invalid_argument("permutation size must be non-negative");
    std::vector<int> values(static_cast<std::size_t>(size));
    for (int index = 0; index < size; ++index) values[static_cast<std::size_t>(index)] = index;
    for (int index = size - 1; index >= 1; --index) {
        const auto selected = static_cast<int>(random_interval(static_cast<std::uint64_t>(index)));
        std::swap(values[static_cast<std::size_t>(index)],
                  values[static_cast<std::size_t>(selected)]);
    }
    return values;
}

std::vector<int> NumpyRandom::choice_without_replacement(int population,
                                                         int size) {
    if (population < 0 || size < 0 || size > population) {
        throw std::invalid_argument("invalid choice population or size");
    }
    std::vector<int> result(static_cast<std::size_t>(size));
    std::vector<std::uint64_t> hash_set;
    std::uint64_t set_size = static_cast<std::uint64_t>(1.2 * size);
    auto mask = set_size;
    mask |= mask >> 1U;
    mask |= mask >> 2U;
    mask |= mask >> 4U;
    mask |= mask >> 8U;
    mask |= mask >> 16U;
    mask |= mask >> 32U;
    set_size = mask + 1U;
    hash_set.assign(static_cast<std::size_t>(set_size),
                    std::numeric_limits<std::uint64_t>::max());
    for (int current = population - size; current < population; ++current) {
        auto value = bounded_uint64(static_cast<std::uint64_t>(current));
        auto location = value & mask;
        while (hash_set[static_cast<std::size_t>(location)] !=
                   std::numeric_limits<std::uint64_t>::max() &&
               hash_set[static_cast<std::size_t>(location)] != value) {
            location = (location + 1U) & mask;
        }
        const auto output_index = static_cast<std::size_t>(current - population + size);
        if (hash_set[static_cast<std::size_t>(location)] ==
            std::numeric_limits<std::uint64_t>::max()) {
            hash_set[static_cast<std::size_t>(location)] = value;
            result[output_index] = static_cast<int>(value);
        } else {
            location = static_cast<std::uint64_t>(current) & mask;
            while (hash_set[static_cast<std::size_t>(location)] !=
                   std::numeric_limits<std::uint64_t>::max()) {
                location = (location + 1U) & mask;
            }
            hash_set[static_cast<std::size_t>(location)] =
                static_cast<std::uint64_t>(current);
            result[output_index] = current;
        }
    }
    for (int index = size - 1; index >= 1; --index) {
        const auto selected = static_cast<int>(
            bounded_uint64(static_cast<std::uint64_t>(index)));
        std::swap(result[static_cast<std::size_t>(index)],
                  result[static_cast<std::size_t>(selected)]);
    }
    return result;
}

}  // namespace gaiazero
