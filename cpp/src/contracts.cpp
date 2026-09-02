#include "gaiazero/contracts.hpp"

#include <algorithm>
#include <stdexcept>

namespace gaiazero {

ActionTuple ActionTuple::create(
    ActionType type,
    std::initializer_list<std::int32_t> values) {
    if (!is_valid_action_type(type)) {
        throw std::invalid_argument("unknown ActionType");
    }
    if (values.size() > kMaxActionArguments) {
        throw std::invalid_argument("ActionTuple has more than 8 arguments");
    }
    ActionTuple result;
    result.action_type = type;
    result.argument_count = static_cast<std::uint8_t>(values.size());
    result.arguments.fill(-1);
    std::copy(values.begin(), values.end(), result.arguments.begin());
    return result;
}

bool ActionTuple::valid() const noexcept {
    return is_valid_action_type(action_type) &&
           argument_count <= kMaxActionArguments;
}

bool ActionTuple::operator==(const ActionTuple& other) const noexcept {
    if (action_type != other.action_type || argument_count != other.argument_count) {
        return false;
    }
    for (std::size_t index = 0; index < argument_count; ++index) {
        if (arguments[index] != other.arguments[index]) {
            return false;
        }
    }
    return true;
}

}  // namespace gaiazero
