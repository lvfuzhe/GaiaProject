#pragma once

#include <string>
#include <string_view>

namespace gaiazero {

[[nodiscard]] std::string sha256_hex(std::string_view value);

}  // namespace gaiazero

