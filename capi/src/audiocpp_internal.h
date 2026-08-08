#pragma once

// Internal helpers shared between the C ABI translation unit and its unit
// tests. NOT part of the public ABI (capi/include/audiocpp.h); do not export.

#include <string>

namespace audiocpp::detail {

// Render a cJSON/JSON number into the string form stored in the options map.
//
// Option consumers later split into int (stoll) and float (stof) parsers.
// If an integral JSON number like 2 were rendered as "2.000000", stoll would
// reject it ("must be an integer") — the original bug this guards against.
//
// Integers within double's exact-integer range (< 2^53) render as plain integer
// strings. Values at or beyond 2^53 cannot be represented exactly by a double,
// so they are emitted via the float serializer (scientific notation); stoll
// then rejects them rather than silently rounding a value the caller never sent.
// This matches the CLI's app/cli/request.cpp json_option_string behavior.
std::string option_number_to_string(double value);

}  // namespace audiocpp::detail
