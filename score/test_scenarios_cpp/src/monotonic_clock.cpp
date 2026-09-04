// *******************************************************************************
// Copyright (c) 2025 Contributors to the Eclipse Foundation
//
// See the NOTICE file(s) distributed with this work for additional
// information regarding copyright ownership.
//
// This program and the accompanying materials are made available under the
// terms of the Apache License Version 2.0 which is available at
// https://www.apache.org/licenses/LICENSE-2.0
//
// SPDX-License-Identifier: Apache-2.0
// *******************************************************************************
#include "monotonic_clock.hpp"

#include <sstream>

MonotonicClock::MonotonicClock() : start_{ClockT::now()} {}

std::string MonotonicClock::format_time() const {
    auto elapsed{ClockT::now() - start_};
    auto elapsed_us{std::chrono::duration_cast<std::chrono::microseconds>(elapsed)};
    return std::to_string(elapsed_us.count());
}
