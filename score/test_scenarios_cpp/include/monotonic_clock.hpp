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
#pragma once

#include <chrono>
#include <string>

/// @brief Timestamp provider using monotonic clock.
class MonotonicClock {
   public:
    MonotonicClock();

    /// @brief Measure and write out the current time.
    /// @return Timestamp as string.
    std::string format_time() const;

   private:
    using ClockT = std::chrono::high_resolution_clock;
    using TimePointT = std::chrono::time_point<ClockT>;

    TimePointT start_;
};
