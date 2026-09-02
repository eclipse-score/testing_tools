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

#include <optional>
#include <stdexcept>
#include <string>

#include "scenario.hpp"

// Replacement for `should_panic` macro.
#define SHOULD_THROW(expression, expected_exception, expected_what) \
    do {                                                            \
        try {                                                       \
            expression;                                             \
            FAIL();                                                 \
        } catch (const expected_exception& ex) {                    \
            ASSERT_STREQ(ex.what(), expected_what);                 \
        } catch (...) {                                             \
            FAIL();                                                 \
        }                                                           \
    } while (false)

// `SHOULD_THROW` for common `std::runtime_error`.
#define SHOULD_THROW_RE(expression, expected_what) \
    SHOULD_THROW(expression, std::runtime_error, expected_what)

class ScenarioStub : public Scenario {
   public:
    ScenarioStub(const std::string& name) : name_{name} {}
    ~ScenarioStub() {}

    std::string name() const override { return name_; }

    void run(const std::string& input) const override {
        if (input == "ok") {
            return;
        } else if (input == "error") {
            throw std::runtime_error{"Requested error"};
        } else {
            throw std::runtime_error{"Unknown value"};
        }
    }

   private:
    std::string name_;
};
