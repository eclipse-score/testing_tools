# *******************************************************************************
# Copyright (c) 2025 Contributors to the Eclipse Foundation
#
# See the NOTICE file(s) distributed with this work for additional
# information regarding copyright ownership.
#
# This program and the accompanying materials are made available under the
# terms of the Apache License Version 2.0 which is available at
# https://www.apache.org/licenses/LICENSE-2.0
#
# SPDX-License-Identifier: Apache-2.0
# *******************************************************************************
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

cc_library(
    name = "test_scenarios_cpp",
    # 'glob' not used due to unhelpful nature of C++ errors.
    srcs = [
        "src/cli.cpp",
        "src/monotonic_clock.cpp",
        "src/scenario.cpp",
        "src/string_utils.cpp",
        "src/string_utils.hpp",
        "src/test_context.cpp",
        "src/tracing.cpp",
    ],
    # Only public headers should be defined here.
    hdrs = [
        "include/cli.hpp",
        "include/monotonic_clock.hpp",
        "include/scenario.hpp",
        "include/test_context.hpp",
        "include/tracing.hpp",
    ],
    includes = ["include/"],
    visibility = ["//visibility:public"],
    deps = ["@nlohmann_json//:json"],
)

cc_test(
    name = "tests",
    srcs = [
        "tests/common.hpp",
        "tests/test_cli.cpp",
        "tests/test_scenario.cpp",
        "tests/test_test_context.cpp",
    ],
    visibility = ["//visibility:private"],
    deps = [
        ":test_scenarios_cpp",
        "@googletest//:gtest_main",
    ],
)
