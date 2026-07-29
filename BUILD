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
load("@rules_python//python:pip.bzl", "compile_pip_requirements")
load("@score_tooling//:defs.bzl", "setup_starpls")

setup_starpls(
    name = "starpls_server",
    visibility = ["//visibility:public"],
)

# To update: bazel run //:requirements.update
compile_pip_requirements(
    name = "requirements",
    srcs = [
        "requirements.txt",
    ],
    constraints = [
        "@score_tooling//python_basics:requirements.txt",
    ],
    extra_args = [
        "--no-annotate",
    ],
    requirements_txt = "requirements.txt.lock",
    tags = ["manual"],
)
