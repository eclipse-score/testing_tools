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

# In order to update the requirements, change the `requirements.txt` file and run:
# `bazel run //:requirements.update`.
# This will update the `requirements.txt.lock` file.
# `@score_tooling//python_basics:requirements.txt` is merged in as a
# constraints source so pytest resolves to the same exact version pinned by
# score_tooling, instead of this repo's own floor (`pyproject.toml`) floating
# independently. Note: bumping score_tooling's version above may rename this
# exported file (it moved to per-Python-version files, e.g.
# `requirements_3_12.txt`, by score_tooling 1.3.x) — this src must be updated
# to match in the same change, or `requirements.update` fails to analyze.
compile_pip_requirements(
    name = "requirements",
    srcs = [
        "requirements.txt",
        "@score_tooling//python_basics:requirements.txt",
    ],
    requirements_txt = "requirements.txt.lock",
    tags = ["manual"],
)
