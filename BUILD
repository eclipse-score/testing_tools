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
    timeout = "moderate",  # metadata build exceeds the rule's default "short" under CI load
    srcs = [
        "pyproject.toml",
        "@score_tooling//python_basics:requirements.txt",
    ],
    # setuptools' build backend needs the package directories to exist to
    # build this package's own metadata when resolving [project.dependencies]
    # from pyproject.toml; the glob tracks [tool.setuptools]'s packages list
    # automatically instead of needing a hand-kept duplicate here.
    data = glob(["testing_utils/**/__init__.py"]),
    extra_args = [
        "--no-annotate",
        "--extra=dev",  # so ruff resolves from this lock too
    ],
    requirements_txt = "requirements.txt.lock",
    tags = ["manual"],
)
