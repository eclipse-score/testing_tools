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

# Resolves [project.dependencies] from pyproject.toml (the default src) against
# score_tooling's pins as constraints, so a score_tooling bump is picked up by
# re-running the update -- no version edit here.
# To update: bazel run //:requirements.update
compile_pip_requirements(
    name = "requirements",
    constraints = [
        "@score_tooling//python_basics:requirements.txt",
    ],
    # pip-compile builds this package's metadata to read [project.dependencies],
    # so the setuptools backend needs the `readme` file and the packages listed
    # in [tool.setuptools] to be present in the sandbox.
    data = ["README.md"] + glob(["testing_utils/**/*.py"]),
    extra_args = [
        "--no-annotate",
        # include [project.optional-dependencies].dev (ruff) so the lint and
        # test jobs install from this one lock
        "--extra=dev",
    ],
    requirements_txt = "requirements.txt.lock",
    tags = ["manual"],
)
