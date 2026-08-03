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
#
# Resolved the same way every downstream S-CORE module resolves it (see
# score_persistency //tests/test_cases:requirements): this module's own
# dependencies and score_tooling's pins are peer `srcs`, so score_tooling's
# pytest pin is what lands in the lock. `pyproject.toml` is the src rather than
# a separate requirements.in so the dependency list has a single source of
# truth -- the same metadata downstream consumers resolve against.
compile_pip_requirements(
    name = "requirements",
    timeout = "moderate",  # that metadata build exceeds the rule's default "short" (60s)
    srcs = [
        "pyproject.toml",
        "@score_tooling//python_basics:requirements.txt",
    ],
    # rules_python pins pip-tools 7.4.1, which has no static-metadata path: it
    # always runs setuptools' PEP 517 `get_requires_for_build_wheel`. That reads
    # `[tool.setuptools] packages` and aborts with
    #   error: package directory 'testing_utils' does not exist
    # unless those directories exist in the runfiles tree. Only the package
    # markers are needed, not the sources -- listing them explicitly instead of
    # globbing `testing_utils/**/*.py` keeps `requirements_test` from re-running a
    # network resolve on every unrelated source edit. Verified: the resolved lock
    # is identical either way.
    data = [
        "testing_utils/__init__.py",
        "testing_utils/net/__init__.py",
    ],
    extra_args = [
        # A `srcs` entry from another module resolves to an absolute path, and
        # pip-compile writes it into the `# via -r ...` annotations -- score_persistency's
        # committed lock carries one developer's `/home/<user>/.cache/bazel/...` path for
        # exactly this reason. That makes the lock machine-specific, so `requirements_test`
        # below could never pass anywhere else. Dropping annotations keeps it reproducible.
        "--no-annotate",
        "--extra=dev",  # so ruff resolves from this lock too
    ],
    requirements_txt = "requirements.txt.lock",
    tags = ["manual"],
)
