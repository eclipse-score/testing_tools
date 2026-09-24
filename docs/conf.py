# *******************************************************************************
# Copyright (c) 2026 Contributors to the Eclipse Foundation
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

project = "Test Scenarios"
project_url = "https://github.com/eclipse-score/testing_tools"
project_prefix = "TS_"
author = "S-CORE"
version = "0.0.1"

extensions = [
    "sphinx_design",
    "sphinx_needs",
    "myst_parser",
    "sphinxcontrib.plantuml",
    "score_plantuml",
    "score_metamodel",
    "score_metrics",
    "sphinx_mounts",
    "score_mounts",
    "score_draw_uml_funcs",
    "score_source_code_linker",
    "score_layout",
]

myst_enable_extensions = ["colon_fence"]

exclude_patterns = [
    # Only needed for unsandboxed builds (bazel run //:docs, esbonio); sandboxed
    # 'bazel build //:docs' never sees these paths.
    "bazel-*",
    ".venv_docs",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ["templates"]

numfig = True
