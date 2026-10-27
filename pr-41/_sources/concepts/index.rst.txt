..
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

Concepts
========

Overview
--------

The C++ and Rust test scenario libraries (``test_scenarios_cpp``, ``test_scenarios_rust``) let
a module expose its integration and system test scenarios as one CLI binary. An external runner,
such as pytest or ITF, calls that binary.

``testing_utils`` is a Python package with Bazel and Cargo build-tool wrappers, a log container
for structured log queries, and result-entry types for structured test reports.

Test Scenarios
---------------

- ``Scenario``: a named unit of work. ``name()`` returns the name and ``run(input)`` executes it.
- ``ScenarioGroup`` / ``ScenarioGroupImpl``: groups ``Scenario``\ s and nested
  ``ScenarioGroup``\ s under a name. Nesting produces dotted names, for example ``list.enumerate``.
- ``TestContext``: wraps a root ``ScenarioGroup``. It runs a scenario by its dotted name or lists
  all scenarios recursively.
- ``run_cli_app``: a CLI frontend over a ``TestContext`` with the options ``--list-scenarios``,
  ``--name``, ``--input`` and ``--help``.

Testing Utils
-------------

``testing_utils`` contains:

- Build tool helpers: Bazel and Cargo metadata queries, target path lookup
- ``LogContainer`` and ``ResultEntry``: storage and queries for test log entries
- ``Scenario`` / ``ScenarioGroup`` / ``run_cli_app``: Python versions of the C++ and Rust types
  above

See the top-level `README <https://github.com/eclipse-score/testing_tools#readme>`_ for
installation and detailed API usage.
