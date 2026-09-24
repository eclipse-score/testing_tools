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

``testing_tools`` provides two things:

- **Test scenario libraries** (``test_scenarios_cpp``, ``test_scenarios_rust``) — a small
  framework for exposing a module's own integration/system test scenarios as a single CLI
  binary, driven externally (e.g. by pytest or ITF).
- **testing_utils** — a Python package of test-automation helpers: Bazel/Cargo build-tool
  wrappers, a log container for structured log queries, and result-entry types for structured
  test reports.

Test Scenarios
---------------

- ``Scenario`` — a single named unit of work: ``name()`` identifies it, ``run(input)`` executes it.
- ``ScenarioGroup`` / ``ScenarioGroupImpl`` — groups ``Scenario``\\ s (and nested
  ``ScenarioGroup``\\ s) under a name. Nesting produces dotted names, e.g. ``list.enumerate``.
- ``TestContext`` — wraps a root ``ScenarioGroup``; runs a scenario by its dotted name, or lists
  all of them recursively.
- ``run_cli_app`` — a ready-made CLI frontend over a ``TestContext``, providing
  ``--list-scenarios``, ``--name``, ``--input``, and ``--help`` for free.

Testing Utils
-------------

A Python package (``testing_utils``) providing:

- Build tool helpers — Bazel/Cargo metadata queries, target path lookup
- ``LogContainer`` and ``ResultEntry`` — structured storage and querying of test log entries
- ``Scenario`` / ``ScenarioGroup`` / ``run_cli_app`` — the Python-side equivalents of the C++/Rust
  concepts above

See the top-level `README <https://github.com/eclipse-score/testing_tools#readme>`_ for
installation and detailed API usage.
