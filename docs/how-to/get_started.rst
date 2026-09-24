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

Get Started
============

Add the library as a Bazel dependency:

.. code-block:: python

   # C++
   deps = ["@score_test_scenarios//test_scenarios_cpp"]

   # Rust
   deps = ["@score_test_scenarios//test_scenarios_rust"]

Define scenarios, group them, and run them through ``TestContext``/``run_cli_app``. For a
complete, runnable walkthrough — including JSON input parsing and structured tracing with a
monotonic clock — see the ``basic`` example:

- C++: ``score/test_scenarios_cpp/examples/basic.cpp``, run with
  ``bazel run //examples:cpp_basic``
- Rust: ``score/test_scenarios_rust/examples/basic.rs``, run with
  ``bazel run //examples:rust_basic``

Build everything:

.. code-block:: bash

   bazel build //...

Run the tests:

.. code-block:: bash

   bazel test //...

Run the examples:

.. code-block:: bash

   bazel run //examples:cpp_basic -- --list-scenarios
   bazel run //examples:rust_basic -- --list-scenarios
