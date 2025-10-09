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
"""
Utilities for interacting with build systems.
"""

__all__ = ["BuildTools", "CargoTools", "BazelTools"]

import json
from abc import ABC, abstractmethod
from pathlib import Path
from subprocess import PIPE, Popen, TimeoutExpired
from typing import Any

import pytest

# region common


class BuildTools(ABC):
    """
    Base class for build system interactions.
    """

    def __init__(self, option_prefix: str = "", command_timeout: float = 10.0, build_timeout: float = 180.0) -> None:
        """
        Create tools instance.

        Parameters
        ----------
        option_prefix : str
            Prefix for options expected by 'select_target_path'.
            - '' will expect '--target-path' and '--target-name'.
            - 'cpp' will expect '--cpp-target-path' and '--cpp-target-name'.
        command_timeout : float
            Common command timeout in seconds.
        build_timeout : float
            Build command timeout in seconds.
        """
        if option_prefix:
            self._target_path_flag = f"--{option_prefix}-target-path"
            self._target_name_flag = f"--{option_prefix}-target-name"
        else:
            self._target_path_flag = "--target-path"
            self._target_name_flag = "--target-name"
        self._command_timeout = command_timeout
        self._build_timeout = build_timeout

    @property
    def command_timeout(self) -> float:
        """
        Common command timeout in seconds.
        """
        return self._command_timeout

    @command_timeout.setter
    def command_timeout(self, command_timeout: float) -> None:
        self._command_timeout = command_timeout

    @property
    def build_timeout(self) -> float:
        """
        Build command timeout in seconds.
        """
        return self._build_timeout

    @build_timeout.setter
    def build_timeout(self, build_timeout: float) -> None:
        self._build_timeout = build_timeout

    @abstractmethod
    def find_target_path(self, target_name: str, *, expect_exists: bool) -> Path:
        """
        Find path to executable.

        Parameters
        ----------
        target_name : str
            Target name.
        expect_exists : bool
            Check that executable exists.
        """

    def select_target_path(self, config: pytest.Config, *, expect_exists: bool) -> Path:
        """
        Select executable path based on "--target-name" option.
        - if "--target-path" is set - use it
        - if "--target-path" not set - search for an executable named "--target-name"
        - if "--target-name" not set - error

        Parameters
        ----------
        config : pytest.Config
            Pytest config object.
        expect_exists : bool
            Check that executable exists.
        """
        # Types are ignored due to 'default' being incorrectly set to 'notset' type.
        if option_target_path := config.getoption(self._target_path_flag, default=None):  # type: ignore
            # Check path is valid.
            if not isinstance(option_target_path, Path):
                raise pytest.UsageError(f"Invalid executable path type: {type(option_target_path)}")
            if expect_exists and not option_target_path.is_file():
                raise pytest.UsageError(f"Invalid executable path: {option_target_path}")

            return option_target_path

        if option_target_name := config.getoption(self._target_name_flag, default=None):  # type: ignore
            # Check name type is valid.
            if not isinstance(option_target_name, str):
                raise pytest.UsageError(f"Invalid executable name type: {type(option_target_name)}")
            # Find path, rethrow as 'UsageError' on errors.
            # Timeouts are rethrown as 'TimeoutExpired'.
            try:
                return self.find_target_path(option_target_name, expect_exists=expect_exists)
            except TimeoutExpired as e:
                raise e
            except Exception as e:
                raise pytest.UsageError from e

        raise pytest.UsageError(f'Either "{self._target_path_flag}" or "{self._target_name_flag}" must be set')

    @abstractmethod
    def build(self, target_name: str) -> Path:
        """
        Run build for selected target.
        Returns path to built executable.

        Parameters
        ----------
        target_name : str
            Name of the target to build.
        """


# endregion

# region cargo tools


class CargoTools(BuildTools):
    """
    Utilities for interacting with Cargo.
    """

    def __init__(self, option_prefix: str = "", command_timeout: float = 10.0, build_timeout: float = 180.0) -> None:
        """
        Create Cargo tools instance.

        Parameters
        ----------
        option_prefix : str
            Prefix for options expected by 'select_target_path'.
            - '' will expect '--target-path' and '--target-name'.
            - 'rust' will expect '--rust-target-path' and '--rust-target-name'.
        command_timeout : float
            Common command timeout in seconds.
        build_timeout : float
            "cargo build" timeout in seconds.
        """
        super().__init__(option_prefix, command_timeout, build_timeout)

    def metadata(self) -> dict[str, Any]:
        """
        Read Cargo metadata and return as dict.
        CWD must be inside Cargo project.
        """
        # Run command.
        command = ["cargo", "metadata", "--format-version", "1"]
        with Popen(command, stdout=PIPE, text=True) as p:
            stdout, _ = p.communicate(timeout=self.command_timeout)
            if p.returncode != 0:
                raise RuntimeError(f"Failed to read Cargo metadata, returncode: {p.returncode}")

        # Load stdout as JSON data.
        return json.loads(stdout)

    def find_target_path(self, target_name: str, *, expect_exists: bool = True) -> Path:
        """
        Find path to executable.
        Target directory is taken from Cargo metadata.
        "debug" configuration is used.

        Parameters
        ----------
        target_name : str
            Target name.
        expect_exists : bool
            Check that executable exists.
        """
        # Read metadata.
        metadata = self.metadata()

        # Read target directory.
        target_directory = Path(metadata["target_directory"]).resolve()

        # Check expected file exists.
        target_path = target_directory / "debug" / target_name
        if expect_exists and not target_path.exists():
            raise RuntimeError(f"Executable not found: {target_path}")

        return target_path

    def build(self, target_name: str) -> Path:
        """
        Run build for selected target.
        Manifest path is taken from Cargo metadata.
        "debug" configuration is built.

        Parameters
        ----------
        target_name : str
            Name of the target to build.
        """
        # Read metadata.
        metadata = self.metadata()

        # Read manifest path from metadata.
        pkg_entries = list(filter(lambda x: x["name"] == target_name, metadata["packages"]))
        if len(pkg_entries) < 1:
            raise RuntimeError(f"No data found for {target_name}")
        if len(pkg_entries) > 1:
            raise RuntimeError(f"Multiple data found for {target_name}")
        pkg_entry = pkg_entries[0]

        manifest_path = Path(pkg_entry["manifest_path"]).resolve()

        # Run build.
        command = ["cargo", "build", "--manifest-path", manifest_path]
        with Popen(command, text=True) as p:
            _, _ = p.communicate(timeout=self.build_timeout)
            if p.returncode != 0:
                raise RuntimeError(f"Failed to run build, returncode: {p.returncode}")

        return self.find_target_path(target_name, expect_exists=True)


# endregion

# region bazel tools


class BazelTools(BuildTools):
    """
    Utilities for interacting with Bazel.
    """

    def __init__(self, option_prefix: str = "", command_timeout: float = 10.0, build_timeout: float = 180.0) -> None:
        """
        Create Bazel tools instance.

        Parameters
        ----------
        option_prefix : str
            Prefix for options expected by 'select_target_path'.
            - '' will expect '--target-path' and '--target-name'.
            - 'cpp' will expect '--cpp-target-path' and '--cpp-target-name'.
        command_timeout : float
            Common command timeout in seconds.
        build_timeout : float
            "bazel build" timeout in seconds.
        """
        super().__init__(option_prefix, command_timeout, build_timeout)

    def query(self, query: str = "//...") -> list[str]:
        """
        Run query and return list of targets.
        CWD must be inside Bazel project.

        Parameters
        ----------
        query : str
            Query to run.
        """
        # Run command.
        command = ["bazel", "query", query]
        with Popen(command, stdout=PIPE, text=True) as p:
            stdout, _ = p.communicate(timeout=self.command_timeout)
            if p.returncode != 0:
                raise RuntimeError(f"Failed to query Bazel, returncode: {p.returncode}")

        # Load stdout as list of strings.
        return stdout.strip().split("\n")

    def find_target_path(self, target_name: str, *, expect_exists: bool = True) -> Path:
        """
        Find path to executable.
        Target directory is taken from Cargo metadata.
        "debug" configuration is used.

        Parameters
        ----------
        target_name : str
            Target name.
        expect_exists : bool
            Check that executable exists.
        """
        # Find workspace root.
        ws_root_cmd = ["bazel", "info", "workspace"]
        with Popen(ws_root_cmd, stdout=PIPE, text=True) as p:
            ws_str, _ = p.communicate(timeout=self.command_timeout)
            ws_str = ws_str.strip()
            if p.returncode != 0:
                raise RuntimeError(f"Failed to determine workspace root, returncode: {p.returncode}")
        ws_path = Path(ws_str)

        # Find executable path relative to workspace root path.
        command = [
            "bazel",
            "cquery",
            "--output=starlark",
            "--starlark:expr=target.files_to_run.executable.path",
            target_name,
        ]
        with Popen(command, stdout=PIPE, text=True) as p:
            target_str, _ = p.communicate(timeout=self.command_timeout)
            target_str = target_str.strip()
            if p.returncode != 0:
                raise RuntimeError(f"Failed to cquery Bazel, returncode: {p.returncode}")

        # Check expected file exists.
        target_path = ws_path / target_str
        if expect_exists and not target_path.exists():
            raise RuntimeError(f"Executable not found: {target_path}")

        return target_path

    def build(self, target_name: str) -> Path:
        """
        Run build for selected target.

        Parameters
        ----------
        target_name : str
            Name of the target to build.
        """
        # Run build.
        command = ["bazel", "build", target_name]
        with Popen(command, text=True) as p:
            _, _ = p.communicate(timeout=self.build_timeout)
            if p.returncode != 0:
                raise RuntimeError(f"Failed to run build, returncode: {p.returncode}")

        return self.find_target_path(target_name, expect_exists=True)


# endregion
