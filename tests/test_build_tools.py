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
Tests for "build_tools" module.
"""

import os
from abc import ABC, abstractmethod
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from subprocess import Popen, TimeoutExpired
from textwrap import dedent
from typing import Any

import pytest

from testing_utils import BazelTools, BuildTools, CargoTools


@pytest.fixture(scope="class")
def class_tmp_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("temp")


@contextmanager
def cwd(new_cwd: Path | str) -> Generator[None, None, None]:
    """
    Temporarily change working directory.

    Parameters
    ----------
    new_cwd : Path
        Directory to set as CWD.
    """
    prev_cwd = Path.cwd()
    os.chdir(new_cwd)
    yield
    os.chdir(prev_cwd)


class Notset:
    def __repr__(self):
        return "<NOTSET>"


notset = Notset()


class MockConfig:
    """
    "Config" object mock.
    """

    def __init__(self, options: dict[str, Any]) -> None:
        self._options = options

    def getoption(self, name: str, default: Any = notset) -> Any:
        value = self._options.get(name, default)
        if value is notset:
            raise ValueError
        return value


class TestBuildTools(ABC):
    """
    Base class containing common test cases for all build systems.
    """

    @pytest.fixture(scope="class")
    @abstractmethod
    def tools_type(self) -> type[BuildTools]:
        """
        Build tools to test.
        """

    @pytest.fixture(scope="class")
    @abstractmethod
    def tmp_project(self, class_tmp_path: Path) -> tuple[str, Path]:
        """
        Create temporary binary project.
        Returns target name and path to project directory.

        Parameters
        ----------
        class_tmp_path : Path
            Temporary directory path.
        """

    @pytest.fixture(scope="class")
    def built_tmp_project(self, tools_type: type[BuildTools], tmp_project: tuple[str, Path]) -> tuple[str, Path]:
        """
        Create and build temporary binary project.
        Returns target name and path to project directory.

        Parameters
        ----------
        tmp_project : Path
            Path to temporary project directory.
        """
        target_name, path = tmp_project
        with cwd(path):
            tools = tools_type()
            _ = tools.build(target_name)
            return tmp_project

    @pytest.fixture(scope="class")
    @abstractmethod
    def expected_target_path(self, tmp_project: tuple[str, Path]) -> Path:
        """
        Expected target path.

        Parameters
        ----------
        project_path : Path
            Path to temporary project directory.
        target_name : str
            Target name.
        """

    class TestBuild:
        def test_build_ok(self, tools_type: type[BuildTools], tmp_project: tuple[str, Path]) -> None:
            target_name, path = tmp_project
            with cwd(path):
                tools = tools_type()
                target_path = tools.build(target_name)

                # Check executable exists.
                assert target_path.exists()

        def test_metadata_timeout(self, tools_type: type[BuildTools], tmp_project: tuple[str, Path]) -> None:
            target_name, path = tmp_project
            with cwd(path):
                command_timeout = 0.00000001
                tools = tools_type(command_timeout=command_timeout)
                with pytest.raises(TimeoutExpired):
                    _ = tools.build(target_name)

        def test_build_timeout(self, tools_type: type[BuildTools], tmp_project: tuple[str, Path]) -> None:
            target_name, path = tmp_project
            with cwd(path):
                build_timeout = 0.00000001
                tools = tools_type(build_timeout=build_timeout)
                with pytest.raises(TimeoutExpired):
                    _ = tools.build(target_name)

        def test_invalid_target_name(self, tools_type: type[BuildTools], tmp_project: tuple[str, Path]) -> None:
            _, path = tmp_project
            with cwd(path):
                invalid_target_name = "xyz"
                tools = tools_type()
                with pytest.raises(RuntimeError):
                    _ = tools.build(invalid_target_name)

        def test_invalid_cwd(self, tools_type: type[BuildTools], tmp_project: tuple[str, Path]) -> None:
            target_name, _ = tmp_project
            invalid_project_path = "/tmp"
            with cwd(invalid_project_path):
                tools = tools_type()
                with pytest.raises(RuntimeError):
                    _ = tools.build(target_name)

    class TestFindBinPath:
        def test_ok(
            self, tools_type: type[BuildTools], built_tmp_project: tuple[str, Path], expected_target_path: Path
        ) -> None:
            target_name, path = built_tmp_project
            with cwd(path):
                tools = tools_type()
                act_target_path = tools.find_target_path(target_name, expect_exists=True)

                # Check returned path is as expected.
                assert act_target_path == expected_target_path

        def test_timeout(self, tools_type: type[BuildTools], built_tmp_project: tuple[str, Path]) -> None:
            target_name, path = built_tmp_project
            with cwd(path):
                command_timeout = 0.00000001
                tools = tools_type(command_timeout=command_timeout)
                with pytest.raises(TimeoutExpired):
                    _ = tools.find_target_path(target_name, expect_exists=True)

        def test_invalid_target_name(self, tools_type: type[BuildTools], built_tmp_project: tuple[str, Path]) -> None:
            _, path = built_tmp_project
            with cwd(path):
                invalid_target_name = "invalid_target_name"
                tools = tools_type()
                with pytest.raises(RuntimeError):
                    _ = tools.find_target_path(invalid_target_name, expect_exists=True)

        def test_invalid_cwd(self, tools_type: type[BuildTools], built_tmp_project: tuple[str, Path]) -> None:
            target_name, _ = built_tmp_project
            invalid_project_path = "/tmp"
            with cwd(invalid_project_path):
                tools = tools_type()
                with pytest.raises(RuntimeError):
                    _ = tools.find_target_path(target_name, expect_exists=True)

        def test_not_expect_exists(
            self, tools_type: type[BuildTools], tmp_project: tuple[str, Path], expected_target_path: Path
        ) -> None:
            target_name, path = tmp_project
            with cwd(path):
                tools = tools_type()
                act_target_path = tools.find_target_path(target_name, expect_exists=False)

                # Check returned path is as expected.
                assert act_target_path == expected_target_path


class TestCargoTools(TestBuildTools):
    """
    Test cases for Cargo tools.
    """

    @pytest.fixture(scope="class")
    def tools_type(self) -> type[BuildTools]:
        return CargoTools

    @pytest.fixture(scope="class")
    def tmp_project(self, class_tmp_path: Path) -> tuple[str, Path]:
        # Target name and path to project.
        target_name = "project"
        project_path = class_tmp_path / target_name

        # Create binary project.
        command = ["cargo", "new", "--bin", project_path]
        with Popen(command, text=True) as p:
            _, _ = p.communicate(timeout=30.0)
            if p.returncode != 0:
                raise RuntimeError("Failed to create temporary binary project")

        return (target_name, project_path)

    @pytest.fixture(scope="class")
    def expected_target_path(self, tmp_project: tuple[str, Path]) -> Path:
        target_name, project_path = tmp_project
        return project_path / "target" / "debug" / target_name

    class TestMetadata:
        def test_ok(self, tmp_project: tuple[str, Path]) -> None:
            _, path = tmp_project
            with cwd(path):
                tools = CargoTools()
                metadata = tools.metadata()

                # Check on "target_directory" if valid.
                act_target_dir = Path(metadata["target_directory"])
                exp_target_dir = path / "target"
                assert act_target_dir == exp_target_dir

        def test_timeout(self, tmp_project: tuple[str, Path]) -> None:
            _, path = tmp_project
            with cwd(path):
                timeout = 0.00000001
                tools = CargoTools(command_timeout=timeout)
                with pytest.raises(TimeoutExpired):
                    _ = tools.metadata()

    class TestSelectBinPath:
        def test_target_path_set_ok(self, built_tmp_project: tuple[str, Path]) -> None:
            target_name, path = built_tmp_project
            with cwd(path):
                tools = CargoTools()
                # Find executable path.
                exp_target_path = tools.find_target_path(target_name, expect_exists=True)

                # Create mock.
                cfg = MockConfig({"--target-path": exp_target_path})

                # Run.
                act_target_path = tools.select_target_path(cfg, expect_exists=True)  # type: ignore

                # Check returned path is as expected.
                assert act_target_path == exp_target_path

        def test_target_path_set_invalid_type(self, built_tmp_project: tuple[str, Path]) -> None:
            target_name, path = built_tmp_project
            with cwd(path):
                tools = CargoTools()
                # Find executable path.
                exp_target_path = tools.find_target_path(target_name, expect_exists=True)

                # Create mock.
                cfg = MockConfig({"--target-path": str(exp_target_path)})

                # Run.
                with pytest.raises(pytest.UsageError):
                    _ = tools.select_target_path(cfg, expect_exists=True)  # type: ignore

        def test_target_path_set_invalid_value(self, built_tmp_project: tuple[str, Path]) -> None:
            _, path = built_tmp_project
            with cwd(path):
                tools = CargoTools()
                # Create mock.
                invalid_target_path = Path("/invalid/path")
                cfg = MockConfig({"--target-path": invalid_target_path})

                # Run.
                with pytest.raises(pytest.UsageError):
                    _ = tools.select_target_path(cfg, expect_exists=True)  # type: ignore

        def test_target_path_not_expect_exists(self, tmp_project: tuple[str, Path]) -> None:
            target_name, path = tmp_project
            with cwd(path):
                tools = CargoTools()
                # Find executable path.
                exp_target_path = tools.find_target_path(target_name, expect_exists=False)

                # Create mock.
                cfg = MockConfig({"--target-path": exp_target_path})

                # Run.
                act_target_path = tools.select_target_path(cfg, expect_exists=False)  # type: ignore

                # Check returned path is as expected.
                assert act_target_path == exp_target_path

        def test_target_name_set_ok(self, built_tmp_project: tuple[str, Path]) -> None:
            target_name, path = built_tmp_project
            with cwd(path):
                tools = CargoTools()
                # Find executable path.
                exp_target_path = tools.find_target_path(target_name, expect_exists=True)

                # Create mock.
                cfg = MockConfig({"--target-name": target_name})

                # Run.
                act_target_path = tools.select_target_path(cfg, expect_exists=True)  # type: ignore

                # Check returned path is as expected.
                assert act_target_path == exp_target_path

        def test_target_name_set_invalid_type(self, built_tmp_project: tuple[str, Path]) -> None:
            target_name, path = built_tmp_project
            with cwd(path):
                tools = CargoTools()
                # Create mock.
                cfg = MockConfig({"--target-name": Path(target_name)})

                # Run.
                with pytest.raises(pytest.UsageError):
                    _ = tools.select_target_path(cfg, expect_exists=True)  # type: ignore

        def test_target_name_set_invalid_value(self, built_tmp_project: tuple[str, Path]) -> None:
            _, path = built_tmp_project
            with cwd(path):
                tools = CargoTools()
                # Create mock.
                invalid_target_name = "invalid_target_name"
                cfg = MockConfig({"--target-name": invalid_target_name})

                # Run.
                with pytest.raises(pytest.UsageError):
                    _ = tools.select_target_path(cfg, expect_exists=True)  # type: ignore

        def test_target_name_not_expect_exists(self, tmp_project: tuple[str, Path]) -> None:
            target_name, path = tmp_project
            with cwd(path):
                tools = CargoTools()
                # Find executable path.
                exp_target_path = tools.find_target_path(target_name, expect_exists=False)

                # Create mock.
                cfg = MockConfig({"--target-name": target_name})

                # Run.
                act_target_path = tools.select_target_path(cfg, expect_exists=False)  # type: ignore

                # Check returned path is as expected.
                assert act_target_path == exp_target_path

        def test_target_name_timeout(self, built_tmp_project: tuple[str, Path]) -> None:
            target_name, path = built_tmp_project
            with cwd(path):
                command_timeout = 0.00000001
                tools = CargoTools(command_timeout=command_timeout)
                # Create mock.
                cfg = MockConfig({"--target-name": target_name})

                # Run.
                with pytest.raises(TimeoutExpired):
                    _ = tools.select_target_path(cfg, expect_exists=True)  # type: ignore

        def test_params_unset(self, built_tmp_project: tuple[str, Path]) -> None:
            _, path = built_tmp_project
            with cwd(path):
                tools = CargoTools()
                # Create mock.
                cfg = MockConfig({})

                # Run.
                with pytest.raises(pytest.UsageError):
                    _ = tools.select_target_path(cfg, expect_exists=True)  # type: ignore


class TestBazelTools(TestBuildTools):
    """
    Test cases for Bazel tools.
    """

    @pytest.fixture(scope="class")
    def tools_type(self) -> type[BuildTools]:
        return BazelTools

    @pytest.fixture(scope="class")
    def tmp_project(self, class_tmp_path: Path) -> tuple[str, Path]:
        """
        Create temporary binary project using Bazel.
        Returns target name and path to project directory.

        Parameters
        ----------
        class_tmp_path : Path
            Temporary directory path.
        """

        def _write_to_file(path: Path, content: str) -> None:
            with open(path, mode="w", encoding="UTF-8") as file:
                file.write(content)

        # Target name and path to project.
        target_name = "project"
        project_path = class_tmp_path / target_name
        project_path.mkdir()

        # Create required files.
        build_path = project_path / "BUILD"
        build_content = dedent("""
            load("@rules_cc//cc:cc_binary.bzl", "cc_binary")

            cc_binary(
                name = "project",
                srcs = ["main.cpp"],
            )

        """)
        _write_to_file(build_path, build_content)

        main_cpp_path = project_path / "main.cpp"
        main_cpp_content = dedent("""
            #include <iostream>

            int main() { std::cout << "Hello, world!" << std::endl; }

        """)
        _write_to_file(main_cpp_path, main_cpp_content)

        module_path = project_path / "MODULE.bazel"
        module_content = dedent("""
            module(name = "project")
            bazel_dep(name = "rules_cc", version = "0.1.1")

        """)
        _write_to_file(module_path, module_content)

        return (target_name, project_path)

    @pytest.fixture(scope="class")
    def expected_target_path(self, tmp_project: tuple[str, Path]) -> Path:
        target_name, project_path = tmp_project
        return project_path / "bazel-out" / "k8-fastbuild" / "bin" / target_name

    # TODO: add query tests.  # noqa: FIX002
