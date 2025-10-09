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
Tests for "log_container" module.
"""

from datetime import timedelta
from typing import Any

import pytest

from testing_utils import LogContainer, ResultEntry


@pytest.fixture
def lc_basic() -> LogContainer:
    """
    Basic log container provider.
    """
    entries = [ResultEntry({"index": i}) for i in range(10)]
    return LogContainer(entries)


class TestInit:
    """
    Tests for `__init__`.
    """

    def test_ok(self, lc_basic: LogContainer):
        # Construction happens in fixture.
        assert len(lc_basic) == 10

    def test_default_empty(self):
        # Make sure internal storage is always list.
        lc = LogContainer()
        assert len(lc) == 0
        assert isinstance(lc._logs, list)  # noqa: SLF001

    def test_explicit_none(self):
        # Make sure internal storage is always list.
        lc = LogContainer(None)
        assert len(lc) == 0
        assert isinstance(lc._logs, list)  # noqa: SLF001

    def test_default_param_not_referenced(self):
        lc1 = LogContainer()
        lc1.add_log(ResultEntry({}))
        lc2 = LogContainer()
        lc2.add_log(ResultEntry({}))

        assert len(lc1) == 1
        assert len(lc2) == 1

    def test_param_copied_not_referenced(self):
        logs1 = [ResultEntry({})]
        lc1 = LogContainer(logs1)
        lc1.add_log(ResultEntry({}))
        lc2 = LogContainer()
        lc2.add_log(ResultEntry({}))

        assert len(logs1) == 1
        assert len(lc1) == 2
        assert len(lc2) == 1


class TestIterNext:
    """
    Tests for `__iter__` and `__next__`.
    """

    def test_iterator_ok(self, lc_basic: LogContainer):
        for i, log in enumerate(lc_basic):
            assert log.index == i

    def test_iterator_empty(self):
        lc = LogContainer()
        for _ in lc:
            raise RuntimeError("Statement shouldn't be reached")

    def test_list_ok(self, lc_basic: LogContainer):
        logs = list(lc_basic)
        assert len(logs) == 10
        assert all(log.index == i for i, log in enumerate(logs))

    def test_list_empty(self):
        lc = LogContainer()
        logs = list(lc)
        assert len(logs) == 0


class TestLen:
    """
    Tests for `__len__`.
    """

    def test_ok(self):
        lc = LogContainer([ResultEntry({}), ResultEntry({}), ResultEntry({})])
        assert len(lc) == 3

    def test_empty(self):
        lc = LogContainer()
        assert len(lc) == 0


class TestGetItem:
    """
    Tests for `__getitem__`.
    """

    def test_ok(self, lc_basic: LogContainer):
        for i in range(10):
            assert lc_basic[i].index == i

    def test_out_of_range(self, lc_basic: LogContainer):
        with pytest.raises(IndexError):
            _ = lc_basic[20]

    def test_negative(self, lc_basic: LogContainer):
        assert lc_basic[-1].index == 9


class TestContainsLog:
    """
    Tests for `contains_logs`.
    """

    def test_both_pattern_and_value(self):
        lc = LogContainer()
        with pytest.raises(RuntimeError):
            _ = lc.contains_log("level", pattern="pattern", value="value")

    def test_field_ok(self):
        lc = LogContainer()
        lc.add_log(
            [
                ResultEntry({"level": "DEBUG"}),
                ResultEntry({"level": "INFO"}),
                ResultEntry({"level": "WARN"}),
            ]
        )
        assert lc.contains_log("level")

    def test_field_not_found(self):
        lc = LogContainer()
        lc.add_log(
            [
                ResultEntry({"level": "DEBUG"}),
                ResultEntry({"level": "INFO"}),
                ResultEntry({"level": "WARN"}),
            ]
        )
        assert not lc.contains_log("invalid")

    def test_pattern_str_ok(self):
        lc = LogContainer()
        lc.add_log(
            [
                ResultEntry({"level": "DEBUG"}),
                ResultEntry({"level": "INFO"}),
                ResultEntry({"level": "WARN"}),
            ]
        )
        assert lc.contains_log("level", pattern=r"WARN|INFO")

    def test_pattern_int_ok(self):
        lc = LogContainer()
        lc.add_log(
            [
                ResultEntry({"someId": 1}),
                ResultEntry({"someId": 2}),
                ResultEntry({"someId": 3}),
                ResultEntry({"someId": 1}),
                ResultEntry({"someId": 1}),
                ResultEntry({"someId": 11}),
                ResultEntry({"someId": 12}),
            ]
        )
        assert lc.contains_log("some_id", pattern=r"^1$")

    def test_pattern_cast_type(self):
        lc = LogContainer()
        lc.add_log(
            [
                ResultEntry({"level": "DEBUG", "someId": "0"}),
                ResultEntry({"level": "DEBUG", "someId": 0}),
                ResultEntry({"level": "DEBUG", "someId": 6543}),
                ResultEntry({"level": "DEBUG", "someId": "10"}),
                ResultEntry({"level": "DEBUG", "someId": 100}),
            ]
        )
        assert lc.contains_log("some_id", pattern="^0$")
        assert lc.contains_log("some_id", pattern="0")

    def test_pattern_invalid_field(self):
        lc = LogContainer()
        lc.add_log(
            [
                ResultEntry({"level": "DEBUG"}),
                ResultEntry({"level": "INFO"}),
                ResultEntry({"level": "WARN"}),
            ]
        )
        assert not lc.contains_log("invalid", pattern="WARN")

    def test_pattern_invalid_value(self):
        lc = LogContainer()
        lc.add_log(
            [
                ResultEntry({"level": "DEBUG"}),
                ResultEntry({"level": "INFO"}),
                ResultEntry({"level": "WARN"}),
            ]
        )
        assert not lc.contains_log("level", pattern="invalid")

    def test_value_str_ok(self):
        lc = LogContainer()
        lc.add_log(
            [
                ResultEntry({"level": "DEBUG"}),
                ResultEntry({"level": "INFO"}),
                ResultEntry({"level": "WARN"}),
            ]
        )
        assert lc.contains_log("level", value="INFO")

    def test_value_int_ok(self):
        lc = LogContainer()
        lc.add_log(
            [
                ResultEntry({"someId": 1}),
                ResultEntry({"someId": 2}),
                ResultEntry({"someId": 3}),
                ResultEntry({"someId": 1}),
                ResultEntry({"someId": 1}),
                ResultEntry({"someId": 11}),
                ResultEntry({"someId": 12}),
            ]
        )
        assert lc.contains_log("some_id", value=1)

    def test_value_none_ok(self):
        lc = LogContainer()
        lc.add_log(
            [
                ResultEntry({"level": "DEBUG", "someId": 1}),
                ResultEntry({"level": "DEBUG", "someId": 2}),
                ResultEntry({"level": "INFO", "someId": None}),
                ResultEntry({"level": "WARN", "someId": 2}),
                ResultEntry({"level": "INFO", "someId": 1}),
            ]
        )
        assert lc.contains_log("some_id", value=None)

    def test_value_filter_type(self):
        lc = LogContainer()
        lc.add_log(
            [
                ResultEntry({"level": "DEBUG", "someId": "0"}),
                ResultEntry({"level": "DEBUG", "someId": 0}),
                ResultEntry({"level": "DEBUG", "someId": 6543}),
                ResultEntry({"level": "DEBUG", "someId": "10"}),
                ResultEntry({"level": "DEBUG", "someId": 100}),
            ]
        )
        assert lc.contains_log("some_id", value="0")
        assert lc.contains_log("some_id", value=0)

    def test_value_invalid_field(self):
        lc = LogContainer()
        lc.add_log(
            [
                ResultEntry({"level": "DEBUG"}),
                ResultEntry({"level": "INFO"}),
                ResultEntry({"level": "WARN"}),
            ]
        )
        assert not lc.contains_log("invalid", value="WARN")

    def test_value_invalid_str_value(self):
        lc = LogContainer()
        lc.add_log(
            [
                ResultEntry({"level": "DEBUG"}),
                ResultEntry({"level": "INFO"}),
                ResultEntry({"level": "WARN"}),
            ]
        )
        assert not lc.contains_log("level", value="invalid")

    def test_value_invalid_int_value(self):
        lc = LogContainer()
        lc.add_log(
            [
                ResultEntry({"someId": 1}),
                ResultEntry({"someId": 2}),
                ResultEntry({"someId": 3}),
                ResultEntry({"someId": 1}),
                ResultEntry({"someId": 1}),
            ]
        )
        assert not lc.contains_log("some_id", value=10)


class TestGetLogs:
    """
    Tests for `get_logs`.
    """

    def test_both_pattern_and_value(self):
        lc = LogContainer()
        with pytest.raises(RuntimeError):
            _ = lc.get_logs("level", pattern="pattern", value="value")

    def test_no_params_ok(self):
        lc = LogContainer()
        lc.add_log(
            [
                ResultEntry({"level": "DEBUG"}),
                ResultEntry({"level": "INFO"}),
                ResultEntry({"level": "WARN"}),
                ResultEntry({"flag": False}),
            ]
        )
        logs = lc.get_logs()
        assert len(logs) == 4

    def test_no_field_value_set(self):
        lc = LogContainer()
        lc.add_log(
            [
                ResultEntry({"level": "DEBUG"}),
                ResultEntry({"level": "INFO"}),
                ResultEntry({"level": "WARN"}),
            ]
        )
        with pytest.raises(RuntimeError):
            _ = lc.get_logs(value="WARN")

    def test_no_field_pattern_set(self):
        lc = LogContainer()
        lc.add_log(
            [
                ResultEntry({"level": "DEBUG"}),
                ResultEntry({"level": "INFO"}),
                ResultEntry({"level": "WARN"}),
            ]
        )
        with pytest.raises(RuntimeError):
            _ = lc.get_logs(pattern=r"WARN|INFO")

    def test_field_ok(self):
        lc = LogContainer()
        lc.add_log(
            [
                ResultEntry({"level": "DEBUG"}),
                ResultEntry({"level": "INFO"}),
                ResultEntry({"level": "WARN"}),
                ResultEntry({"flag": False}),
            ]
        )
        logs = lc.get_logs("level")
        assert len(logs) == 3

    def test_field_not_found(self):
        lc = LogContainer()
        lc.add_log(
            [
                ResultEntry({"level": "DEBUG"}),
                ResultEntry({"level": "INFO"}),
                ResultEntry({"level": "WARN"}),
                ResultEntry({"flag": False}),
            ]
        )
        logs = lc.get_logs("invalid")
        assert len(logs) == 0

    def test_pattern_str_ok(self):
        lc = LogContainer()
        lc.add_log(
            [
                ResultEntry({"level": "DEBUG"}),
                ResultEntry({"level": "INFO"}),
                ResultEntry({"level": "WARN"}),
            ]
        )
        logs = lc.get_logs("level", pattern=r"WARN|INFO")
        assert len(logs) == 2
        assert logs[0].level == "INFO"
        assert logs[1].level == "WARN"

    def test_pattern_int_ok(self):
        lc = LogContainer()
        lc.add_log(
            [
                ResultEntry({"someId": 1}),
                ResultEntry({"someId": 2}),
                ResultEntry({"someId": 3}),
                ResultEntry({"someId": 1}),
                ResultEntry({"someId": 1}),
                ResultEntry({"someId": 11}),
                ResultEntry({"someId": 12}),
            ]
        )
        logs = lc.get_logs("some_id", pattern=r"^1$")
        assert len(logs) == 3
        assert all(log.some_id == 1 for log in logs)

    def test_pattern_cast_type(self):
        lc = LogContainer()
        lc.add_log(
            [
                ResultEntry({"level": "DEBUG", "someId": "0"}),
                ResultEntry({"level": "DEBUG", "someId": 0}),
                ResultEntry({"level": "DEBUG", "someId": 6543}),
                ResultEntry({"level": "DEBUG", "someId": "10"}),
                ResultEntry({"level": "DEBUG", "someId": 100}),
            ]
        )
        logs = lc.get_logs("some_id", pattern="^0$")
        assert len(logs) == 2
        assert logs[0].some_id == "0"
        assert logs[1].some_id == 0

        logs = lc.get_logs("some_id", pattern="0")
        assert len(logs) == 4
        assert logs[0].some_id == "0"
        assert logs[1].some_id == 0
        assert logs[2].some_id == "10"
        assert logs[3].some_id == 100

    def test_pattern_invalid_field(self):
        lc = LogContainer()
        lc.add_log(
            [
                ResultEntry({"level": "DEBUG"}),
                ResultEntry({"level": "INFO"}),
                ResultEntry({"level": "WARN"}),
            ]
        )
        logs = lc.get_logs("invalid", pattern="WARN")
        assert len(logs) == 0

    def test_pattern_invalid_value(self):
        lc = LogContainer()
        lc.add_log(
            [
                ResultEntry({"level": "DEBUG"}),
                ResultEntry({"level": "INFO"}),
                ResultEntry({"level": "WARN"}),
            ]
        )
        logs = lc.get_logs("level", pattern="invalid")
        assert len(logs) == 0

    def test_value_str_ok(self):
        lc = LogContainer()
        lc.add_log(
            [
                ResultEntry({"level": "DEBUG"}),
                ResultEntry({"level": "INFO"}),
                ResultEntry({"level": "WARN"}),
            ]
        )
        logs = lc.get_logs("level", value="INFO")
        assert len(logs) == 1
        assert logs[0].level == "INFO"

    def test_value_int_ok(self):
        lc = LogContainer()
        lc.add_log(
            [
                ResultEntry({"someId": 1}),
                ResultEntry({"someId": 2}),
                ResultEntry({"someId": 3}),
                ResultEntry({"someId": 1}),
                ResultEntry({"someId": 1}),
                ResultEntry({"someId": 11}),
                ResultEntry({"someId": 12}),
            ]
        )
        logs = lc.get_logs("some_id", value=1)
        assert len(logs) == 3
        assert all(log.some_id == 1 for log in logs)

    def test_value_none_ok(self):
        lc = LogContainer()
        lc.add_log(
            [
                ResultEntry({"level": "DEBUG", "someId": 1}),
                ResultEntry({"level": "DEBUG", "someId": 2}),
                ResultEntry({"level": "INFO", "someId": None}),
                ResultEntry({"level": "WARN", "someId": 2}),
                ResultEntry({"level": "INFO", "someId": 1}),
            ]
        )
        logs = lc.get_logs("some_id", value=None)
        assert len(logs) == 1
        assert logs[0].some_id is None

    def test_value_filter_type(self):
        lc = LogContainer()
        lc.add_log(
            [
                ResultEntry({"level": "DEBUG", "someId": "0"}),
                ResultEntry({"level": "DEBUG", "someId": 0}),
                ResultEntry({"level": "DEBUG", "someId": 6543}),
                ResultEntry({"level": "DEBUG", "someId": "10"}),
                ResultEntry({"level": "DEBUG", "someId": 100}),
            ]
        )
        logs = lc.get_logs("some_id", value="0")
        assert len(logs) == 1
        assert logs[0].some_id == "0"

        logs = lc.get_logs("some_id", value=0)
        assert len(logs) == 1
        assert logs[0].some_id == 0

    def test_value_invalid_field(self):
        lc = LogContainer()
        lc.add_log(
            [
                ResultEntry({"level": "DEBUG"}),
                ResultEntry({"level": "INFO"}),
                ResultEntry({"level": "WARN"}),
            ]
        )
        logs = lc.get_logs("invalid", value="WARN")
        assert len(logs) == 0

    def test_value_invalid_str_value(self):
        lc = LogContainer()
        lc.add_log(
            [
                ResultEntry({"level": "DEBUG"}),
                ResultEntry({"level": "INFO"}),
                ResultEntry({"level": "WARN"}),
            ]
        )
        logs = lc.get_logs("level", value="invalid")
        assert len(logs) == 0

    def test_value_invalid_int_value(self):
        lc = LogContainer()
        lc.add_log(
            [
                ResultEntry({"someId": 1}),
                ResultEntry({"someId": 2}),
                ResultEntry({"someId": 3}),
                ResultEntry({"someId": 1}),
                ResultEntry({"someId": 1}),
            ]
        )
        logs = lc.get_logs("some_id", value=10)
        assert len(logs) == 0


class TestFindLog:
    """
    Tests for `find_log`.
    """

    def test_both_pattern_and_value(self):
        lc = LogContainer()
        with pytest.raises(RuntimeError):
            _ = lc.find_log("level", pattern="pattern", value="value")

    def test_field_ok(self):
        lc = LogContainer()
        lc.add_log([ResultEntry({"level": "DEBUG"}), ResultEntry({"level": "INFO"}), ResultEntry({"flag": True})])
        log = lc.find_log("flag")
        assert log
        assert log.flag

    def test_field_not_found(self):
        lc = LogContainer()
        lc.add_log([ResultEntry({"level": "DEBUG"}), ResultEntry({"level": "INFO"}), ResultEntry({"flag": True})])
        log = lc.find_log("invalid")
        assert log is None

    def test_pattern_str_ok(self):
        lc = LogContainer()
        lc.add_log(
            [
                ResultEntry({"level": "DEBUG"}),
                ResultEntry({"level": "INFO"}),
            ]
        )
        log = lc.find_log("level", pattern=r"WARN|INFO")
        assert log
        assert log.level == "INFO"

    def test_pattern_int_ok(self):
        lc = LogContainer()
        lc.add_log(
            [
                ResultEntry({"someId": 1}),
                ResultEntry({"someId": 2}),
                ResultEntry({"someId": 3}),
                ResultEntry({"someId": 11}),
                ResultEntry({"someId": 12}),
            ]
        )
        log = lc.find_log("some_id", pattern=r"^1$")
        assert log
        assert log.some_id == 1

    def test_pattern_many_found(self):
        lc = LogContainer()
        lc.add_log(
            [
                ResultEntry({"level": "DEBUG"}),
                ResultEntry({"level": "INFO"}),
                ResultEntry({"level": "WARN"}),
            ]
        )
        with pytest.raises(ValueError):  # noqa: PT011
            _ = lc.find_log("level", pattern=r"WARN|INFO")

    def test_pattern_cast_type(self):
        lc = LogContainer()
        lc.add_log(
            [
                ResultEntry({"level": "DEBUG", "someId": 0}),
                ResultEntry({"level": "DEBUG", "someId": 6543}),
                ResultEntry({"level": "DEBUG", "someId": "10"}),
                ResultEntry({"level": "DEBUG", "someId": 100}),
            ]
        )
        log = lc.find_log("some_id", pattern="^0$")
        assert log
        assert log.some_id == 0

    def test_pattern_invalid_field(self):
        lc = LogContainer()
        lc.add_log(
            [
                ResultEntry({"level": "DEBUG"}),
                ResultEntry({"level": "INFO"}),
                ResultEntry({"level": "WARN"}),
            ]
        )
        assert lc.find_log("invalid", pattern="WARN") is None

    def test_pattern_invalid_value(self):
        lc = LogContainer()
        lc.add_log(
            [
                ResultEntry({"level": "DEBUG"}),
                ResultEntry({"level": "INFO"}),
                ResultEntry({"level": "WARN"}),
            ]
        )
        assert lc.find_log("level", pattern="invalid") is None

    def test_value_str_ok(self):
        lc = LogContainer()
        lc.add_log(
            [
                ResultEntry({"level": "DEBUG"}),
                ResultEntry({"level": "INFO"}),
                ResultEntry({"level": "WARN"}),
            ]
        )
        log = lc.find_log("level", value="INFO")
        assert log
        assert log.level == "INFO"

    def test_value_int_ok(self):
        lc = LogContainer()
        lc.add_log(
            [
                ResultEntry({"someId": 1}),
                ResultEntry({"someId": 2}),
                ResultEntry({"someId": 3}),
                ResultEntry({"someId": 11}),
                ResultEntry({"someId": 12}),
            ]
        )
        log = lc.find_log("some_id", value=1)
        assert log
        assert log.some_id == 1

    def test_value_none_ok(self):
        lc = LogContainer()
        lc.add_log(
            [
                ResultEntry({"level": "DEBUG", "someId": 1}),
                ResultEntry({"level": "DEBUG", "someId": 2}),
                ResultEntry({"level": "INFO", "someId": None}),
                ResultEntry({"level": "WARN", "someId": 2}),
                ResultEntry({"level": "INFO", "someId": 1}),
            ]
        )
        log = lc.find_log("some_id", value=None)
        assert log
        assert log.some_id is None

    def test_value_filter_type(self):
        lc = LogContainer()
        lc.add_log(
            [
                ResultEntry({"level": "DEBUG", "someId": "0"}),
                ResultEntry({"level": "DEBUG", "someId": 0}),
                ResultEntry({"level": "DEBUG", "someId": 6543}),
                ResultEntry({"level": "DEBUG", "someId": "10"}),
                ResultEntry({"level": "DEBUG", "someId": 100}),
            ]
        )
        log = lc.find_log("some_id", value="0")
        assert log
        assert log.some_id == "0"

        log = lc.find_log("some_id", value=0)
        assert log
        assert log.some_id == 0

    def test_value_invalid_field(self):
        lc = LogContainer()
        lc.add_log(
            [
                ResultEntry({"level": "DEBUG"}),
                ResultEntry({"level": "INFO"}),
                ResultEntry({"level": "WARN"}),
            ]
        )
        assert lc.find_log("invalid", value="WARN") is None

    def test_value_invalid_str_value(self):
        lc = LogContainer()
        lc.add_log(
            [
                ResultEntry({"level": "DEBUG"}),
                ResultEntry({"level": "INFO"}),
                ResultEntry({"level": "WARN"}),
            ]
        )
        assert lc.find_log("level", value="invalid") is None

    def test_value_invalid_int_value(self):
        lc = LogContainer()
        lc.add_log(
            [
                ResultEntry({"someId": 1}),
                ResultEntry({"someId": 2}),
                ResultEntry({"someId": 3}),
                ResultEntry({"someId": 1}),
                ResultEntry({"someId": 1}),
            ]
        )
        assert lc.find_log("some_id", value=10) is None


class TestAddLog:
    """
    Tests for `add_log`.
    """

    def _common_entries(self) -> list[ResultEntry]:
        return [
            ResultEntry(
                {
                    "timestamp": "0:00:01.000100",
                    "level": "DEBUG",
                    "fields": {"message": "Debug message"},
                    "target": "target::DEBUG_message",
                    "threadId": "ThreadId(1)",
                }
            ),
            ResultEntry(
                {
                    "timestamp": "0:00:01.000101",
                    "level": "INFO",
                    "target": "target::INFO_message",
                    "threadId": "ThreadId(2)",
                }
            ),
        ]

    def _common_check(self, lc: LogContainer) -> None:
        logs = list(lc)
        assert len(logs) == 2

        assert logs[0].timestamp == str(timedelta(microseconds=1000100))
        assert logs[0].level == "DEBUG"
        assert logs[0].message == "Debug message"
        assert logs[0].target == "target::DEBUG_message"
        assert logs[0].thread_id == "ThreadId(1)"

        assert logs[1].timestamp == str(timedelta(microseconds=1000101))
        assert logs[1].level == "INFO"
        assert logs[1].target == "target::INFO_message"
        assert logs[1].thread_id == "ThreadId(2)"

    def test_single_ok(self):
        lc = LogContainer()
        for entry in self._common_entries():
            lc.add_log(entry)
        self._common_check(lc)

    def test_single_param_copied_not_referenced(self):
        logs = [ResultEntry({})]
        lc = LogContainer(logs)
        lc.add_log(ResultEntry({}))

        assert len(logs) == 1
        assert len(lc) == 2

    def test_many_ok(self):
        lc = LogContainer()
        lc.add_log(self._common_entries())
        self._common_check(lc)

    def test_many_param_copied_not_referenced(self):
        logs1 = [ResultEntry({}), ResultEntry({})]
        lc = LogContainer(logs1)
        logs2 = [ResultEntry({}), ResultEntry({}), ResultEntry({})]
        lc.add_log(logs2)

        assert len(logs1) == 2
        assert len(logs2) == 3
        assert len(lc) == 5

    @pytest.mark.parametrize("value", ["raw_string", None, 123, False])
    def test_invalid_type(self, value: Any):
        lc = LogContainer()
        with pytest.raises(TypeError):
            lc.add_log(value)  # type: ignore


class TestRemoveLogs:
    """
    Tests for `remove_logs`.
    """

    def test_both_pattern_and_value(self):
        lc = LogContainer()
        with pytest.raises(RuntimeError):
            _ = lc.remove_logs("level", pattern="pattern", value="value")

    def test_field_ok(self):
        lc = LogContainer()
        lc.add_log(
            [
                ResultEntry({"level": "DEBUG"}),
                ResultEntry({"level": "INFO"}),
                ResultEntry({"level": "WARN"}),
                ResultEntry({"flag": False}),
            ]
        )
        logs = lc.remove_logs("level")
        assert len(logs) == 1
        assert not logs[0].flag

    def test_field_not_found(self):
        lc = LogContainer()
        lc.add_log(
            [
                ResultEntry({"level": "DEBUG"}),
                ResultEntry({"level": "INFO"}),
                ResultEntry({"level": "WARN"}),
                ResultEntry({"flag": False}),
            ]
        )
        logs = lc.remove_logs("invalid")
        assert len(logs) == 4

    def test_pattern_str_ok(self):
        lc = LogContainer()
        lc.add_log(
            [
                ResultEntry({"level": "DEBUG"}),
                ResultEntry({"level": "INFO"}),
                ResultEntry({"level": "WARN"}),
            ]
        )
        logs = lc.remove_logs("level", pattern=r"WARN|INFO")
        assert len(logs) == 1
        assert logs[0].level == "DEBUG"

    def test_pattern_int_ok(self):
        lc = LogContainer()
        lc.add_log(
            [
                ResultEntry({"someId": 1}),
                ResultEntry({"someId": 2}),
                ResultEntry({"someId": 3}),
                ResultEntry({"someId": 1}),
                ResultEntry({"someId": 1}),
                ResultEntry({"someId": 11}),
                ResultEntry({"someId": 12}),
            ]
        )
        logs = lc.remove_logs("some_id", pattern=r"^1$")
        assert len(logs) == 4
        assert logs[0].some_id == 2
        assert logs[1].some_id == 3
        assert logs[2].some_id == 11
        assert logs[3].some_id == 12

    def test_pattern_cast_type(self):
        lc = LogContainer()
        lc.add_log(
            [
                ResultEntry({"level": "DEBUG", "someId": "0"}),
                ResultEntry({"level": "DEBUG", "someId": 0}),
                ResultEntry({"level": "DEBUG", "someId": 6543}),
                ResultEntry({"level": "DEBUG", "someId": "10"}),
                ResultEntry({"level": "DEBUG", "someId": 100}),
            ]
        )
        logs = lc.remove_logs("some_id", pattern="^0$")
        assert len(logs) == 3
        assert logs[0].some_id == 6543
        assert logs[1].some_id == "10"
        assert logs[2].some_id == 100

    def test_pattern_invalid_field(self):
        lc = LogContainer()
        lc.add_log(
            [
                ResultEntry({"level": "DEBUG"}),
                ResultEntry({"level": "INFO"}),
                ResultEntry({"level": "WARN"}),
            ]
        )
        logs = lc.remove_logs("invalid", pattern="WARN")
        assert len(logs) == 3

    def test_pattern_invalid_value(self):
        lc = LogContainer()
        lc.add_log(
            [
                ResultEntry({"level": "DEBUG"}),
                ResultEntry({"level": "INFO"}),
                ResultEntry({"level": "WARN"}),
            ]
        )
        logs = lc.remove_logs("level", pattern="invalid")
        assert len(logs) == 3

    def test_value_str_ok(self):
        lc = LogContainer()
        lc.add_log(
            [
                ResultEntry({"level": "DEBUG"}),
                ResultEntry({"level": "INFO"}),
                ResultEntry({"level": "WARN"}),
            ]
        )
        logs = lc.remove_logs("level", value="INFO")
        assert len(logs) == 2
        assert logs[0].level == "DEBUG"
        assert logs[1].level == "WARN"

    def test_value_int_ok(self):
        lc = LogContainer()
        lc.add_log(
            [
                ResultEntry({"someId": 1}),
                ResultEntry({"someId": 2}),
                ResultEntry({"someId": 3}),
                ResultEntry({"someId": 1}),
                ResultEntry({"someId": 1}),
                ResultEntry({"someId": 11}),
                ResultEntry({"someId": 12}),
            ]
        )
        logs = lc.remove_logs("some_id", value=1)
        assert len(logs) == 4
        assert logs[0].some_id == 2
        assert logs[1].some_id == 3
        assert logs[2].some_id == 11
        assert logs[3].some_id == 12

    def test_value_none_ok(self):
        lc = LogContainer()
        lc.add_log(
            [
                ResultEntry({"level": "DEBUG", "someId": 1}),
                ResultEntry({"level": "DEBUG", "someId": 2}),
                ResultEntry({"level": "INFO", "someId": None}),
                ResultEntry({"level": "WARN", "someId": 2}),
                ResultEntry({"level": "INFO", "someId": 1}),
            ]
        )
        logs = lc.remove_logs("some_id", value=None)
        assert len(logs) == 4
        assert all(log is not None for log in logs)

    def test_value_filter_type(self):
        lc = LogContainer()
        lc.add_log(
            [
                ResultEntry({"level": "DEBUG", "someId": "0"}),
                ResultEntry({"level": "DEBUG", "someId": 0}),
                ResultEntry({"level": "DEBUG", "someId": 6543}),
                ResultEntry({"level": "DEBUG", "someId": "10"}),
                ResultEntry({"level": "DEBUG", "someId": 100}),
            ]
        )

        logs = lc.remove_logs("some_id", value="0")
        assert len(logs) == 4
        assert logs[0].some_id == 0
        assert logs[1].some_id == 6543
        assert logs[2].some_id == "10"
        assert logs[3].some_id == 100

        logs = lc.remove_logs("some_id", value=0)
        assert len(logs) == 4
        assert logs[0].some_id == "0"
        assert logs[1].some_id == 6543
        assert logs[2].some_id == "10"
        assert logs[3].some_id == 100

    def test_value_invalid_field(self):
        lc = LogContainer()
        lc.add_log(
            [
                ResultEntry({"level": "DEBUG"}),
                ResultEntry({"level": "INFO"}),
                ResultEntry({"level": "WARN"}),
            ]
        )
        logs = lc.remove_logs("invalid", value="WARN")
        assert len(logs) == 3

    def test_value_invalid_str_value(self):
        lc = LogContainer()
        lc.add_log(
            [
                ResultEntry({"level": "DEBUG"}),
                ResultEntry({"level": "INFO"}),
                ResultEntry({"level": "WARN"}),
            ]
        )
        logs = lc.remove_logs("level", value="invalid")
        assert len(logs) == 3

    def test_value_invalid_int_value(self):
        lc = LogContainer()
        lc.add_log(
            [
                ResultEntry({"someId": 1}),
                ResultEntry({"someId": 2}),
                ResultEntry({"someId": 3}),
                ResultEntry({"someId": 1}),
                ResultEntry({"someId": 1}),
            ]
        )
        logs = lc.remove_logs("some_id", value=10)
        assert len(logs) == 5


class TestGroupBy:
    """
    Tests for `group_by`.
    """

    def test_ok(self):
        lc = LogContainer()
        lc.add_log(
            ResultEntry(
                {
                    "timestamp": "0:00:00.000001",
                    "level": "INFO",
                    "fields": {"message": "Info message 1"},
                    "target": "target::INFO_message",
                    "threadId": "ThreadId(2)",
                }
            )
        )
        lc.add_log(
            ResultEntry(
                {
                    "timestamp": "0:00:00.000002",
                    "level": "INFO",
                    "fields": {"message": "Info message 2"},
                    "target": "target::INFO_message",
                    "threadId": "ThreadId(1)",
                }
            )
        )
        lc.add_log(
            ResultEntry(
                {
                    "timestamp": "0:00:00.000003",
                    "level": "INFO",
                    "fields": {"message": "Info message 3"},
                    "target": "target::INFO_message",
                    "threadId": "ThreadId(2)",
                }
            )
        )
        groups = lc.group_by("thread_id")
        assert len(groups) == 2
        assert len(groups["ThreadId(1)"]) == 1
        assert len(groups["ThreadId(2)"]) == 2
        assert groups["ThreadId(1)"][0].message == "Info message 2"
        assert groups["ThreadId(2)"][0].message == "Info message 1"
        assert groups["ThreadId(2)"][1].message == "Info message 3"
