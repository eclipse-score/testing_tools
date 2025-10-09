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
A container for storing and querying logs.
"""

__all__ = ["LogContainer"]

import re
from itertools import groupby
from operator import attrgetter
from typing import Any

from .result_entry import ResultEntry


class _NotSet:
    """
    Internal type for representing values not set.
    """

    def __repr__(self) -> str:
        return "<NOT_SET>"


_not_set = _NotSet()


class LogContainer:
    """
    A container for storing and querying logs.
    """

    def __init__(self, entries: list[ResultEntry] | None = None) -> None:
        """
        Create log container.
        Entries are copied on construction.

        Parameters
        ----------
        entries : list[ResultEntry] | None
            List of ResultEntry objects.
        """
        self._logs = list(entries or [])
        self._index = 0

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        if self._index < len(self._logs):
            result = self._logs[self._index]
            self._index += 1
            return result
        raise StopIteration

    def __len__(self):
        """
        Get the number of logs in the container.
        """
        return len(self._logs)

    def __getitem__(self, subscript):
        """
        Get the log by index.
        """
        return self._logs[subscript]

    def _logs_by_field_field_only(self, field: str, *, reverse: bool) -> list[ResultEntry]:
        """
        Filter logs using field only.

        Parameters
        ----------
        field : str
            Name of the field to match.
        reverse : bool
            Return logs not matched.
        """
        logs = []
        for log in self._logs:
            found_value = getattr(log, field, _not_set)
            if isinstance(found_value, _NotSet):
                if reverse:
                    logs.append(log)
                continue

            if not reverse:
                logs.append(log)
        return logs

    def _logs_by_field_regex_match(self, field: str, pattern: str, *, reverse: bool) -> list[ResultEntry]:
        """
        Filter logs using regex matching.
        Underlying field value is casted to str.

        Parameters
        ----------
        field : str
            Name of the field to match.
        reverse : bool
            Return logs not matched.
        pattern : str | _NotSet
            Regex pattern to match.
        """
        if not isinstance(pattern, str):
            raise TypeError("Pattern must be a string")

        logs = []
        regex = re.compile(pattern)
        for log in self._logs:
            found_value = getattr(log, field, _not_set)
            # Field must be set.
            if isinstance(found_value, _NotSet):
                if reverse:
                    logs.append(log)
                continue

            # Value casted to "str" must be matched.
            found = regex.search(str(found_value)) is not None
            if found ^ reverse:
                logs.append(log)
        return logs

    def _logs_by_field_exact_match(self, field: str, value: Any, *, reverse: bool) -> list[ResultEntry]:
        """
        Filter logs using exact matching.

        Parameters
        ----------
        field : str
            Name of the field to match.
        reverse : bool
            Return logs not matched.
        value : Any
            Exact value to match.
        """
        logs = []
        for log in self._logs:
            found_value = getattr(log, field, _not_set)
            # Field must be set.
            if isinstance(found_value, _NotSet):
                if reverse:
                    logs.append(log)
                continue

            # Type and value must be matched.
            found = isinstance(found_value, type(value)) and found_value == value
            if found ^ reverse:
                logs.append(log)
        return logs

    def _logs_by_field(
        self, field: str, *, reverse: bool, pattern: str | _NotSet = _not_set, value: Any | _NotSet = _not_set
    ) -> list[ResultEntry]:
        """
        Select filtration method and filter logs.

        Parameters
        ----------
        field : str
            Name of the field to match.
        reverse : bool
            Return logs not matched.
        pattern : str | _NotSet
            Regex pattern to match.
            Underlying field value is casted to str.
            Mutually exclusive with "value".
        value : Any | _NotSet
            Exact value to match.
            Mutually exclusive with "pattern".
        """
        pattern_set = not isinstance(pattern, _NotSet)
        value_set = not isinstance(value, _NotSet)

        if pattern_set and not value_set:
            return self._logs_by_field_regex_match(field, pattern, reverse=reverse)
        if not pattern_set and value_set:
            return self._logs_by_field_exact_match(field, value, reverse=reverse)
        if not pattern_set and not value_set:
            return self._logs_by_field_field_only(field, reverse=reverse)
        raise RuntimeError("Pattern and value parameters are mutually exclusive")

    def contains_log(self, field: str, *, pattern: str | _NotSet = _not_set, value: Any | _NotSet = _not_set) -> bool:
        """
        Check if the container contains logs matching the given field and pattern or value.

        Parameters
        ----------
        field : str
            Name of the field to match.
        pattern : str | _NotSet
            Regex pattern to match.
            Underlying field value is casted to str.
            Mutually exclusive with "value".
        value : Any | _NotSet
            Exact value to match.
            Mutually exclusive with "pattern".
        """
        return len(self._logs_by_field(field, reverse=False, pattern=pattern, value=value)) > 0

    def get_logs(
        self, field: str | _NotSet = _not_set, *, pattern: str | _NotSet = _not_set, value: Any | _NotSet = _not_set
    ) -> "LogContainer":
        """
        Get all logs matching the given field and pattern or value.
        - All logs are returned when no parameters are provided.
        - Only logs containing `field` are returned when provided.
        - Logs matching `field` and `pattern` or `value` are returned when provided.

        Parameters
        ----------
        field : str | _NotSet
            Name of the field to match.
            All logs are returned if not set.
        pattern : str | _NotSet
            Regex pattern to match.
            Underlying field value is casted to str.
            Mutually exclusive with "value".
        value : Any | _NotSet
            Exact value to match.
            Mutually exclusive with "pattern".
        """
        # Return copy of all logs.
        if isinstance(field, _NotSet):
            if not isinstance(pattern, _NotSet) or not isinstance(value, _NotSet):
                raise RuntimeError("Matching by pattern or value without field is not supported")
            return LogContainer(self._logs[:])

        return LogContainer(self._logs_by_field(field, reverse=False, pattern=pattern, value=value))

    def find_log(
        self, field: str, *, pattern: str | _NotSet = _not_set, value: Any | _NotSet = _not_set
    ) -> ResultEntry | None:
        """
        Find a log that matches the given field and pattern or value.
        Returns the first match, or None if no match is found.
        Raises ValueError if multiple matches were found.

        Parameters
        ----------
        field : str
            Name of the field to match.
        pattern : str | _NotSet
            Regex pattern to match.
            Underlying field value is casted to str.
            Mutually exclusive with "value".
        value : Any | _NotSet
            Exact value to match.
            Mutually exclusive with "pattern".
        """
        findings = self._logs_by_field(field, reverse=False, pattern=pattern, value=value)
        if len(findings) == 1:
            return findings[0]
        if len(findings) > 1:
            raise ValueError(f"Multiple logs found for {field=} with {pattern=} and {value=}")

        return None

    def add_log(self, log: ResultEntry | list[ResultEntry]) -> None:
        """
        Add log to the container.

        Parameters
        ----------
        log : ResultEntry | list[ResultEntry]
            Logs to be added.
        """
        if isinstance(log, ResultEntry):
            self._logs.append(log)
        elif isinstance(log, list) and all(isinstance(x, ResultEntry) for x in log):
            self._logs.extend(log)
        else:
            raise TypeError("log must be a ResultEntry or list[ResultEntry]")

    def remove_logs(
        self, field: str, *, pattern: str | _NotSet = _not_set, value: Any | _NotSet = _not_set
    ) -> "LogContainer":
        """
        Remove all logs matching the given field and pattern or value.

        Parameters
        ----------
        field : str
            Name of the field to match.
        pattern : str | _NotSet
            Regex pattern to match.
            Underlying field value is casted to str.
            Mutually exclusive with "value".
        value : Any | _NotSet
            Exact value to match.
            Mutually exclusive with "pattern".
        """
        return LogContainer(self._logs_by_field(field, reverse=True, pattern=pattern, value=value))

    def group_by(self, attribute: str) -> dict[str, "LogContainer"]:
        """
        Group logs by the given attribute.
        Returns a dictionary where the keys are the unique values of the attribute,
        and the values are LogContainer instances containing the grouped logs.

        Parameters
        ----------
        attribute : str
            Attribute to group logs.
        """
        sorted_logs_by_attr = sorted(self._logs, key=attrgetter(attribute))
        grouped = groupby(sorted_logs_by_attr, key=attrgetter(attribute))
        return {key: LogContainer(list(group)) for key, group in grouped}
