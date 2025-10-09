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
from dataclasses import dataclass
from ipaddress import IPv4Address, IPv6Address, ip_address
from socket import (
    AF_INET,
    AF_INET6,
    AddressFamily,
)
from typing import Any

type IPAddress = IPv4Address | IPv6Address


@dataclass
class Address:
    ip: IPAddress
    port: int

    @classmethod
    def from_raw(cls, *address) -> "Address":
        """
        Convert address in tuple format to 'Address' object.

        Parameters
        ----------
        *address
            Address tuple.
            Only IP and port (first two fields) are used.
        """
        # Only 'ip' and 'port' are used for address.
        # Ignore 'flowinfo' and 'scope_id' provided with 'AF_INET6'.
        ip, port = address[:2]
        return cls(ip_address(ip), port)

    @classmethod
    def from_dict(cls, address_dict: dict[str, Any]) -> "Address":
        """
        Convert address in dict format with 'ip' and 'port' keys to 'Address' object.

        Parameters
        ----------
        address_dict
            Dictionary with 'ip' and 'port' keys.
        """
        return cls(ip_address(address_dict["ip"]), address_dict["port"])

    def to_raw(self) -> tuple[str, int]:
        """
        Convert this object to tuple address format.
        """
        return (str(self.ip), self.port)

    def family(self) -> AddressFamily:
        """
        Return current address family.
        """
        if isinstance(self.ip, IPv4Address):
            return AF_INET
        if isinstance(self.ip, IPv6Address):
            return AF_INET6
        raise RuntimeError("Unsupported address family")

    def __str__(self) -> str:
        if self.family() == AF_INET:
            return f"{self.ip}:{self.port}"
        if self.family() == AF_INET6:
            return f"[{self.ip}]:{self.port}"
        raise RuntimeError("Unsupported address family")
