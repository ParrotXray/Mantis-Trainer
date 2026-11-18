import ipaddress
from typing import Dict, Iterable, List, Sequence, Tuple, Union

import pandas as pd

Network = Union[ipaddress.IPv4Network, ipaddress.IPv6Network]


class IpFilter:
    def __init__(self, networks: Sequence[Network]) -> None:
        self.networks = list(networks)

    @staticmethod
    def parse(values: Iterable[str]) -> List[Network]:
        networks = []
        for value in values:
            try:
                networks.append(ipaddress.ip_network(value.strip(), strict=False))
            except ValueError as exc:
                raise ValueError(
                    f"'{value}' is not an IP address or CIDR block"
                ) from exc
        return networks

    @classmethod
    def from_values(cls, values: Iterable[str]) -> "IpFilter":
        return cls(cls.parse(values))

    def __bool__(self) -> bool:
        return bool(self.networks)

    def _owner(self, value: str) -> Union[Network, None]:
        try:
            address = ipaddress.ip_address(value)
        except ValueError:
            return None
        return next(
            (
                net
                for net in self.networks
                if net.version == address.version and address in net
            ),
            None,
        )

    def match(self, column: pd.Series) -> Tuple[pd.Series, Dict[str, Tuple[int, int]]]:
        addresses = column.astype(str).str.strip()
        owners = {value: self._owner(value) for value in addresses.unique()}
        matched = {value for value, net in owners.items() if net is not None}
        mask = addresses.isin(matched)
        counts = addresses[mask].value_counts()
        report: Dict[str, Tuple[int, int]] = {str(net): (0, 0) for net in self.networks}
        for value, rows in counts.items():
            key = str(owners[value])
            total_rows, hosts = report[key]
            report[key] = (total_rows + int(rows), hosts + 1)
        return mask, report
