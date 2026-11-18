import os

import psutil

from ..schemas import CpuMetrics


class CpuMonitor:
    def __init__(self, sample_interval: float) -> None:
        self.sample_interval = sample_interval

    def collect(self) -> CpuMetrics:
        per_core = psutil.cpu_percent(interval=self.sample_interval, percpu=True)
        frequency = psutil.cpu_freq()
        load = list(os.getloadavg()) if hasattr(os, "getloadavg") else None
        return CpuMetrics(
            physical_cores=psutil.cpu_count(logical=False),
            logical_cores=psutil.cpu_count(logical=True) or len(per_core),
            usage_percent=round(sum(per_core) / max(len(per_core), 1), 2),
            per_core_percent=per_core,
            frequency_mhz=frequency.current if frequency else None,
            load_average=load,
        )
