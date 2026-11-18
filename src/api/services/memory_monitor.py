import psutil

from ..schemas import MemoryMetrics

_MB = 1024**2


class MemoryMonitor:
    def collect(self) -> MemoryMetrics:
        virtual = psutil.virtual_memory()
        swap = psutil.swap_memory()
        return MemoryMetrics(
            total_mb=round(virtual.total / _MB, 2),
            available_mb=round(virtual.available / _MB, 2),
            used_mb=round(virtual.used / _MB, 2),
            usage_percent=virtual.percent,
            swap_total_mb=round(swap.total / _MB, 2),
            swap_used_mb=round(swap.used / _MB, 2),
            swap_percent=swap.percent,
        )
