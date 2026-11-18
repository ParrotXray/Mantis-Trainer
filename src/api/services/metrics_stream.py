import asyncio
from typing import AsyncIterator

from starlette.concurrency import run_in_threadpool

from model.error import InvalidParameterError
from ..schemas import SystemMetrics
from .system_monitor import SystemMonitor


class MetricsStream:
    def __init__(
        self, monitor: SystemMonitor, min_interval: float, max_interval: float
    ) -> None:
        self.monitor = monitor
        self.min_interval = min_interval
        self.max_interval = max_interval

    def validate_interval(self, interval: float) -> float:
        if not self.min_interval <= interval <= self.max_interval:
            raise InvalidParameterError(
                f"interval must be between {self.min_interval} and {self.max_interval}"
            )
        return interval

    async def stream(self, interval: float) -> AsyncIterator[SystemMetrics]:
        loop = asyncio.get_running_loop()
        while True:
            started = loop.time()
            yield await run_in_threadpool(self.monitor.collect)
            await asyncio.sleep(max(interval - (loop.time() - started), 0.0))
