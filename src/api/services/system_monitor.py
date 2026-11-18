import time

from ..schemas import SystemMetrics
from .cpu_monitor import CpuMonitor
from .gpu_monitor import GpuMonitor
from .memory_monitor import MemoryMonitor


class SystemMonitor:
    def __init__(
        self,
        node_id: str,
        cpu: CpuMonitor,
        memory: MemoryMonitor,
        gpu: GpuMonitor,
    ) -> None:
        self.node_id = node_id
        self.cpu = cpu
        self.memory = memory
        self.gpu = gpu

    def collect(self) -> SystemMetrics:
        return SystemMetrics(
            node_id=self.node_id,
            timestamp=time.time(),
            cpu=self.cpu.collect(),
            memory=self.memory.collect(),
            gpu=self.gpu.collect(),
        )
