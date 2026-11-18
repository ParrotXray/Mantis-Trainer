import threading
from typing import List, Optional

from ..schemas import GpuDevice, GpuMetrics

try:
    import pynvml
except ImportError:
    pynvml = None

_MB = 1024**2


class GpuMonitor:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._initialized: Optional[bool] = None

    def _ensure_initialized(self) -> bool:
        with self._lock:
            if self._initialized is None:
                self._initialized = self._try_init()
            return self._initialized

    @staticmethod
    def _try_init() -> bool:
        if pynvml is None:
            return False
        try:
            pynvml.nvmlInit()
            return True
        except pynvml.NVMLError:
            return False

    @staticmethod
    def _optional(reader, *args) -> Optional[float]:
        try:
            return float(reader(*args))
        except pynvml.NVMLError:
            return None

    def _read_device(self, index: int) -> GpuDevice:
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        utilization = self._optional(
            lambda h: pynvml.nvmlDeviceGetUtilizationRates(h).gpu, handle
        )
        power_mw = self._optional(pynvml.nvmlDeviceGetPowerUsage, handle)
        name = pynvml.nvmlDeviceGetName(handle)
        return GpuDevice(
            index=index,
            name=name.decode() if isinstance(name, bytes) else name,
            utilization_percent=utilization,
            memory_total_mb=round(memory.total / _MB, 2),
            memory_used_mb=round(memory.used / _MB, 2),
            memory_percent=(
                round(memory.used / memory.total * 100, 2) if memory.total else 0.0
            ),
            temperature_c=self._optional(
                pynvml.nvmlDeviceGetTemperature, handle, pynvml.NVML_TEMPERATURE_GPU
            ),
            power_w=round(power_mw / 1000, 2) if power_mw is not None else None,
        )

    def collect(self) -> GpuMetrics:
        if not self._ensure_initialized():
            return GpuMetrics(available=False, driver_version=None, devices=[])
        try:
            driver = pynvml.nvmlSystemGetDriverVersion()
            count = pynvml.nvmlDeviceGetCount()
        except pynvml.NVMLError:
            return GpuMetrics(available=False, driver_version=None, devices=[])
        devices: List[GpuDevice] = []
        for index in range(count):
            try:
                devices.append(self._read_device(index))
            except pynvml.NVMLError:
                continue
        return GpuMetrics(
            available=True,
            driver_version=driver.decode() if isinstance(driver, bytes) else driver,
            devices=devices,
        )
