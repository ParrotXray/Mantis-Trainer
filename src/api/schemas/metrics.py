from typing import List, Optional

from pydantic import BaseModel


class CpuMetrics(BaseModel):
    physical_cores: Optional[int]
    logical_cores: int
    usage_percent: float
    per_core_percent: List[float]
    frequency_mhz: Optional[float]
    load_average: Optional[List[float]]


class MemoryMetrics(BaseModel):
    total_mb: float
    available_mb: float
    used_mb: float
    usage_percent: float
    swap_total_mb: float
    swap_used_mb: float
    swap_percent: float


class GpuDevice(BaseModel):
    index: int
    name: str
    utilization_percent: Optional[float]
    memory_total_mb: float
    memory_used_mb: float
    memory_percent: float
    temperature_c: Optional[float]
    power_w: Optional[float]


class GpuMetrics(BaseModel):
    available: bool
    driver_version: Optional[str]
    devices: List[GpuDevice]


class SystemMetrics(BaseModel):
    node_id: str
    timestamp: float
    cpu: CpuMetrics
    memory: MemoryMetrics
    gpu: GpuMetrics
