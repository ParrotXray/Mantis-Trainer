from typing import List, Optional

from pydantic import BaseModel

from .job import JobResponse
from .metrics import SystemMetrics


class NodeInfo(BaseModel):
    node_id: str
    hostname: str
    busy: bool
    current_job: Optional[str]
    stages: List[str]


class HealthStatus(BaseModel):
    status: str
    node_id: str


class NodeSnapshot(BaseModel):
    node: NodeInfo
    metrics: SystemMetrics
    jobs: Optional[List[JobResponse]]
