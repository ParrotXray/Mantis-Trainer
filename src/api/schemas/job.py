from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field

from pipeline import Stage


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


class StageProgress(BaseModel):
    stage: Stage
    status: StageStatus
    started_at: Optional[float]
    finished_at: Optional[float]


class JobRequest(BaseModel):
    stages: List[Stage] = Field(min_length=1)
    args: str = Field(default="", max_length=4096)


class JobPlan(BaseModel):
    stages: List[Stage]
    datasets: List[str]
    model_dir: Optional[str]
    resume: Optional[str]


class JobResponse(BaseModel):
    job_id: str
    node_id: str
    status: JobStatus
    cancel_requested: bool
    stages: List[Stage]
    stage_progress: List[StageProgress]
    current_stage: Optional[Stage]
    datasets: List[str]
    args: str
    created_at: float
    started_at: Optional[float]
    finished_at: Optional[float]
    exit_code: Optional[int]
