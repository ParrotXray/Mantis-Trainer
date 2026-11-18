from dataclasses import dataclass

from pipeline import PipelineOptions


@dataclass(frozen=True)
class JobSpec:
    job_id: str
    workspace: str
    log_file: str
    progress_file: str
    options: PipelineOptions
