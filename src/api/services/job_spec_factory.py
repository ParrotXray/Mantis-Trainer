from pipeline import (
    MODEL_STAGES,
    ArgsParser,
    CsvPool,
    PipelineOptions,
    PipelinePlanner,
    PipelineRequest,
    Stage,
)

from ..schemas import JobPlan, JobRequest
from ..settings import ApiSettings
from ..worker import JobSpec, StageProgressFile
from .path_guard import resolve_inside


class JobSpecFactory:
    def __init__(self, settings: ApiSettings, pool: CsvPool) -> None:
        self.settings = settings
        self.planner = PipelinePlanner(pool, ArgsParser(), self._inside_workspace)

    def _inside_workspace(self, value: str, label: str) -> str:
        return str(resolve_inside(self.settings.workspace, value, label))

    def _options(self, request: JobRequest) -> PipelineOptions:
        return self.planner.plan(
            PipelineRequest(stages=list(request.stages), args=request.args)
        )

    def plan(self, request: JobRequest) -> JobPlan:
        options = self._options(request)
        uses_model = bool(MODEL_STAGES.intersection(options.stages))
        return JobPlan(
            stages=options.stages,
            datasets=options.data_files,
            model_dir=options.model_dir if uses_model else None,
            resume=options.resume if Stage.DA in options.stages else None,
        )

    def create(self, job_id: str, request: JobRequest) -> JobSpec:
        return JobSpec(
            job_id=job_id,
            workspace=str(self.settings.workspace),
            log_file=str(self.settings.job_log_dir / f"{job_id}.log"),
            progress_file=str(
                StageProgressFile.for_job(self.settings.job_log_dir, job_id).path
            ),
            options=self._options(request),
        )
