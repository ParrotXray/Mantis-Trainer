import signal
import time
import uuid
from pathlib import Path
from typing import List, Optional

from model.error import JobConflictError
from utils import Logger

from ..schemas import (
    JobPlan,
    JobRequest,
    JobResponse,
    JobStatus,
    StageProgress,
    StageStatus,
)
from ..worker import StageProgressFile
from .job_executor import JobExecutor
from .job_registry import JobRecord, JobRegistry
from .job_spec_factory import JobSpecFactory


class JobService:
    def __init__(
        self,
        node_id: str,
        registry: JobRegistry,
        executor: JobExecutor,
        spec_factory: JobSpecFactory,
    ) -> None:
        self.node_id = node_id
        self.registry = registry
        self.executor = executor
        self.spec_factory = spec_factory
        self.log = Logger(__name__)

    @staticmethod
    def _interrupted(record: JobRecord) -> StageStatus:
        if record.status == JobStatus.CANCELLED:
            return StageStatus.CANCELLED
        return StageStatus.FAILED

    def _stage_status(
        self, record: JobRecord, started: Optional[float], finished: Optional[float]
    ) -> StageStatus:
        if finished is not None or record.status == JobStatus.SUCCEEDED:
            return StageStatus.SUCCEEDED
        if started is not None:
            return (
                StageStatus.RUNNING if record.is_active else self._interrupted(record)
            )
        if record.is_active:
            return StageStatus.PENDING
        return StageStatus.SKIPPED

    def _progress(self, record: JobRecord) -> List[StageProgress]:
        times = StageProgressFile.for_job(
            Path(record.log_file).parent, record.job_id
        ).read()
        progress = []
        for stage in record.stages:
            entry = times.get(stage.value) or {}
            started = entry.get("started_at")
            finished = entry.get("finished_at")
            progress.append(
                StageProgress(
                    stage=stage,
                    status=self._stage_status(record, started, finished),
                    started_at=started,
                    finished_at=finished,
                )
            )
        return progress

    def _to_response(self, record: JobRecord) -> JobResponse:
        progress = self._progress(record)
        current = next(
            (p.stage for p in progress if p.status == StageStatus.RUNNING), None
        )
        return JobResponse(
            job_id=record.job_id,
            node_id=self.node_id,
            status=record.status,
            cancel_requested=record.cancel_requested,
            stages=record.stages,
            stage_progress=progress,
            current_stage=current,
            datasets=record.datasets,
            args=record.args,
            created_at=record.created_at,
            started_at=record.started_at,
            finished_at=record.finished_at,
            exit_code=record.exit_code,
        )

    def check(self, request: JobRequest) -> JobPlan:
        return self.spec_factory.plan(request)

    def submit(self, request: JobRequest) -> JobResponse:
        job_id = uuid.uuid4().hex
        spec = self.spec_factory.create(job_id, request)
        stages = spec.options.stages
        with self.registry.lock:
            active = self.registry.active()
            if active is not None:
                raise JobConflictError(
                    f"Node '{self.node_id}' is busy with job '{active.job_id}'"
                )
            record = JobRecord(
                job_id=job_id,
                stages=stages,
                datasets=spec.options.data_files,
                args=request.args,
                log_file=spec.log_file,
            )
            self.registry.add(record)
            try:
                Path(spec.log_file).parent.mkdir(parents=True, exist_ok=True)
                Path(spec.log_file).touch()
                self.executor.start(spec, self._on_exit)
            except Exception:
                record.status = JobStatus.FAILED
                record.finished_at = time.time()
                self.registry.save(record)
                raise
            record.status = JobStatus.RUNNING
            record.started_at = time.time()
            self.registry.save(record)
        self.log.info(f"Job {job_id} submitted: {[s.value for s in stages]}")
        return self._to_response(record)

    def _on_exit(self, job_id: str, exit_code: Optional[int]) -> None:
        with self.registry.lock:
            record = self.registry.get(job_id)
            record.exit_code = exit_code
            record.finished_at = time.time()
            if record.cancel_requested:
                record.status = JobStatus.CANCELLED
            else:
                record.status = (
                    JobStatus.SUCCEEDED if exit_code == 0 else JobStatus.FAILED
                )
            self.registry.save(record)
        self.log.info(f"Job {job_id} exited with code {exit_code}")
        if exit_code == -signal.SIGKILL and not record.cancel_requested:
            self.log.warning(
                f"Job {job_id} was killed by SIGKILL without a cancel request; "
                "the node most likely ran out of memory (check `dmesg | grep -i oom` "
                "or the container's memory limit) and the OOM killer stopped it"
            )

    def get(self, job_id: str) -> JobResponse:
        return self._to_response(self.registry.get(job_id))

    def list(self) -> List[JobResponse]:
        return [self._to_response(record) for record in self.registry.list()]

    def cancel(self, job_id: str) -> JobResponse:
        with self.registry.lock:
            record = self.registry.get(job_id)
            if not record.is_active:
                raise JobConflictError(
                    f"Job '{job_id}' is already {record.status.value}"
                )
            record.cancel_requested = True
            self.registry.save(record)
        self.log.info(f"Job {job_id} cancel requested")
        self.executor.terminate(job_id)
        return self._to_response(record)

    def current(self) -> Optional[JobResponse]:
        record = self.registry.active()
        return self._to_response(record) if record else None

    def ensure_datasets_idle(self) -> None:
        active = self.registry.active()
        if active is not None and active.datasets:
            raise JobConflictError(f"Datasets are in use by job '{active.job_id}'")

    def is_active(self, job_id: str) -> bool:
        return self.registry.get(job_id).is_active

    def log_name(self, job_id: str) -> str:
        return f"jobs/{Path(self.registry.get(job_id).log_file).name}"

    def shutdown(self) -> None:
        self.executor.shutdown()
