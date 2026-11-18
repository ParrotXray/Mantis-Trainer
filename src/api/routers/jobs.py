from typing import Any, Dict, List

from fastapi import APIRouter, Depends, Query, status
from fastapi.responses import StreamingResponse

from pipeline import ArgsParser

from ..dependencies import get_job_service, get_log_reader
from ..schemas import JobPlan, JobRequest, JobResponse, LogContent
from ..services import JobService, LogReader

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("/params", response_model=Dict[str, Dict[str, Any]])
def read_default_params() -> Dict[str, Dict[str, Any]]:
    return ArgsParser.defaults()


@router.post("/check", response_model=JobPlan)
def check_job(
    request: JobRequest, jobs: JobService = Depends(get_job_service)
) -> JobPlan:
    return jobs.check(request)


@router.post("", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED)
def submit_job(
    request: JobRequest, jobs: JobService = Depends(get_job_service)
) -> JobResponse:
    return jobs.submit(request)


@router.get("", response_model=List[JobResponse])
def list_jobs(jobs: JobService = Depends(get_job_service)) -> List[JobResponse]:
    return jobs.list()


@router.get("/{job_id}", response_model=JobResponse)
def read_job(job_id: str, jobs: JobService = Depends(get_job_service)) -> JobResponse:
    return jobs.get(job_id)


@router.delete("/{job_id}", response_model=JobResponse)
def cancel_job(job_id: str, jobs: JobService = Depends(get_job_service)) -> JobResponse:
    return jobs.cancel(job_id)


@router.get("/{job_id}/logs", response_model=LogContent)
def read_job_logs(
    job_id: str,
    lines: int = Query(200, ge=1, le=10000),
    jobs: JobService = Depends(get_job_service),
    reader: LogReader = Depends(get_log_reader),
) -> LogContent:
    return reader.tail(jobs.log_name(job_id), lines)


@router.get("/{job_id}/logs/stream")
def stream_job_logs(
    job_id: str,
    lines: int = Query(50, ge=0, le=10000),
    jobs: JobService = Depends(get_job_service),
    reader: LogReader = Depends(get_log_reader),
) -> StreamingResponse:
    name = jobs.log_name(job_id)
    events = reader.follow(name, lines, lambda: jobs.is_active(job_id))

    async def _sse():
        async for line in events:
            yield f"data: {line}\n\n"
        yield "event: end\ndata: \n\n"

    return StreamingResponse(
        _sse(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
