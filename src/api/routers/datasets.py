from typing import List

from fastapi import APIRouter, Depends, File, Query, UploadFile, status

from model import VERDICT_REPORT_TOP, VERDICT_REPORT_WINDOW_SECS, VerdictReportConfig

from ..dependencies import (
    get_dataset_store,
    get_job_service,
    get_verdict_report_service,
)
from ..schemas import DatasetInventory, VerdictReportResponse
from ..services import DatasetStore, JobService, VerdictReportService

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.get("", response_model=DatasetInventory)
def list_datasets(store: DatasetStore = Depends(get_dataset_store)) -> DatasetInventory:
    return store.inventory()


@router.get("/verdicts", response_model=VerdictReportResponse)
def verdict_report(
    trigger_verdicts: str = Query(
        ",".join(VerdictReportConfig.trigger_verdicts),
        description="Comma-separated ML Verdict values that flag a source",
    ),
    window_secs: int = Query(
        VerdictReportConfig.window_secs,
        ge=VERDICT_REPORT_WINDOW_SECS[0],
        le=VERDICT_REPORT_WINDOW_SECS[1],
    ),
    top: int = Query(
        VerdictReportConfig.top, ge=VERDICT_REPORT_TOP[0], le=VERDICT_REPORT_TOP[1]
    ),
    refresh: bool = Query(False),
    service: VerdictReportService = Depends(get_verdict_report_service),
) -> VerdictReportResponse:
    config = VerdictReportConfig(
        trigger_verdicts=tuple(
            v.strip() for v in trigger_verdicts.split(",") if v.strip()
        ),
        window_secs=window_secs,
        top=top,
    )
    return service.build(config, refresh)


@router.post("", response_model=DatasetInventory, status_code=status.HTTP_201_CREATED)
async def import_csv(
    files: List[UploadFile] = File(...),
    replace: bool = Query(False),
    store: DatasetStore = Depends(get_dataset_store),
    jobs: JobService = Depends(get_job_service),
) -> DatasetInventory:
    jobs.ensure_datasets_idle()
    return await store.import_csv(files, replace)


@router.delete("", status_code=status.HTTP_204_NO_CONTENT)
def clear_datasets(
    store: DatasetStore = Depends(get_dataset_store),
    jobs: JobService = Depends(get_job_service),
) -> None:
    jobs.ensure_datasets_idle()
    store.clear()


@router.delete("/{path:path}", status_code=status.HTTP_204_NO_CONTENT)
def delete_csv(
    path: str,
    store: DatasetStore = Depends(get_dataset_store),
    jobs: JobService = Depends(get_job_service),
) -> None:
    jobs.ensure_datasets_idle()
    store.delete(path)
