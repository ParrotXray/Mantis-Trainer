from typing import List

from fastapi import APIRouter, Depends, File, Query, UploadFile, status

from ..dependencies import get_dataset_store, get_job_service
from ..schemas import DatasetInventory
from ..services import DatasetStore, JobService

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.get("", response_model=DatasetInventory)
def list_datasets(store: DatasetStore = Depends(get_dataset_store)) -> DatasetInventory:
    return store.inventory()


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
