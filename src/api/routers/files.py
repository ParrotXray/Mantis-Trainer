import os
from typing import List

from fastapi import APIRouter, Depends, Query
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from ..dependencies import get_file_store, get_settings
from ..schemas import FileCategory, FileInfo
from ..services import FileStore
from ..settings import ApiSettings

router = APIRouter(prefix="/files", tags=["files"])


@router.get("/{category}", response_model=List[FileInfo])
def list_files(
    category: FileCategory, store: FileStore = Depends(get_file_store)
) -> List[FileInfo]:
    return store.list(category)


@router.get("/{category}/download")
def download_file(
    category: FileCategory,
    path: str = Query(..., min_length=1),
    store: FileStore = Depends(get_file_store),
) -> FileResponse:
    target = store.file_of(category, path)
    return FileResponse(target, filename=target.name)


@router.get("/{category}/archive")
def download_archive(
    category: FileCategory,
    path: str = Query(""),
    store: FileStore = Depends(get_file_store),
    settings: ApiSettings = Depends(get_settings),
) -> FileResponse:
    archive = store.archive(category, path)
    suffix = f"-{path.strip('/').replace('/', '_')}" if path else ""
    return FileResponse(
        archive,
        media_type="application/zip",
        filename=f"{settings.node_id}-{category.value}{suffix}.zip",
        background=BackgroundTask(os.unlink, archive),
    )
