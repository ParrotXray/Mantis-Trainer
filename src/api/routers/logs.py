from typing import List

from fastapi import APIRouter, Depends, Query

from ..dependencies import get_log_reader
from ..schemas import FileInfo, LogContent
from ..services import LogReader

router = APIRouter(prefix="/logs", tags=["logs"])


@router.get("", response_model=List[FileInfo])
def list_logs(reader: LogReader = Depends(get_log_reader)) -> List[FileInfo]:
    return reader.list()


@router.get("/{name:path}", response_model=LogContent)
def read_log(
    name: str,
    lines: int = Query(200, ge=1, le=10000),
    reader: LogReader = Depends(get_log_reader),
) -> LogContent:
    return reader.tail(name, lines)
