from typing import List

from pydantic import BaseModel


class DatasetFile(BaseModel):
    name: str
    size_bytes: int
    modified_at: float


class DatasetInventory(BaseModel):
    files: List[DatasetFile]
    total_bytes: int
