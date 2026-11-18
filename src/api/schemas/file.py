from enum import Enum

from pydantic import BaseModel


class FileCategory(str, Enum):
    EXPORTS = "exports"
    ARTIFACTS = "artifacts"
    PLOTS = "plots"
    OUTPUTS = "outputs"
    LOGS = "logs"


class FileInfo(BaseModel):
    name: str
    size_bytes: int
    modified_at: float
