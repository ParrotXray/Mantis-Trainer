import dataclasses
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from pipeline import Stage

from ..schemas import JobStatus
from ..worker import StageProgressFile
from .job_registry import JobRecord

_SUFFIX = ".job.json"


class JobStore:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, job_id: str) -> Path:
        return self.directory / f"{job_id}{_SUFFIX}"

    @staticmethod
    def _encode(record: JobRecord) -> Dict[str, Any]:
        data = dataclasses.asdict(record)
        data["stages"] = [stage.value for stage in record.stages]
        data["status"] = record.status.value
        return data

    @staticmethod
    def _decode(data: Dict[str, Any]) -> JobRecord:
        data["stages"] = [Stage(stage) for stage in data["stages"]]
        data["status"] = JobStatus(data["status"])
        return JobRecord(**data)

    def save(self, record: JobRecord) -> None:
        target = self._path(record.job_id)
        temp = target.with_suffix(".tmp")
        temp.write_text(json.dumps(self._encode(record)), encoding="utf-8")
        os.replace(temp, target)

    def delete(self, job_id: str) -> None:
        self._path(job_id).unlink(missing_ok=True)
        StageProgressFile.for_job(self.directory, job_id).delete()

    def load(self) -> List[JobRecord]:
        records = []
        for path in sorted(self.directory.glob(f"*{_SUFFIX}")):
            try:
                records.append(self._decode(json.loads(path.read_text("utf-8"))))
            except (ValueError, KeyError, TypeError):
                continue
        return records
