import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

from pipeline import Stage

StageTimes = Dict[str, Dict[str, Optional[float]]]


class StageProgressFile:
    SUFFIX = ".stages.json"

    def __init__(self, path: str) -> None:
        self.path = Path(path)

    @classmethod
    def for_job(cls, directory: Path, job_id: str) -> "StageProgressFile":
        return cls(str(directory / f"{job_id}{cls.SUFFIX}"))

    def read(self) -> StageTimes:
        try:
            data = json.loads(self.path.read_text("utf-8"))
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def write(self, times: StageTimes) -> None:
        temp = self.path.with_suffix(".tmp")
        temp.write_text(json.dumps(times), encoding="utf-8")
        os.replace(temp, self.path)

    def delete(self) -> None:
        self.path.unlink(missing_ok=True)


class StageProgressWriter:
    def __init__(self, path: str, stages: List[Stage]) -> None:
        self.file = StageProgressFile(path)
        self.times: StageTimes = {
            stage.value: {"started_at": None, "finished_at": None} for stage in stages
        }
        self.file.write(self.times)

    def started(self, stage: Stage) -> None:
        self.times[stage.value]["started_at"] = time.time()
        self.file.write(self.times)

    def finished(self, stage: Stage) -> None:
        self.times[stage.value]["finished_at"] = time.time()
        self.file.write(self.times)
