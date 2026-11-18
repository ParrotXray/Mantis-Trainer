import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol

from pipeline import Stage

from model.error import ResourceNotFoundError
from ..schemas import JobStatus


@dataclass
class JobRecord:
    job_id: str
    stages: List[Stage]
    datasets: List[str]
    args: str
    log_file: str
    status: JobStatus = JobStatus.PENDING
    cancel_requested: bool = False
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    exit_code: Optional[int] = None

    @property
    def is_active(self) -> bool:
        return self.status in (JobStatus.PENDING, JobStatus.RUNNING)


class JobStorage(Protocol):
    def save(self, record: JobRecord) -> None: ...

    def delete(self, job_id: str) -> None: ...

    def load(self) -> List[JobRecord]: ...


class JobRegistry:
    def __init__(self, storage: JobStorage, history_limit: int = 200) -> None:
        self._lock = threading.RLock()
        self._storage = storage
        self.history_limit = history_limit
        self._records: Dict[str, JobRecord] = {}
        for record in storage.load():
            if record.is_active:
                record.status = JobStatus.FAILED
                record.finished_at = record.finished_at or time.time()
                storage.save(record)
            self._records[record.job_id] = record
        self._prune()

    @property
    def lock(self) -> threading.RLock:
        return self._lock

    def add(self, record: JobRecord) -> None:
        with self._lock:
            self._records[record.job_id] = record
            self._storage.save(record)
            self._prune()

    def save(self, record: JobRecord) -> None:
        with self._lock:
            self._storage.save(record)

    def _prune(self) -> None:
        finished = sorted(
            (r for r in self._records.values() if not r.is_active),
            key=lambda r: r.created_at,
        )
        for record in finished[: max(len(self._records) - self.history_limit, 0)]:
            del self._records[record.job_id]
            self._storage.delete(record.job_id)

    def get(self, job_id: str) -> JobRecord:
        with self._lock:
            if job_id not in self._records:
                raise ResourceNotFoundError(f"Job '{job_id}' not found")
            return self._records[job_id]

    def list(self) -> List[JobRecord]:
        with self._lock:
            return sorted(self._records.values(), key=lambda r: r.created_at)

    def active(self) -> Optional[JobRecord]:
        with self._lock:
            return next((r for r in self._records.values() if r.is_active), None)
