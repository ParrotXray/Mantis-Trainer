import dataclasses
import threading
import time
from typing import Dict, Final, Optional, Tuple

from model import ReportError, VerdictReportConfig
from model.error import InvalidParameterError, ResourceNotFoundError, StorageError
from pipeline import CsvPool
from report import VerdictReport

from ..schemas import VerdictReportResponse

_CACHE_SIZE: Final[int] = 8

_Key = Tuple[Tuple[Tuple[str, int, float], ...], Tuple[str, ...], int, int]


class VerdictReportService:
    """Builds VerdictReport over the node's CSV pool.

    Reading a large pool takes a while, so results are cached per pool
    content (name, size, mtime of every file) and parameters, and one lock
    keeps concurrent requests from reading the pool twice.
    """

    def __init__(self, pool: CsvPool) -> None:
        self.pool = pool
        self._lock = threading.Lock()
        self._cache: Dict[_Key, VerdictReportResponse] = {}

    def _signature(self) -> Tuple[Tuple[str, int, float], ...]:
        entries = []
        for path in self.pool.files():
            stat = path.stat()
            entries.append(
                (str(path.relative_to(self.pool.root)), stat.st_size, stat.st_mtime)
            )
        return tuple(entries)

    def build(
        self, config: VerdictReportConfig, refresh: bool = False
    ) -> VerdictReportResponse:
        try:
            report = VerdictReport(config)
        except ReportError as exc:
            raise InvalidParameterError(str(exc)) from exc
        with self._lock:
            signature = self._signature()
            if not signature:
                raise ResourceNotFoundError("The node has no CSV files to report on")
            key: _Key = (
                signature,
                tuple(config.trigger_verdicts),
                config.window_secs,
                config.top,
            )
            cached: Optional[VerdictReportResponse] = self._cache.get(key)
            if cached is not None and not refresh:
                return cached.model_copy(update={"cached": True})
            try:
                result = report.build(self.pool.files(), self.pool.root)
            except (OSError, ValueError) as exc:
                raise StorageError(f"Cannot read the CSV pool: {exc}") from exc
            response = VerdictReportResponse(
                **dataclasses.asdict(result), generated_at=time.time(), cached=False
            )
            if len(self._cache) >= _CACHE_SIZE:
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = response
            return response
