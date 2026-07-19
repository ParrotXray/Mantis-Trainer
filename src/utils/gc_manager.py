import gc
import os
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional

import psutil

from .logger import Logger

log = Logger(__name__)


@dataclass
class MemorySnapshot:
    rss_mb: float
    vms_mb: float
    percent: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class StageMemoryReport:
    stage_name: str
    before: MemorySnapshot
    after: MemorySnapshot
    peak_mb: float
    duration_sec: float

    @property
    def delta_mb(self) -> float:
        return self.after.rss_mb - self.before.rss_mb

    @property
    def was_leaked(self) -> bool:
        return self.delta_mb > 0

    def summary(self) -> str:
        sign = "+" if self.delta_mb >= 0 else ""
        return (
            f"[{self.stage_name}] "
            f"RSS: {self.before.rss_mb:.1f} -> {self.after.rss_mb:.1f} MB "
            f"({sign}{self.delta_mb:.1f} MB), "
            f"Peak: {self.peak_mb:.1f} MB, "
            f"Duration: {self.duration_sec:.2f}s"
        )


def _get_memory_snapshot() -> Optional[MemorySnapshot]:
    proc = psutil.Process(os.getpid())
    info = proc.memory_info()
    return MemorySnapshot(
        rss_mb=info.rss / 1024**2,
        vms_mb=info.vms / 1024**2,
        percent=proc.memory_percent(),
    )


def _get_peak_mb(before_rss: float) -> float:
    try:
        if tracemalloc.is_tracing():
            _, peak = tracemalloc.get_traced_memory()
            return peak / 1024**2
    except Exception:
        pass
    snap = _get_memory_snapshot()
    return snap.rss_mb if snap else before_rss


@contextmanager
def pipeline_stage(
    name: str,
    delay: float = 3.0,
    warn_delta_mb: float = 100.0,
    warn_rss_mb: float = 2048.0,
    enable_memory_report: bool = True,
    trace_alloc: bool = False,
):
    if trace_alloc:

        tracemalloc.start()

    before = _get_memory_snapshot()
    start_time = time.perf_counter()

    log.info(f"[{name}] Start" + (f" | RSS: {before.rss_mb:.1f} MB" if before else ""))

    try:
        yield
    except:
        log.warning(f"[{name}] An error occurred during execution...")
        raise
    finally:
        gc.collect()

        duration = time.perf_counter() - start_time
        after = _get_memory_snapshot()
        peak_mb = _get_peak_mb(before.rss_mb if before else 0.0)

        if trace_alloc:
            tracemalloc.stop()

        if enable_memory_report and before and after:
            report = StageMemoryReport(
                stage_name=name,
                before=before,
                after=after,
                peak_mb=peak_mb,
                duration_sec=duration,
            )
            log.info(report.summary())

            if report.delta_mb > warn_delta_mb:
                log.warning(
                    f"[{name}] Memory not fully released!"
                    f" Increased by {report.delta_mb:.1f} MB (threshold: {warn_delta_mb} MB)"
                )

            if after.rss_mb > warn_rss_mb:
                log.warning(
                    f"[{name}] RSS ({after.rss_mb:.1f} MB) exceeded warning threshold {warn_rss_mb} MB"
                )
        else:
            log.info(f"[{name}] Done | Duration: {duration:.2f}s")

        if delay > 0:
            time.sleep(delay)
