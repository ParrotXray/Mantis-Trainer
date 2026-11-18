from dataclasses import dataclass

from pipeline import CsvPool

from .security import TokenGuard
from .services import (
    CpuMonitor,
    DatasetStore,
    FileStore,
    GpuMonitor,
    JobExecutor,
    JobRegistry,
    JobService,
    JobSpecFactory,
    JobStore,
    LogReader,
    MemoryMonitor,
    MetricsStream,
    SystemMonitor,
)
from .settings import ApiSettings


@dataclass(frozen=True)
class Container:
    settings: ApiSettings
    guard: TokenGuard
    monitor: SystemMonitor
    metrics_stream: MetricsStream
    files: FileStore
    datasets: DatasetStore
    logs: LogReader
    jobs: JobService

    @classmethod
    def build(cls, settings: ApiSettings) -> "Container":
        pool = CsvPool(settings.dataset_root)
        datasets = DatasetStore(pool, settings.max_upload_mb * 1024**2)
        monitor = SystemMonitor(
            node_id=settings.node_id,
            cpu=CpuMonitor(settings.cpu_sample_interval),
            memory=MemoryMonitor(),
            gpu=GpuMonitor(),
        )
        jobs = JobService(
            node_id=settings.node_id,
            registry=JobRegistry(JobStore(settings.job_log_dir)),
            executor=JobExecutor(),
            spec_factory=JobSpecFactory(settings, pool),
        )
        return cls(
            settings=settings,
            guard=TokenGuard(settings.api_token),
            monitor=monitor,
            metrics_stream=MetricsStream(
                monitor, settings.ws_min_interval, settings.ws_max_interval
            ),
            files=FileStore(settings.workspace),
            datasets=datasets,
            logs=LogReader(settings.log_dir),
            jobs=jobs,
        )
