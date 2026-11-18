from starlette.requests import HTTPConnection

from .container import Container
from .services import (
    DatasetStore,
    FileStore,
    JobService,
    LogReader,
    MetricsStream,
    SystemMonitor,
)
from .settings import ApiSettings


def get_container(connection: HTTPConnection) -> Container:
    return connection.app.state.container


def require_token(connection: HTTPConnection) -> None:
    get_container(connection).guard.verify(connection)


def get_settings(connection: HTTPConnection) -> ApiSettings:
    return get_container(connection).settings


def get_monitor(connection: HTTPConnection) -> SystemMonitor:
    return get_container(connection).monitor


def get_metrics_stream(connection: HTTPConnection) -> MetricsStream:
    return get_container(connection).metrics_stream


def get_file_store(connection: HTTPConnection) -> FileStore:
    return get_container(connection).files


def get_dataset_store(connection: HTTPConnection) -> DatasetStore:
    return get_container(connection).datasets


def get_log_reader(connection: HTTPConnection) -> LogReader:
    return get_container(connection).logs


def get_job_service(connection: HTTPConnection) -> JobService:
    return get_container(connection).jobs
