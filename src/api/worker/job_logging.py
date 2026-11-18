import logging
from pathlib import Path

from utils import Logger

_FORMAT = (
    "%(asctime)s.%(msecs)03d  %(levelname)-5s ThreadId(%(thread)d) "
    "%(name)s %(filename)s:%(lineno)d: %(message)s"
)


def configure_job_logging(log_file: str) -> None:
    Logger.setup_logging()
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter(fmt=_FORMAT, datefmt="%Y-%m-%d %H:%M:%S"))
    logging.root.addHandler(handler)
