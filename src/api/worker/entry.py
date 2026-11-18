import os
import sys

from utils import Logger

from .job_logging import configure_job_logging
from .spec import JobSpec
from .stage_progress import StageProgressWriter


def run_job(spec: JobSpec) -> None:
    os.chdir(spec.workspace)
    configure_job_logging(spec.log_file)
    log = Logger(__name__)
    log.info(f"Job {spec.job_id} started")
    try:
        from pipeline.runner import PipelineRunner

        progress = StageProgressWriter(spec.progress_file, spec.options.stages)
        PipelineRunner(progress).run(spec.options)
    except Exception as exc:
        log.error(f"Job {spec.job_id} failed: {exc!r}")
        log.logger.exception(exc)
        sys.exit(1)
    log.info(f"Job {spec.job_id} finished")
