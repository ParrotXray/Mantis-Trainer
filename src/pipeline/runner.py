import time
from datetime import timedelta
from typing import Optional

from utils import Logger, pipeline_stage

from .options import PipelineOptions
from .stage_listener import NullStageListener, StageListener
from .stages import STAGES


class PipelineRunner:
    def __init__(self, listener: Optional[StageListener] = None) -> None:
        self.log = Logger(__name__)
        self.listener = listener or NullStageListener()

    def run(self, options: PipelineOptions) -> None:
        names = ", ".join(stage.value for stage in options.stages)
        self.log.info(f"Pipeline stages: {names}")
        if options.data_files:
            self.log.info(f"CSV files: {', '.join(options.data_files)}")
        total_start = time.perf_counter()
        for key in options.stages:
            stage = STAGES[key]
            start = time.perf_counter()
            self.listener.started(key)
            with pipeline_stage(stage.title, delay=stage.delay):
                stage.run(options)
            self.listener.finished(key)
            elapsed = timedelta(seconds=time.perf_counter() - start)
            self.log.info(f"{stage.title} execution time: {elapsed}")
        total = timedelta(seconds=time.perf_counter() - total_start)
        self.log.info(f"Execution time: {total}")
