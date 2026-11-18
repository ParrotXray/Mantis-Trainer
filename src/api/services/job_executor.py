import multiprocessing
import threading
from typing import Callable, Dict, List, Optional

import psutil

from ..worker import JobSpec, run_job

ExitCallback = Callable[[str, Optional[int]], None]


class JobExecutor:
    def __init__(self, terminate_timeout: float = 10.0) -> None:
        self._context = multiprocessing.get_context("spawn")
        self._processes: Dict[str, multiprocessing.Process] = {}
        self._lock = threading.Lock()
        self.terminate_timeout = terminate_timeout

    def start(self, spec: JobSpec, on_exit: ExitCallback) -> None:
        process = self._context.Process(
            target=run_job, args=(spec,), name=f"job-{spec.job_id}"
        )
        process.start()
        with self._lock:
            self._processes[spec.job_id] = process
        threading.Thread(
            target=self._wait, args=(spec.job_id, process, on_exit), daemon=True
        ).start()

    def _wait(
        self, job_id: str, process: multiprocessing.Process, on_exit: ExitCallback
    ) -> None:
        process.join()
        with self._lock:
            self._processes.pop(job_id, None)
        on_exit(job_id, process.exitcode)

    def terminate(self, job_id: str) -> bool:
        with self._lock:
            process = self._processes.get(job_id)
        if process is None or not process.is_alive():
            return False
        descendants = self._descendants(process.pid)
        process.terminate()
        self._signal(descendants, psutil.Process.terminate)
        process.join(self.terminate_timeout)
        _, alive = psutil.wait_procs(descendants, timeout=self.terminate_timeout)
        if process.is_alive():
            process.kill()
            process.join()
        self._signal(alive, psutil.Process.kill)
        return True

    @staticmethod
    def _descendants(pid: int) -> List[psutil.Process]:
        try:
            return psutil.Process(pid).children(recursive=True)
        except psutil.NoSuchProcess:
            return []

    @staticmethod
    def _signal(processes: List[psutil.Process], action) -> None:
        for proc in processes:
            try:
                action(proc)
            except psutil.NoSuchProcess:
                pass

    def shutdown(self) -> None:
        with self._lock:
            job_ids = list(self._processes)
        for job_id in job_ids:
            self.terminate(job_id)
