from typing import Protocol

from .stage import Stage


class StageListener(Protocol):
    def started(self, stage: Stage) -> None: ...

    def finished(self, stage: Stage) -> None: ...


class NullStageListener:
    def started(self, stage: Stage) -> None:
        return None

    def finished(self, stage: Stage) -> None:
        return None
