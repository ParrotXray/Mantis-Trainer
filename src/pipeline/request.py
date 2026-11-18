from dataclasses import dataclass
from typing import List

from .stage import Stage


@dataclass(frozen=True)
class PipelineRequest:
    stages: List[Stage]
    args: str = ""
