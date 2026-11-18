from enum import Enum
from typing import Final, Iterable, List

from model.error import PipelineError


class Stage(str, Enum):
    DP = "dp"
    DA = "da"
    FT = "ft"
    SC = "sc"
    FA = "fa"
    EX = "ex"
    ALL = "all"


PIPELINE_ORDER: Final[List[Stage]] = [
    Stage.DP,
    Stage.DA,
    Stage.FT,
    Stage.SC,
    Stage.FA,
    Stage.EX,
]

ALL_STAGES: Final[List[Stage]] = [Stage.DP, Stage.DA, Stage.EX]

MODEL_STAGES: Final[frozenset] = frozenset({Stage.FT, Stage.SC})


class StagePlanner:
    @staticmethod
    def resolve(stages: Iterable[Stage]) -> List[Stage]:
        selected = set(stages)
        if Stage.ALL in selected:
            selected = (selected - {Stage.ALL}) | set(ALL_STAGES)
        ordered = [stage for stage in PIPELINE_ORDER if stage in selected]
        StagePlanner._validate(ordered)
        return ordered

    @staticmethod
    def _validate(stages: List[Stage]) -> None:
        if not stages:
            raise PipelineError("No stage selected")
        if Stage.FT in stages and Stage.DA in stages:
            raise PipelineError("Stage 'ft' replaces 'da' training; use one of them")
        if Stage.SC in stages and Stage.DP in stages:
            raise PipelineError(
                "Stage 'sc' reads the preprocessed parquet; run 'dp' separately first"
            )
