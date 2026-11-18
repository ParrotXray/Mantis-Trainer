from pathlib import Path
from typing import Final, List, Optional

from model.error import PipelineError
from utils import IpFilter

from .args_parser import SECTIONS, ArgsParser
from .config_builder import ConfigBuilder
from .csv_pool import CsvPool
from .options import PipelineOptions
from .request import PipelineRequest
from .stage import MODEL_STAGES, Stage, StagePlanner

IP_EXCLUSION_KEYS: Final[tuple[str, str]] = ("exclude_src_ips", "exclude_dst_ips")

ARTIFACTS_DIR: Final[str] = "./artifacts"


class PipelinePlanner:
    def __init__(
        self,
        pool: CsvPool,
        parser: ArgsParser,
    ) -> None:
        self.pool = pool
        self.parser = parser

    def _accept_path(self, value: str, _: str) -> str:
        return value

    def _data_files(self, stages: List[Stage]) -> List[str]:
        if Stage.DP not in stages:
            return []
        files = self.pool.names()
        if not files:
            raise PipelineError(
                f"Stage 'dp' requires CSV files in {self.pool.root}, import them first"
            )
        return files

    def _locate(self, value: str, label: str, is_file: bool) -> Optional[str]:
        bare = "/" not in value and "\\" not in value
        candidates = [f"{ARTIFACTS_DIR}/{value}", value] if bare else [value]
        for candidate in candidates:
            checked = self._accept_path(candidate, label)
            path = Path(checked)
            if path.is_file() if is_file else path.is_dir():
                return checked
        return None

    def _model_dir(self, stages: List[Stage], value: Optional[str]) -> str:
        directory = value or ARTIFACTS_DIR
        if value and not MODEL_STAGES.intersection(stages):
            raise PipelineError("'run.model_dir' only applies to stages 'ft' and 'sc'")
        if MODEL_STAGES.intersection(stages):
            found = self._locate(directory, "run.model_dir", is_file=False)
            if found is None:
                raise PipelineError(
                    f"Model directory '{directory}' not found (a bare name is looked up "
                    f"in {ARTIFACTS_DIR} first)"
                )
            return found
        return directory

    def _resume(self, stages: List[Stage], value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        if Stage.DA not in stages:
            raise PipelineError("'run.resume' only applies to stage 'da'")
        found = self._locate(value, "run.resume", is_file=True)
        if found is None:
            raise PipelineError(
                f"Checkpoint '{value}' not found (a bare file name is looked up in "
                f"{ARTIFACTS_DIR} first)"
            )
        return found

    def plan(self, request: PipelineRequest) -> PipelineOptions:
        stages = StagePlanner.resolve(request.stages)
        args = self.parser.parse(request.args)
        configs = {
            section: ConfigBuilder.build(config_cls, args.overrides[section])
            for section, config_cls in SECTIONS.items()
        }
        for name in IP_EXCLUSION_KEYS:
            try:
                IpFilter.parse(getattr(configs["dp"], name))
            except ValueError as exc:
                raise PipelineError(f"Invalid dp.{name}: {exc}") from exc
        return PipelineOptions(
            stages=stages,
            data_root=str(self.pool.root),
            data_files=self._data_files(stages),
            model_dir=self._model_dir(stages, args.run.get("model_dir")),
            resume=self._resume(stages, args.run.get("resume")),
            preprocess=configs["dp"],
            autoencoder=configs["da"],
            export=configs["ex"],
        )
