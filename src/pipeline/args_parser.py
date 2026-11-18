import dataclasses
import shlex
from dataclasses import dataclass
from typing import Dict, Final, List, Tuple

from model import DeepAutoencoderConfig, ExportConfig, PreprocessConfig
from model.error import PipelineError

SECTIONS: Final[Dict[str, type]] = {
    "dp": PreprocessConfig,
    "da": DeepAutoencoderConfig,
    "ex": ExportConfig,
}

RUN_SECTION: Final[str] = "run"

RUN_KEYS: Final[Tuple[str, ...]] = ("resume", "model_dir")


@dataclass(frozen=True)
class PipelineArgs:
    overrides: Dict[str, Dict[str, str]]
    run: Dict[str, str]


class ArgsParser:
    def __init__(self) -> None:
        self._fields: Dict[str, Tuple[str, ...]] = {
            section: tuple(item.name for item in dataclasses.fields(config_cls))
            for section, config_cls in SECTIONS.items()
        }
        self._fields[RUN_SECTION] = RUN_KEYS

    @staticmethod
    def _split(text: str) -> List[str]:
        try:
            return shlex.split(text)
        except ValueError as exc:
            raise PipelineError(f"Invalid args: {exc}") from exc

    def _suggest(self, name: str) -> str:
        owners = [section for section, names in self._fields.items() if name in names]
        if not owners:
            return f"Unknown parameter '{name}'"
        choices = " or ".join(f"{owner}.{name}" for owner in owners)
        return f"Parameter '{name}' needs a prefix, use {choices}"

    def _locate(self, key: str) -> Tuple[str, str]:
        section, dot, name = key.partition(".")
        if not dot:
            raise PipelineError(self._suggest(key))
        if section not in self._fields:
            raise PipelineError(
                f"Unknown prefix '{section}' in '{key}', use dp, da, ex or run"
            )
        if name not in self._fields[section]:
            raise PipelineError(f"Unknown parameter '{key}'")
        return section, name

    def parse(self, text: str) -> PipelineArgs:
        overrides: Dict[str, Dict[str, str]] = {section: {} for section in SECTIONS}
        run: Dict[str, str] = {}
        for token in self._split(text):
            key, sep, value = token.partition("=")
            if not sep or not key:
                raise PipelineError(f"Expected section.key=value, got '{token}'")
            section, name = self._locate(key)
            target = run if section == RUN_SECTION else overrides[section]
            if name in target:
                raise PipelineError(f"Duplicate parameter '{section}.{name}'")
            target[name] = value
        return PipelineArgs(overrides=overrides, run=run)

    @staticmethod
    def defaults() -> Dict[str, Dict[str, object]]:
        values = {
            section: dataclasses.asdict(config_cls())
            for section, config_cls in SECTIONS.items()
        }
        values[RUN_SECTION] = {"resume": None, "model_dir": "./artifacts"}
        return values
