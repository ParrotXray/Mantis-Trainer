import dataclasses
import typing
from typing import Any, Dict, Type, TypeVar

from pydantic import TypeAdapter, ValidationError

from model.error import PipelineError

T = TypeVar("T")

_NONE_WORDS = frozenset({"none", "null"})


class ConfigBuilder:
    @staticmethod
    def _is_tuple(annotation: Any) -> bool:
        return typing.get_origin(annotation) is tuple

    @staticmethod
    def _allows_none(annotation: Any) -> bool:
        return type(None) in typing.get_args(annotation)

    @classmethod
    def _coerce(cls, annotation: Any, value: Any) -> Any:
        if isinstance(value, str):
            if cls._is_tuple(annotation):
                if value.strip().lower() in _NONE_WORDS:
                    return ()
                return tuple(item.strip() for item in value.split(",") if item.strip())
            if cls._allows_none(annotation) and value.strip().lower() in _NONE_WORDS:
                return None
        if isinstance(value, bool) and annotation is not bool:
            raise ValueError("boolean is not accepted")
        return TypeAdapter(annotation).validate_python(value)

    @classmethod
    def build(cls, config_cls: Type[T], overrides: Dict[str, Any]) -> T:
        hints = typing.get_type_hints(config_cls)
        names = {item.name for item in dataclasses.fields(config_cls)}
        unknown = sorted(set(overrides) - names)
        if unknown:
            raise PipelineError(
                f"Unknown parameter(s) for {config_cls.__name__}: {', '.join(unknown)}"
            )
        values: Dict[str, Any] = {}
        for key, value in overrides.items():
            try:
                values[key] = cls._coerce(hints[key], value)
            except (ValidationError, ValueError) as exc:
                raise PipelineError(
                    f"Invalid value for {config_cls.__name__}.{key}: {value!r}"
                ) from exc
        return config_cls(**values)
