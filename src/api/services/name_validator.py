import re
from typing import Final

from model.error import InvalidNameError

_SAFE_NAME: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,127}$")


def validate_name(value: str, kind: str) -> str:
    if not _SAFE_NAME.fullmatch(value) or ".." in value:
        raise InvalidNameError(f"Invalid {kind} name: '{value}'")
    return value
