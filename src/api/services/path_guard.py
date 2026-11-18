from pathlib import Path

from model.error import InvalidNameError, ResourceNotFoundError


def resolve_inside(base: Path, relative: str, label: str = "path") -> Path:
    root = base.resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise InvalidNameError(f"Invalid {label}: '{relative}'")
    return path


def resolve_within(base: Path, relative: str) -> Path:
    path = resolve_inside(base, relative)
    if not path.exists():
        raise ResourceNotFoundError(f"'{relative}' not found")
    return path
