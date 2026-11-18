import os
import tempfile
import zipfile
from pathlib import Path
from typing import List

from ..errors import InvalidNameError, ResourceNotFoundError
from ..schemas import FileCategory, FileInfo
from .path_guard import resolve_within


class FileStore:
    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace

    def _root(self, category: FileCategory) -> Path:
        return self.workspace / category.value

    def list(self, category: FileCategory) -> List[FileInfo]:
        root = self._root(category)
        if not root.is_dir():
            return []
        return [
            FileInfo(
                name=str(path.relative_to(root)),
                size_bytes=path.stat().st_size,
                modified_at=path.stat().st_mtime,
            )
            for path in sorted(root.rglob("*"))
            if path.is_file()
        ]

    def file_of(self, category: FileCategory, relative: str) -> Path:
        path = resolve_within(self._root(category), relative)
        if not path.is_file():
            raise InvalidNameError(f"'{relative}' is not a file")
        return path

    def archive(self, category: FileCategory, relative: str) -> Path:
        root = self._root(category)
        source = resolve_within(root, relative) if relative else root
        if not source.is_dir():
            raise ResourceNotFoundError(f"Directory '{relative}' not found")
        handle, name = tempfile.mkstemp(prefix="mantis-", suffix=".zip")
        os.close(handle)
        target = Path(name)
        try:
            with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
                for path in sorted(source.rglob("*")):
                    if path.is_file():
                        archive.write(path, path.relative_to(source))
        except Exception:
            target.unlink(missing_ok=True)
            raise
        return target
