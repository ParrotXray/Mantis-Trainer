import shutil
import uuid
from pathlib import Path
from typing import Final, List

from fastapi import UploadFile

from pipeline import CsvPool

from model.error import InvalidNameError, PayloadTooLargeError, StorageError
from ..schemas import DatasetFile, DatasetInventory
from .name_validator import validate_name
from .path_guard import resolve_within

_CHUNK_SIZE: Final[int] = 1024 * 1024
_STAGING_DIR: Final[str] = ".staging"


class DatasetStore:
    def __init__(self, pool: CsvPool, max_upload_bytes: int) -> None:
        self.pool = pool
        self.root = pool.root
        self.staging_root = self.root / _STAGING_DIR
        self.max_upload_bytes = max_upload_bytes
        self.root.mkdir(parents=True, exist_ok=True)

    def _csv_files(self) -> List[Path]:
        return self.pool.files()

    def inventory(self) -> DatasetInventory:
        files = [
            DatasetFile(
                name=str(path.relative_to(self.root)),
                size_bytes=path.stat().st_size,
                modified_at=path.stat().st_mtime,
            )
            for path in self._csv_files()
        ]
        return DatasetInventory(
            files=files, total_bytes=sum(f.size_bytes for f in files)
        )

    @staticmethod
    def _csv_name(upload: UploadFile) -> str:
        filename = Path(upload.filename or "").name
        validate_name(filename, "file")
        if Path(filename).suffix.lower() != ".csv":
            raise InvalidNameError(f"Only .csv files are accepted: '{filename}'")
        return filename

    async def _write(self, upload: UploadFile, target: Path) -> None:
        written = 0
        with target.open("wb") as handle:
            while chunk := await upload.read(_CHUNK_SIZE):
                written += len(chunk)
                if written > self.max_upload_bytes:
                    raise PayloadTooLargeError(
                        f"File '{target.name}' exceeds upload limit"
                    )
                handle.write(chunk)

    def _commit(self, staging: Path, replace: bool) -> None:
        if replace:
            self.clear()
        for path in staging.iterdir():
            path.replace(self.root / path.name)

    async def import_csv(
        self, uploads: List[UploadFile], replace: bool
    ) -> DatasetInventory:
        names = [self._csv_name(upload) for upload in uploads]
        if len(set(names)) != len(names):
            raise InvalidNameError("Duplicate file names in upload")
        staging = self.staging_root / uuid.uuid4().hex
        staging.mkdir(parents=True)
        try:
            for upload, filename in zip(uploads, names):
                try:
                    await self._write(upload, staging / filename)
                except OSError as exc:
                    raise StorageError(
                        f"Cannot store '{filename}' on the node: {exc.strerror or exc}"
                    ) from exc
            try:
                self._commit(staging, replace)
            except OSError as exc:
                raise StorageError(
                    f"Cannot move the upload into the dataset folder: {exc.strerror or exc}"
                ) from exc
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        return self.inventory()

    def delete(self, relative: str) -> None:
        path = resolve_within(self.root, relative)
        if not path.is_file() or path.suffix.lower() != ".csv":
            raise InvalidNameError(f"'{relative}' is not a CSV file")
        path.unlink()

    def clear(self) -> None:
        for path in self._csv_files():
            path.unlink()
