import os
from pathlib import Path
from typing import Final, List

SOURCE_NAME: Final[str] = "imported"


class CsvPool:
    def __init__(self, root: Path) -> None:
        self.root = root

    @staticmethod
    def default_data_root() -> Path:
        return Path(os.getenv("DATASET_ROOT", "./datasets")).resolve()

    def files(self) -> List[Path]:
        if not self.root.is_dir():
            return []
        return sorted(
            path
            for path in self.root.rglob("*")
            if path.is_file()
            and path.suffix.lower() == ".csv"
            and not any(
                part.startswith(".") for part in path.relative_to(self.root).parts
            )
        )

    def names(self) -> List[str]:
        return [str(path.relative_to(self.root)) for path in self.files()]
