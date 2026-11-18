import asyncio
from collections import deque
from pathlib import Path
from typing import AsyncIterator, Callable, List, Optional

from model.error import ResourceNotFoundError
from ..schemas import FileInfo, LogContent
from .path_guard import resolve_within

_BLOCK_SIZE = 8192


class LogReader:
    def __init__(self, log_dir: Path, poll_interval: float = 0.5) -> None:
        self.log_dir = log_dir
        self.poll_interval = poll_interval

    def resolve(self, relative: str) -> Path:
        path = resolve_within(self.log_dir, relative)
        if not path.is_file():
            raise ResourceNotFoundError(f"Log '{relative}' not found")
        return path

    def list(self) -> List[FileInfo]:
        if not self.log_dir.is_dir():
            return []
        return [
            FileInfo(
                name=str(path.relative_to(self.log_dir)),
                size_bytes=path.stat().st_size,
                modified_at=path.stat().st_mtime,
            )
            for path in sorted(self.log_dir.rglob("*"))
            if path.is_file() and ".log" in path.name
        ]

    @staticmethod
    def _tail_lines(path: Path, count: int, end: Optional[int] = None) -> List[str]:
        if count <= 0:
            return []
        with path.open("rb") as handle:
            position = end if end is not None else handle.seek(0, 2)
            buffer = b""
            while position > 0 and buffer.count(b"\n") <= count:
                step = min(_BLOCK_SIZE, position)
                position -= step
                handle.seek(position)
                buffer = handle.read(step) + buffer
        lines = buffer.decode("utf-8", errors="replace").splitlines()
        return list(deque(lines, maxlen=count))

    def tail(self, relative: str, count: int) -> LogContent:
        path = self.resolve(relative)
        return LogContent(name=relative, lines=self._tail_lines(path, count))

    def follow(
        self, relative: str, count: int, is_active: Callable[[], bool]
    ) -> AsyncIterator[str]:
        return self._follow(self.resolve(relative), count, is_active)

    async def _follow(
        self, path: Path, count: int, is_active: Callable[[], bool]
    ) -> AsyncIterator[str]:
        with path.open("rb") as handle:
            start = handle.seek(0, 2)
            for line in self._tail_lines(path, count, start):
                yield line
            pending = b""
            while True:
                active = is_active()
                chunk = handle.read()
                if chunk:
                    pending += chunk
                    *lines, pending = pending.split(b"\n")
                    for line in lines:
                        yield line.decode("utf-8", errors="replace").rstrip("\r")
                    continue
                if not active:
                    break
                await asyncio.sleep(self.poll_interval)
            if pending:
                yield pending.decode("utf-8", errors="replace").rstrip("\r")
