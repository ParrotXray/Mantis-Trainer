import os
import socket
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Optional, Tuple

from model.error import SettingsError

MIN_TOKEN_LENGTH: Final[int] = 16


def _default_node_id() -> str:
    return f"{socket.gethostname()}-{uuid.uuid4().hex[:8]}"


@dataclass(frozen=True)
class ApiSettings:
    workspace: Path
    dataset_root: Path
    node_id: str
    api_token: str
    host: str = "0.0.0.0"
    port: int = 8000
    max_upload_mb: int = 10240
    cpu_sample_interval: float = 0.2
    ws_min_interval: float = 0.5
    ws_max_interval: float = 60.0
    cors_origins: Tuple[str, ...] = ()

    @staticmethod
    def _token(value: str) -> str:
        token = value.strip()
        if not token:
            raise SettingsError(
                "API_TOKEN is required: set it to a random string of at least "
                f"{MIN_TOKEN_LENGTH} characters (e.g. openssl rand -hex 32)"
            )
        if len(token) < MIN_TOKEN_LENGTH:
            raise SettingsError(
                f"API_TOKEN is too short: use at least {MIN_TOKEN_LENGTH} characters"
            )
        return token

    @classmethod
    def from_env(cls) -> "ApiSettings":
        workspace = Path(os.getenv("WORKSPACE", ".")).resolve()
        dataset_root: Optional[str] = os.getenv("DATASET_ROOT")
        return cls(
            workspace=workspace,
            dataset_root=(
                Path(dataset_root).resolve() if dataset_root else workspace / "datasets"
            ),
            node_id=os.getenv("NODE_ID") or _default_node_id(),
            host=os.getenv("API_HOST", cls.host),
            port=int(os.getenv("API_PORT", str(cls.port))),
            max_upload_mb=int(os.getenv("MAX_UPLOAD_MB", str(cls.max_upload_mb))),
            cpu_sample_interval=float(
                os.getenv("CPU_SAMPLE_INTERVAL", str(cls.cpu_sample_interval))
            ),
            ws_min_interval=float(
                os.getenv("WS_MIN_INTERVAL", str(cls.ws_min_interval))
            ),
            ws_max_interval=float(
                os.getenv("WS_MAX_INTERVAL", str(cls.ws_max_interval))
            ),
            api_token=cls._token(os.getenv("API_TOKEN", "")),
            cors_origins=tuple(
                origin.strip()
                for origin in os.getenv("CORS_ORIGINS", "").split(",")
                if origin.strip()
            ),
        )

    @property
    def log_dir(self) -> Path:
        return self.workspace / "logs"

    @property
    def job_log_dir(self) -> Path:
        return self.log_dir / "jobs"

    @property
    def artifact_dir(self) -> Path:
        return self.workspace / "artifacts"
