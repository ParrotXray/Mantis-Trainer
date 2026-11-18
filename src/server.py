#!/usr/bin/env python3

import os
import sys

import uvicorn

from api import ApiSettings, SettingsError, create_app
from utils import Logger

if __name__ == "__main__":
    try:
        settings = ApiSettings.from_env()
    except SettingsError as exc:
        sys.exit(f"Cannot start the API server: {exc}")
    finally:
        os.environ["PYTHONPATH"] = str(settings.workspace)

    os.chdir(settings.workspace)
    Logger.setup_logging()

    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        workers=1,
        log_config=None,
    )
