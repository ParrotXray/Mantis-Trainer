from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .settings import ApiSettings


def register_cors(app: FastAPI, settings: ApiSettings) -> None:
    if not settings.cors_origins:
        return
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["Content-Disposition"],
    )
