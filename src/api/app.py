from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI

from utils import Logger

from .container import Container
from .cors import register_cors
from .dependencies import require_token
from .error_middleware import UnhandledErrorMiddleware
from .exception_handlers import register_exception_handlers
from .routers import PUBLIC_ROUTERS, ROUTERS
from .settings import ApiSettings

API_PREFIX = "/api/v1"


def create_app(settings: Optional[ApiSettings] = None) -> FastAPI:
    container = Container.build(settings or ApiSettings.from_env())

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        Logger.setup_logging()
        yield
        container.jobs.shutdown()

    app = FastAPI(title="Mantis Trainer API", version="1.0.0", lifespan=lifespan)
    app.state.container = container
    register_exception_handlers(app)
    app.add_middleware(UnhandledErrorMiddleware)
    register_cors(app, container.settings)
    for router in PUBLIC_ROUTERS:
        app.include_router(router, prefix=API_PREFIX)
    for router in ROUTERS:
        app.include_router(
            router, prefix=API_PREFIX, dependencies=[Depends(require_token)]
        )
    return app
