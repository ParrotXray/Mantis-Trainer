from .datasets import router as datasets_router
from .files import router as files_router
from .health import router as health_router
from .jobs import router as jobs_router
from .logs import router as logs_router
from .metrics import router as metrics_router
from .node import router as node_router

ROUTERS = (
    node_router,
    metrics_router,
    datasets_router,
    jobs_router,
    logs_router,
    files_router,
)

PUBLIC_ROUTERS = (health_router,)

__all__ = ("PUBLIC_ROUTERS", "ROUTERS")
