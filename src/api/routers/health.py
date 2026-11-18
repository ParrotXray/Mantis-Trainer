from fastapi import APIRouter, Depends

from ..dependencies import get_settings
from ..schemas import HealthStatus
from ..settings import ApiSettings

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthStatus)
def health(settings: ApiSettings = Depends(get_settings)) -> HealthStatus:
    return HealthStatus(status="ok", node_id=settings.node_id)
