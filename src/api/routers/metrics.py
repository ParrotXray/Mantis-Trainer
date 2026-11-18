from fastapi import APIRouter, Depends, Query, WebSocket, status

from ..dependencies import get_metrics_stream, get_monitor
from ..errors import InvalidParameterError
from ..schemas import CpuMetrics, GpuMetrics, MemoryMetrics, SystemMetrics
from ..services import MetricsStream, SystemMonitor
from .ws_session import WsSession

router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.get("", response_model=SystemMetrics)
def read_all(monitor: SystemMonitor = Depends(get_monitor)) -> SystemMetrics:
    return monitor.collect()


@router.get("/cpu", response_model=CpuMetrics)
def read_cpu(monitor: SystemMonitor = Depends(get_monitor)) -> CpuMetrics:
    return monitor.cpu.collect()


@router.get("/memory", response_model=MemoryMetrics)
def read_memory(monitor: SystemMonitor = Depends(get_monitor)) -> MemoryMetrics:
    return monitor.memory.collect()


@router.get("/gpu", response_model=GpuMetrics)
def read_gpu(monitor: SystemMonitor = Depends(get_monitor)) -> GpuMetrics:
    return monitor.gpu.collect()


@router.websocket("/ws")
async def stream_metrics(
    websocket: WebSocket,
    interval: float = Query(1.0),
    stream: MetricsStream = Depends(get_metrics_stream),
) -> None:
    try:
        stream.validate_interval(interval)
    except InvalidParameterError as exc:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason=exc.message)
        return

    async def push(ws: WebSocket) -> None:
        async for metrics in stream.stream(interval):
            await ws.send_text(metrics.model_dump_json())

    await WsSession.run(websocket, push)
