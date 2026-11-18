import json
import socket

from fastapi import APIRouter, Depends, Query, WebSocket, status
from starlette.concurrency import run_in_threadpool

from pipeline import PIPELINE_ORDER

from ..dependencies import get_job_service, get_metrics_stream, get_settings
from ..errors import InvalidParameterError
from ..schemas import NodeInfo, NodeSnapshot
from ..services import JobService, MetricsStream
from ..settings import ApiSettings
from .ws_session import WsSession

router = APIRouter(tags=["node"])


def _node_info(settings: ApiSettings, jobs: JobService) -> NodeInfo:
    current = jobs.current()
    return NodeInfo(
        node_id=settings.node_id,
        hostname=socket.gethostname(),
        busy=current is not None,
        current_job=current.job_id if current else None,
        stages=[stage.value for stage in PIPELINE_ORDER],
    )


@router.get("/node", response_model=NodeInfo)
def node_info(
    settings: ApiSettings = Depends(get_settings),
    jobs: JobService = Depends(get_job_service),
) -> NodeInfo:
    return _node_info(settings, jobs)


@router.websocket("/node/ws")
async def stream_node(
    websocket: WebSocket,
    interval: float = Query(5.0),
    settings: ApiSettings = Depends(get_settings),
    jobs: JobService = Depends(get_job_service),
    stream: MetricsStream = Depends(get_metrics_stream),
) -> None:
    try:
        stream.validate_interval(interval)
    except InvalidParameterError as exc:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason=exc.message)
        return

    async def push(ws: WebSocket) -> None:
        last_jobs = None
        async for metrics in stream.stream(interval):
            job_list = await run_in_threadpool(jobs.list)
            encoded = json.dumps([job.model_dump(mode="json") for job in job_list])
            snapshot = NodeSnapshot(
                node=_node_info(settings, jobs),
                metrics=metrics,
                jobs=job_list if encoded != last_jobs else None,
            )
            last_jobs = encoded
            await ws.send_text(snapshot.model_dump_json())

    await WsSession.run(websocket, push)
