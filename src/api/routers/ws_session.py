import asyncio
from typing import Awaitable, Callable

from fastapi import WebSocket, status


class WsSession:
    @staticmethod
    async def _wait_disconnect(websocket: WebSocket) -> None:
        while (await websocket.receive())["type"] != "websocket.disconnect":
            continue

    @staticmethod
    async def run(
        websocket: WebSocket, push: Callable[[WebSocket], Awaitable[None]]
    ) -> None:
        await websocket.accept()
        tasks = {
            asyncio.create_task(push(websocket)),
            asyncio.create_task(WsSession._wait_disconnect(websocket)),
        }
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        for task in done:
            if not task.cancelled() and task.exception() is not None:
                await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
