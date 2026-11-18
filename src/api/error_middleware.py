from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from utils import Logger


class UnhandledErrorMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        self.log = Logger(__name__)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        started = False

        async def tracked(message: Message) -> None:
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, receive, tracked)
        except Exception as exc:
            self.log.error(f"{scope['method']} {scope['path']} failed: {exc!r}")
            self.log.logger.exception(exc)
            if started:
                raise
            response = JSONResponse(
                status_code=500,
                content={
                    "detail": f"Internal error on the node: {type(exc).__name__}: {exc}"
                },
            )
            await response(scope, receive, send)
