import secrets
from typing import Final

from fastapi import WebSocketException, status
from starlette.requests import HTTPConnection

from model.error import UnauthorizedError

_BEARER: Final[str] = "bearer"


class TokenGuard:
    def __init__(self, token: str) -> None:
        self.token = token

    @staticmethod
    def _presented(connection: HTTPConnection) -> str:
        scheme, _, credentials = connection.headers.get("authorization", "").partition(
            " "
        )
        if scheme.lower() == _BEARER:
            return credentials.strip()
        return connection.query_params.get("token", "")

    def verify(self, connection: HTTPConnection) -> None:
        if secrets.compare_digest(
            self._presented(connection).encode(), self.token.encode()
        ):
            return
        if connection.scope["type"] == "websocket":
            raise WebSocketException(
                code=status.WS_1008_POLICY_VIOLATION, reason="Invalid or missing token"
            )
        raise UnauthorizedError("Invalid or missing token")
