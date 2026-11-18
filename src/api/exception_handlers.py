from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from model.error import PipelineError

from model.error import ApiError


async def _handle_api_error(_: Request, exc: ApiError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})


async def _handle_pipeline_error(_: Request, exc: PipelineError) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, content={"detail": str(exc)}
    )


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(ApiError, _handle_api_error)
    app.add_exception_handler(PipelineError, _handle_pipeline_error)
