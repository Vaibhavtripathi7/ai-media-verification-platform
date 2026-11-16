import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from .config import Settings
from .detector import Detector
from .imaging import ImageTooLarge, InvalidImage
from .weights import download

# uvicorn's own logger, so our startup lines come out in the same format as its own.
log = logging.getLogger("uvicorn.error")


class Detection(BaseModel):
    logit: float
    model: str
    decode_ms: float
    wait_ms: float
    model_ms: float


class Health(BaseModel):
    status: str
    model: str


async def read_body(request: Request, limit: int) -> bytes:
    if int(request.headers.get("content-length") or 0) > limit:
        raise HTTPException(413, "image is too large")
    chunks, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > limit:
            raise HTTPException(413, "image is too large")
        chunks.append(chunk)
    if size == 0:
        raise HTTPException(400, "request body is empty")
    return b"".join(chunks)


def create_app(settings: Settings | None = None, detector: Detector | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    # One model call at a time. ONNX Runtime already spreads a single call over
    # every core it was given, so running two at once only makes both slower.
    model_thread = ThreadPoolExecutor(max_workers=1, thread_name_prefix="model")
    pending = 0

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal detector
        if detector is None:
            started = time.perf_counter()
            detector = Detector(
                download(settings), settings.model_file, settings.threads, settings.max_pixels
            )
            size = detector.preprocess.crop_size
            detector.predict(np.zeros((1, 3, size, size), dtype=np.float32))
            log.info("loaded %s in %.1fs", settings.model_id, time.perf_counter() - started)
        yield
        model_thread.shutdown(wait=True)

    app = FastAPI(title="inference", lifespan=lifespan, docs_url=None, redoc_url=None)

    @app.get("/health")
    async def health() -> Health:
        return Health(status="ok", model=settings.model_id)

    @app.post("/v1/detect")
    async def detect(request: Request) -> Detection:
        nonlocal pending
        if pending >= settings.max_pending:
            raise HTTPException(503, "detector is busy", headers={"Retry-After": "1"})
        pending += 1
        try:
            data = await read_body(request, settings.max_body_bytes)

            started = time.perf_counter()
            try:
                pixels = await run_in_threadpool(
                    lambda: detector.preprocess(detector.preprocess.decode(data))
                )
            except InvalidImage as exc:
                raise HTTPException(422, str(exc)) from exc
            except ImageTooLarge as exc:
                raise HTTPException(413, str(exc)) from exc
            decoded = time.perf_counter()

            def run_model() -> tuple[float, float]:
                begin = time.perf_counter()
                return float(detector.predict(pixels[None])[0]), begin

            logit, model_started = await asyncio.get_running_loop().run_in_executor(
                model_thread, run_model
            )
            finished = time.perf_counter()
        finally:
            pending -= 1

        return Detection(
            logit=logit,
            model=settings.model_id,
            decode_ms=round((decoded - started) * 1000, 1),
            wait_ms=round((model_started - decoded) * 1000, 1),
            model_ms=round((finished - model_started) * 1000, 1),
        )

    return app
