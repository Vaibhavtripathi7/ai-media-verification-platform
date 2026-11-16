import asyncio

import httpx
import pytest

from app.main import create_app

from .support import FakeDetector, encode


def client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


async def test_returns_logit_and_timings(settings):
    async with client(create_app(settings, FakeDetector(1.5))) as http:
        res = await http.post("/v1/detect", content=encode((64, 48)))
    assert res.status_code == 200
    body = res.json()
    assert body["logit"] == 1.5
    assert body["model"] == settings.model_id
    assert {"decode_ms", "wait_ms", "model_ms"} <= body.keys()


async def test_health_reports_model(settings):
    async with client(create_app(settings, FakeDetector())) as http:
        res = await http.get("/health")
    assert res.json() == {"status": "ok", "model": settings.model_id}


@pytest.mark.parametrize(
    ("body", "status"),
    [
        (b"", 400),
        (b"not an image", 422),
        (b"x" * 200_001, 413),
    ],
)
async def test_rejects_bad_uploads(settings, body, status):
    async with client(create_app(settings, FakeDetector())) as http:
        res = await http.post("/v1/detect", content=body)
    assert res.status_code == status


async def test_rejects_images_over_the_pixel_limit(settings):
    detector = FakeDetector(max_pixels=100)
    async with client(create_app(settings, detector)) as http:
        res = await http.post("/v1/detect", content=encode((20, 20)))
    assert res.status_code == 413


async def test_sheds_load_once_the_queue_is_full(settings):
    detector = FakeDetector()
    detector.release.clear()
    async with client(create_app(settings, detector)) as http:
        stuck = [
            asyncio.create_task(http.post("/v1/detect", content=encode((64, 64))))
            for _ in range(settings.max_pending)
        ]
        await asyncio.sleep(0.2)
        overflow = await http.post("/v1/detect", content=encode((64, 64)))
        detector.release.set()
        done = await asyncio.gather(*stuck)

    assert overflow.status_code == 503
    assert overflow.headers["retry-after"] == "1"
    assert [r.status_code for r in done] == [200] * settings.max_pending
