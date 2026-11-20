"""VikataKavi web service (Anvaya AI Lab).

Serves the call page and issues short-lived LiveKit tokens that also dispatch
the VikataKavi agent into a fresh room. No AI API keys ever reach the browser.
"""

from __future__ import annotations

import json
import os
from datetime import timedelta
import secrets
import time
from collections import defaultdict, deque
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from livekit import api
from pydantic import BaseModel

load_dotenv()

LIVEKIT_URL = os.getenv("LIVEKIT_URL", "")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY", "")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET", "")
AGENT_NAME = os.getenv("AGENT_NAME", "vikatakavi")
# Optional shared passcode so strangers can't spend your credits on the prototype.
ACCESS_CODE = os.getenv("ACCESS_CODE", "")
CALLS_PER_HOUR_PER_IP = int(os.getenv("CALLS_PER_HOUR_PER_IP", "6"))
LANGS = {"te-IN", "en-IN"}

STATIC = Path(__file__).parent / "static"
app = FastAPI(title="VikataKavi — Anvaya AI Lab")
app.mount("/static", StaticFiles(directory=STATIC), name="static")

_calls: dict[str, deque[float]] = defaultdict(deque)


class SessionRequest(BaseModel):
    language: str = "te-IN"
    name: str | None = None
    access_code: str | None = None


def _rate_limited(ip: str) -> bool:
    now = time.time()
    q = _calls[ip]
    while q and now - q[0] > 3600:
        q.popleft()
    if len(q) >= CALLS_PER_HOUR_PER_IP:
        return True
    q.append(now)
    return False


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/healthz")
def health() -> dict:
    return {"ok": True, "livekit_configured": bool(LIVEKIT_URL and LIVEKIT_API_KEY)}


@app.get("/api/config")
def config() -> dict:
    return {"requires_access_code": bool(ACCESS_CODE)}


@app.post("/api/session")
def create_session(body: SessionRequest, request: Request) -> JSONResponse:
    if not (LIVEKIT_URL and LIVEKIT_API_KEY and LIVEKIT_API_SECRET):
        raise HTTPException(500, "Server is missing LiveKit configuration")
    if ACCESS_CODE and not secrets.compare_digest(body.access_code or "", ACCESS_CODE):
        raise HTTPException(401, "Wrong access code")
    ip = request.headers.get("x-forwarded-for", request.client.host if request.client else "?")
    if _rate_limited(ip.split(",")[0].strip()):
        raise HTTPException(429, "Too many calls this hour — please try later")

    language = body.language if body.language in LANGS else "te-IN"
    room = f"vikatakavi-{secrets.token_hex(6)}"
    identity = f"guest-{secrets.token_hex(4)}"

    token = (
        api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
        .with_identity(identity)
        .with_name((body.name or "Guest")[:40])
        .with_ttl(timedelta(minutes=10))
        .with_grants(api.VideoGrants(room_join=True, room=room, can_publish=True, can_subscribe=True))
        .with_room_config(
            api.RoomConfiguration(
                agents=[
                    api.RoomAgentDispatch(
                        agent_name=AGENT_NAME,
                        metadata=json.dumps({"language": language}),
                    )
                ],
                empty_timeout=30,
            )
        )
        .to_jwt()
    )
    return JSONResponse({"serverUrl": LIVEKIT_URL, "token": token, "room": room})
