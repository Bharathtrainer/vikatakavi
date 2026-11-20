"""VikataKavi's eyes (Anvaya AI Lab).

Keeps ONLY the latest frame from the user's camera in memory (nothing is stored
on disk) and, when VikataKavi decides to "look", sends that single frame to a
vision model (Sarvam-hosted gemma4) for a short, neutral description.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time

from livekit import rtc
from livekit.agents import llm
from livekit.plugins import sarvam

logger = logging.getLogger("vikatakavi.vision")

VISION_MODEL = os.getenv("SARVAM_VISION_MODEL", "gemma4")
MAX_FRAME_AGE_S = 5.0  # don't describe a stale frame (camera frozen / turned off)

VISION_PROMPT = """Describe what this webcam image shows, for a friendly storyteller
who will chat with the person about it. Be concrete and brief (max 3 sentences):
objects, what the person is holding or showing, clothing colours, visible text
(transcribe it exactly), setting and mood.

Strict rules:
- Never identify who anyone is, and never guess a name, even for famous people.
- Never guess age, caste, religion, ethnicity, health, disability or body shape.
- If the image is too dark or blurry, say so.
- Plain text only."""


class CameraWatcher:
    """Tracks the user's camera and keeps the most recent frame."""

    def __init__(self, room: rtc.Room) -> None:
        self._room = room
        self._latest: rtc.VideoFrame | None = None
        self._latest_at = 0.0
        self._task: asyncio.Task | None = None
        self._track_sid: str | None = None
        self._llm = sarvam.LLM(model=VISION_MODEL, temperature=0.2, max_tokens=200)

        room.on("track_subscribed", self._on_track_subscribed)
        room.on("track_unsubscribed", self._on_track_unsubscribed)
        # Camera may already be published when we join.
        for p in room.remote_participants.values():
            for pub in p.track_publications.values():
                if pub.track is not None:
                    self._on_track_subscribed(pub.track, pub, p)

    # --- camera tracking -------------------------------------------------------
    def _is_user_camera(self, track: rtc.Track, pub: rtc.RemoteTrackPublication,
                        participant: rtc.RemoteParticipant) -> bool:
        if track.kind != rtc.TrackKind.KIND_VIDEO:
            return False
        if participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_AGENT:
            return False  # e.g. the Runway avatar's own video
        return pub.source == rtc.TrackSource.SOURCE_CAMERA

    def _on_track_subscribed(self, track, pub, participant) -> None:
        if not self._is_user_camera(track, pub, participant):
            return
        if self._task:
            self._task.cancel()
        self._track_sid = pub.sid
        self._task = asyncio.create_task(self._read(track))
        logger.info("camera on: watching %s", participant.identity)

    def _on_track_unsubscribed(self, track, pub, participant) -> None:
        if pub.sid == self._track_sid and self._task:
            self._task.cancel()
            self._task = None
            self._latest = None
            logger.info("camera off")

    async def _read(self, track: rtc.Track) -> None:
        stream = rtc.VideoStream(track)
        try:
            async for ev in stream:
                self._latest = ev.frame  # overwrite: only the newest frame is kept
                self._latest_at = time.monotonic()
        except asyncio.CancelledError:
            pass
        finally:
            await stream.aclose()

    @property
    def camera_on(self) -> bool:
        return self._latest is not None and time.monotonic() - self._latest_at < MAX_FRAME_AGE_S

    # --- looking ---------------------------------------------------------------
    async def describe(self, focus: str = "") -> str | None:
        """Return a short description of the current frame, or None if no camera."""
        if not self.camera_on:
            return None
        frame = self._latest
        ctx = llm.ChatContext.empty()
        prompt = VISION_PROMPT + (f"\n\nThe user asked specifically about: {focus}" if focus else "")
        ctx.add_message(
            role="user",
            content=[prompt, llm.ImageContent(image=frame, inference_width=768, inference_height=768)],
        )
        parts: list[str] = []
        async with self._llm.chat(chat_ctx=ctx) as stream:
            async for chunk in stream:
                if chunk.delta and chunk.delta.content:
                    parts.append(chunk.delta.content)
        text = "".join(parts).strip()
        logger.info("vision: %s", text[:160])
        return text or None

    async def aclose(self) -> None:
        if self._task:
            self._task.cancel()
        self._latest = None
