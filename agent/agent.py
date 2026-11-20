"""VikataKavi conversational agent (Anvaya AI Lab).

Pipeline:  user mic -> Sarvam Saaras STT (Telugu/English, code-mixed)
                    -> Sarvam LLM (persona)
                    -> Sarvam Bulbul TTS
                    -> avatar layer (Runway Characters today, self-hosted GPU later)

Run locally:   python agent.py dev
Run on Render: python agent.py start   (Background Worker)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RunContext,
    WorkerOptions,
    cli,
    function_tool,
)
# Import every plugin at module level: LiveKit requires plugins to register on the main thread.
from livekit.plugins import runway, sarvam, silero

import persona
from vision import CameraWatcher

load_dotenv()
logger = logging.getLogger("vikatakavi")

AGENT_NAME = os.getenv("AGENT_NAME", "vikatakavi")
SUPPORTED_LANGS = {"te-IN", "en-IN"}

# --- Configuration (all via environment variables) -----------------------------
DEFAULT_LANG = os.getenv("DEFAULT_LANGUAGE", "te-IN")
LLM_MODEL = os.getenv("SARVAM_LLM_MODEL", "sarvam-105b-conversations")
STT_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v3")
TTS_MODEL = os.getenv("SARVAM_TTS_MODEL", "bulbul:v3")
TTS_SPEAKER = os.getenv("SARVAM_TTS_SPEAKER", "aditya")  # mature male storyteller voice
TTS_PACE = float(os.getenv("SARVAM_TTS_PACE", "1.0"))
# AVATAR_PROVIDER:
#   "none"       -> default: the browser animates the portrait itself (lip sync, blinks,
#                   head motion) from the agent's voice. Zero avatar cost.
#   "runway"     -> hosted photoreal avatar (Runway Characters, paid per minute).
#   "selfhosted" -> phase 2: your own GPU neural lip-sync worker.
AVATAR_PROVIDER = os.getenv("AVATAR_PROVIDER", "none").lower()
RUNWAY_AVATAR_ID = os.getenv("RUNWAY_AVATAR_ID", "")
# Runway sessions are capped at 5 minutes; keep the prototype cheap.
MAX_CALL_SECONDS = int(os.getenv("MAX_CALL_SECONDS", "180"))
VISION_ENABLED = os.getenv("VISION_ENABLED", "true").lower() == "true"


class VikataKavi(Agent):
    def __init__(self, language: str, eyes: CameraWatcher | None) -> None:
        super().__init__(instructions=persona.instructions(language, vision=eyes is not None))
        self.language = language
        self.eyes = eyes

    async def on_enter(self) -> None:
        # Speak a fixed greeting first (fast, no LLM round-trip).
        await self.session.say(persona.greeting(self.language), allow_interruptions=True)

    @function_tool()
    async def look_at_camera(self, context: RunContext, focus: str = "") -> str:
        """Look through the user's camera right now and get a short description of what is visible.

        Call this whenever answering needs your eyes: the user asks you to look or see,
        asks what they are holding or showing, how they or something looks, or to read
        text they are holding up.

        Args:
            focus: Optional short note in English on what to pay attention to,
                for example "the object in their hand" or "the text on the paper".
        """
        if self.eyes is None or not self.eyes.camera_on:
            return ("CAMERA_OFF: the user's camera is off. Playfully ask them to tap "
                    "the 'Show camera' button so you can see.")
        filler = "ఒక్క క్షణం, చూస్తున్నాను..." if self.language == "te-IN" else "One moment, let me look..."
        context.session.say(filler, add_to_chat_ctx=False, allow_interruptions=False)
        try:
            seen = await self.eyes.describe(focus)
        except Exception:
            logger.exception("vision call failed")
            return "VISION_ERROR: you could not see clearly. Say so lightly and ask them to try again."
        if not seen:
            return "CAMERA_OFF: no clear picture. Ask them to check their camera."
        return (f"WHAT_YOU_SEE: {seen}\n"
                "Reply in character, briefly, in the conversation language. Never identify "
                "who anyone is or comment on age, caste, religion, health or body.")


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()


def _language_from_metadata(ctx: JobContext) -> str:
    try:
        meta = json.loads(ctx.job.metadata or "{}")
    except json.JSONDecodeError:
        meta = {}
    lang = meta.get("language", DEFAULT_LANG)
    return lang if lang in SUPPORTED_LANGS else DEFAULT_LANG


async def _start_avatar(session: AgentSession, ctx: JobContext) -> None:
    if AVATAR_PROVIDER == "none":
        logger.info("AVATAR_PROVIDER=none: browser-animated portrait")
        return
    if AVATAR_PROVIDER == "runway":
        if not RUNWAY_AVATAR_ID:
            raise RuntimeError("Set RUNWAY_AVATAR_ID to your VikataKavi Character ID")
        avatar = runway.AvatarSession(
            avatar_id=RUNWAY_AVATAR_ID,
            max_duration=min(MAX_CALL_SECONDS, 300),
            avatar_participant_name="VikataKavi",
        )
        await avatar.start(session, room=ctx.room)
        return
    if AVATAR_PROVIDER == "selfhosted":
        # Phase 2: plug in your GPU lip-sync worker here (see docs/gpu-migration.md).
        raise NotImplementedError("selfhosted avatar provider is planned for phase 2")
    raise ValueError(f"Unknown AVATAR_PROVIDER: {AVATAR_PROVIDER}")


async def entrypoint(ctx: JobContext) -> None:
    language = _language_from_metadata(ctx)
    logger.info("starting VikataKavi call, language=%s, avatar=%s", language, AVATAR_PROVIDER)

    await ctx.connect()

    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        stt=sarvam.STT(
            language=language,
            model=STT_MODEL,
            # codemix keeps English words in English script inside Telugu speech
            mode="codemix" if language == "te-IN" else "transcribe",
        ),
        llm=sarvam.LLM(model=LLM_MODEL, temperature=0.7, max_tokens=300),
        tts=sarvam.TTS(
            target_language_code=language,
            model=TTS_MODEL,
            speaker=TTS_SPEAKER,
            pace=TTS_PACE,
        ),
        preemptive_generation=True,
        # Ignore short noises / one-word sounds so stories aren't cut off mid-sentence.
        min_interruption_duration=float(os.getenv("MIN_INTERRUPTION_SECONDS", "0.8")),
        min_interruption_words=int(os.getenv("MIN_INTERRUPTION_WORDS", "2")),
    )

    eyes = CameraWatcher(ctx.room) if VISION_ENABLED else None
    if eyes:
        ctx.add_shutdown_callback(eyes.aclose)

    try:
        await _start_avatar(session, ctx)
    except Exception as e:
        # Never let the face service (credits, outage) kill the call: fall back to voice-only.
        logger.warning("avatar unavailable, continuing voice-only: %s", e)
    await session.start(agent=VikataKavi(language, eyes), room=ctx.room)

    # Hard stop so a forgotten tab never burns credits.
    async def _limit() -> None:
        await asyncio.sleep(max(MAX_CALL_SECONDS - 10, 10))
        logger.info("max call duration reached, ending call")
        await session.say(
            "మన సంభాషణ సమయం ముగిసింది. మళ్ళీ కలుద్దాం!"
            if language == "te-IN"
            else "Our time is up for now. Come back soon!",
            allow_interruptions=False,
        )
        ctx.shutdown(reason="max_duration")

    limit_task = asyncio.create_task(_limit())
    ctx.add_shutdown_callback(lambda: _cancel(limit_task))


async def _cancel(task: asyncio.Task) -> None:
    task.cancel()


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            agent_name=AGENT_NAME,  # explicit dispatch: only rooms that request us
        )
    )
