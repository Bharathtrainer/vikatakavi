# Phase 2: moving the face to your own GPU

The prototype uses Runway Characters for the face, because Render has no GPUs.
Everything else (web page, agent, Sarvam speech and LLM) is already provider-neutral.
Only the **avatar layer** changes.

## What stays the same

- `web/`: unchanged, still on Render.
- `agent/agent.py`: same STT → LLM → TTS pipeline.
- LiveKit: same rooms and tokens. The self-hosted avatar joins the room as a
  participant, just like Runway's avatar does today.

## What changes

Set `AVATAR_PROVIDER=selfhosted` and implement the `selfhosted` branch in
`_start_avatar()`. LiveKit Agents' avatar framework (`livekit.agents.voice.avatar`)
gives you the same pattern the Runway plugin uses:

1. The agent streams its TTS audio to an avatar worker over a LiveKit data stream
   (`DataStreamAudioOutput`).
2. The avatar worker runs on your GPU box. It receives the audio, runs a real-time
   lip-sync / talking-head model on the VikataKavi portrait, and publishes video plus
   audio back into the room (`AvatarRunner`).

## GPU sizing (starting point)

| Load | GPU | Notes |
|---|---|---|
| 1–2 concurrent calls (pilot) | 1× L4 24 GB or A10 24 GB | Enough for one real-time talking-head stream at 25 fps |
| 4–8 concurrent calls | 1× L40S 48 GB or A100 80 GB | Batch frames across sessions |
| Fully sovereign (no Sarvam API) | add 1× A100/H100 | Host the Sarvam open-weight LLM with vLLM, plus IndicConformer STT and IndicF5 TTS |

Indian GPU clouds (E2E Networks, Yotta, NxtGen) keep data in India, which matters for
SwaTantra-style regulated customers.

## Evaluating open lip-sync / talking-head models

Measure each on your VikataKavi portrait against these:

- Real-time at 25 fps or more on an L4, with under 200 ms added latency.
- Stable identity across a 3-minute call (no face drift).
- Good Telugu mouth shapes (test with retroflex and aspirated sounds).
- A licence that allows commercial use.

## Cut-over checklist

1. Run the avatar worker on the GPU box and connect it to the same LiveKit project.
2. Run an A/B test: half the calls `runway`, half `selfhosted`. Compare latency,
   lip-sync quality and user ratings.
3. When it's good enough, switch the default, and remove `RUNWAYML_API_SECRET` from Render.
