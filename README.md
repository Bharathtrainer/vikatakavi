# VikataKavi · వికటకవి
**Anvaya AI Lab: real-time Telugu + English conversational AI with a talking face**

VikataKavi is a witty AI storyteller inspired by the folk tales of Tenali
Ramakrishna. Users open a web page, tap **Start**, and talk to him by voice in
Telugu or English. He answers with a realistic talking face.

```
Browser ──WebRTC──► LiveKit Cloud room ◄──── vikatakavi-agent (Render worker)
   ▲                     ▲                    ├─ Sarvam Saaras  : speech → text (te/en, code-mixed)
   │ token               │ avatar video       │                                          ├─ Sarvam 105B    : persona brain
vikatakavi-web                                ├─ Sarvam Bulbul  : text → Telugu speech
(Render web)                                  └─ Sarvam gemma4  : eyes (camera, on request)

Face: the browser animates the portrait itself (WebGL mesh: lip sync from his voice,
blinks, gaze, head sway). Optional: Runway Characters (paid) or a GPU neural avatar later.
```

## Repository layout

| Path | What it is |
|---|---|
| `web/app.py` | FastAPI: serves the page and issues LiveKit tokens that dispatch the agent |
| `web/static/index.html` | The call page: living portrait, controls, camera, collapsible transcript |
| `web/static/talking-portrait.js` | Browser face animation: lip sync from audio, blinks, gaze, head motion |
| `web/static/vikatakavi.rig.json` | Face rig for the portrait (generated) |
| `tools/build_face_rig.py` | Regenerates the rig when you change the portrait |
| `agent/agent.py` | LiveKit agent: Sarvam STT/LLM/TTS and the pluggable avatar layer |
| `agent/persona.py` | VikataKavi's character, safety rules and greetings |
| `agent/vision.py` | His eyes: keeps the latest camera frame in memory and describes it on demand |
| `render.yaml` | Render Blueprint for both services |
| `docs/portrait-prompt.md` | How to make the realistic portrait and the Runway Character |
| `docs/gpu-migration.md` | Phase 2: moving the face to your own GPU |
| `gpu/ditto-benchmark/` | One-hour GPU test of Ditto (neural talking head) with VikataKavi's portrait and voice |

## Accounts you need

1. **Sarvam AI** (https://dashboard.sarvam.ai): API key for STT, LLM and TTS.
2. **LiveKit Cloud** (https://cloud.livekit.io): free project. Copy the URL, API key and secret.
3. **Runway Developer** (https://dev.runwayml.com): API secret, plus the VikataKavi
   Character (see `docs/portrait-prompt.md`). Add credits to the project that owns
   the Character.
4. **Render** (https://render.com), connected to your GitHub.

## Run locally first

```bash
cp .env.example .env               # fill in your keys
python -m venv .venv && source .venv/bin/activate
pip install -r agent/requirements.txt -r web/requirements.txt
cd agent && python agent.py download-files && python agent.py dev &   # terminal 1
cd web && uvicorn app:app --reload --port 8000                         # terminal 2
# open http://localhost:8000
```

`AVATAR_PROVIDER=none` (the default) uses the free browser-animated portrait. Set it to
`runway` only if you want the hosted photoreal face and have Runway credits.

### Changing the portrait

Replace `web/static/vikatakavi.png`, then rebuild the face rig (Python 3.9–3.12):

```bash
pip install "mediapipe==0.10.14" "numpy<2" pillow scipy
python tools/build_face_rig.py web/static/vikatakavi.png web/static/vikatakavi.rig.json
```

## Deploy on Render

1. Push this folder to a GitHub repo (for example `Bharathtrainer/vikatakavi`).
2. Render → **New → Blueprint** → select the repo. Render reads `render.yaml` and
   creates `vikatakavi-web` and `vikatakavi-agent` in the Singapore region.
3. Enter the secrets when Render asks: the LiveKit trio, `SARVAM_API_KEY`,
   `RUNWAYML_API_SECRET`, `RUNWAY_AVATAR_ID` and `ACCESS_CODE`.
4. Open the `vikatakavi-web` URL, enter the access code and start a call.

**Render costs:** the web service runs on the free plan, which sleeps when idle, so the
first load takes about 30 seconds. The agent worker needs a paid plan, because
background workers aren't free. Upgrade the web service when you demo to others.

## Cost controls built in

- Every call is capped at `MAX_CALL_SECONDS`, 180 by default (Runway's hard limit is 300).
- An access code plus a per-IP hourly limit stop strangers spending your credits.
- The Runway avatar is billed per second of active session (see the Runway pricing
  page). Sarvam is billed per character or second on your Sarvam plan.

## Tuning VikataKavi

| Env var | Default | Notes |
|---|---|---|
| `SARVAM_TTS_SPEAKER` | `aditya` | Try `kabir`, `varun`, `rahul`; pick the most "kavi" voice |
| `SARVAM_TTS_PACE` | `1.0` | 0.9 sounds more storyteller-like |
| `SARVAM_LLM_MODEL` | `sarvam-105b-conversations` | Tuned for live multi-turn chat (plain `sarvam-105b` stalled in testing) |
| `DEFAULT_LANGUAGE` | `te-IN` | `en-IN` for English-first |
| `MAX_CALL_SECONDS` | `180` | |
| `VISION_ENABLED` | `true` | Lets VikataKavi look through the user's camera (camera is off until the user turns it on) |
| `SARVAM_VISION_MODEL` | `gemma4` | Sarvam-hosted vision model used only when he looks |

## Responsible-AI notes

- On screen and in the persona, VikataKavi says he is an AI, not a human and not the
  historical poet.
- The face is an original design. Never use a real person's or actor's likeness
  without written consent.
- Voice and video are processed by Sarvam, LiveKit and Runway. Add a privacy notice
  and get consent before any public launch, as India's DPDP Act requires.
