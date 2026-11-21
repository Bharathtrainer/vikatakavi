"""Make VikataKavi test clips with the same Sarvam voice used in calls.

Writes audio/te.wav (Telugu) and audio/en.wav (English), 16 kHz mono.
Needs:  export SARVAM_API_KEY=...
"""

import base64
import io
import os
import sys
import wave

import requests

SPEAKER = os.getenv("SARVAM_TTS_SPEAKER", "aditya")
MODEL = os.getenv("SARVAM_TTS_MODEL", "bulbul:v3")

CLIPS = {
    "te": ("te-IN",
           "నమస్కారం! నేను వికటకవిని. ఒకరోజు శ్రీకృష్ణదేవరాయల ఆస్థానానికి ఒక పండితుడు వచ్చాడు. "
           "తన ప్రశ్నకు ఎవరూ జవాబు చెప్పలేరని గర్వంగా అన్నాడు. అప్పుడు నేను చిరునవ్వుతో ఒక చిన్న ప్రశ్న అడిగాను. "
           "ఆ పండితుడు ఆలోచనలో పడ్డాడు, సభ అంతా నవ్వులతో నిండిపోయింది. మీకూ ఒక పొడుపు కథ చెప్పమంటారా?"),
    "en": ("en-IN",
           "Namaskaram! I am VikataKavi from Anvaya AI Lab. Here is a quick riddle. "
           "I have keys but open no locks. I have space but no room. You can enter, but you can never go inside. "
           "What am I? Yes, a keyboard! Shall we try another one?"),
}


def tts(text: str, lang: str, key: str) -> bytes:
    r = requests.post(
        "https://api.sarvam.ai/text-to-speech",
        headers={"api-subscription-key": key, "Content-Type": "application/json"},
        json={
            "text": text,
            "target_language_code": lang,
            "speaker": SPEAKER,
            "model": MODEL,
            "speech_sample_rate": 16000,
            "output_audio_codec": "wav",
        },
        timeout=120,
    )
    if r.status_code != 200:
        sys.exit(f"Sarvam TTS failed ({r.status_code}): {r.text[:300]}")
    chunks = [base64.b64decode(a) for a in r.json()["audios"]]
    if len(chunks) == 1:
        return chunks[0]
    # stitch multiple WAV chunks
    frames, params = [], None
    for c in chunks:
        with wave.open(io.BytesIO(c)) as w:
            params = params or w.getparams()
            frames.append(w.readframes(w.getnframes()))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setparams(params)
        for f in frames:
            w.writeframes(f)
    return buf.getvalue()


def main() -> None:
    key = os.getenv("SARVAM_API_KEY")
    if not key:
        sys.exit("Set SARVAM_API_KEY first:  export SARVAM_API_KEY=...")
    os.makedirs("audio", exist_ok=True)
    for name, (lang, text) in CLIPS.items():
        data = tts(text, lang, key)
        path = f"audio/{name}.wav"
        with open(path, "wb") as f:
            f.write(data)
        with wave.open(path) as w:
            print(f"{path}: {w.getnframes() / w.getframerate():.1f} s @ {w.getframerate()} Hz")


if __name__ == "__main__":
    main()
