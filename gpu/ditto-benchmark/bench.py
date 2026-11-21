"""Ditto talking-head benchmark for VikataKavi (Anvaya AI Lab).

Answers three questions on YOUR GPU, with VikataKavi's portrait and Sarvam voice:
  1. Quality  - renders MP4s you can watch (Telugu + English).
  2. Speed    - offline: frames/second and real-time factor.
  3. Live use - online streaming mode fed in real time (like a call):
                first-frame latency and whether it keeps up at 25 fps.

Run (inside `conda activate ditto`):
    python bench.py                      # TensorRT, offline + online, te + en
    python bench.py --pytorch            # also try the PyTorch model
Results go to ./results (MP4s + results.json). Send results.json and the MP4s back.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DITTO_DIR = os.environ.get("DITTO_DIR", os.path.expanduser("~/ditto-talkinghead"))
sys.path.insert(0, DITTO_DIR)

import librosa  # noqa: E402

FPS = 25
SR = 16000
CHUNK = (3, 5, 2)          # Ditto online chunk: 3 past, 5 new, 2 future frames (40 ms each)


def trt_dir() -> str:
    env = os.path.expanduser("~/.ditto_bench_env")
    if os.path.exists(env):
        for line in open(env):
            if line.startswith("TRT_DIR="):
                return line.strip().split("=", 1)[1]
    return os.path.join(DITTO_DIR, "checkpoints/ditto_trt_Ampere_Plus")


class TimedWriter:
    """Wraps Ditto's video writer to timestamp every finished frame."""

    def __init__(self, inner):
        self.inner, self.times = inner, []

    def __call__(self, img, fmt="rgb"):
        self.times.append(time.perf_counter())
        return self.inner(img, fmt=fmt)

    def close(self):
        return self.inner.close()


def mux(sdk, audio_path: str, out_path: str) -> None:
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", sdk.tmp_output_path, "-i", audio_path,
                    "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", out_path], check=True)
    try:
        os.remove(sdk.tmp_output_path)
    except OSError:
        pass


def gpu_name() -> str:
    try:
        return subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                                       text=True).strip()
    except Exception:
        return "unknown"


def run_offline(sdk_cls, cfg, root, image, audio_path, out_path) -> dict:
    t = time.perf_counter()
    sdk = sdk_cls(cfg, root)
    load_s = time.perf_counter() - t

    t = time.perf_counter()
    sdk.setup(image, out_path)
    setup_s = time.perf_counter() - t

    audio, _ = librosa.load(audio_path, sr=SR)
    dur = len(audio) / SR
    sdk.setup_Nd(N_d=math.ceil(dur * FPS))
    sdk.writer = TimedWriter(sdk.writer)

    t0 = time.perf_counter()
    sdk.audio2motion_queue.put(sdk.wav2feat.wav2feat(audio))
    sdk.close()
    gen_s = time.perf_counter() - t0
    mux(sdk, audio_path, out_path)
    n = len(sdk.writer.times)
    return {
        "mode": "offline", "audio_s": round(dur, 2), "frames": n,
        "model_load_s": round(load_s, 2), "avatar_setup_s": round(setup_s, 2),
        "generate_s": round(gen_s, 2), "fps": round(n / gen_s, 1) if gen_s else None,
        "real_time_factor": round(gen_s / dur, 3), "video": os.path.basename(out_path),
    }


def run_online(sdk_cls, cfg, root, image, audio_path, out_path, paced=True, setup_kwargs=None) -> dict:
    t = time.perf_counter()
    sdk = sdk_cls(cfg, root)
    load_s = time.perf_counter() - t

    t = time.perf_counter()
    sdk.setup(image, out_path, **(setup_kwargs or {}))
    setup_s = time.perf_counter() - t
    if not sdk.online_mode:
        raise RuntimeError("config is not in online mode")

    audio, _ = librosa.load(audio_path, sr=SR)
    dur = len(audio) / SR
    sdk.setup_Nd(N_d=math.ceil(dur * FPS))
    sdk.writer = TimedWriter(sdk.writer)

    padded = np.concatenate([np.zeros((CHUNK[0] * 640,), dtype=np.float32), audio], 0)
    split_len = int(sum(CHUNK) * 0.04 * SR) + 80
    step = CHUNK[1] * 640                      # 200 ms of new audio per chunk
    t0 = time.perf_counter()
    for idx, i in enumerate(range(0, len(padded), step)):
        if paced:                              # feed exactly as fast as speech arrives in a call
            wait = t0 + idx * step / SR - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
        chunk = padded[i:i + split_len]
        if len(chunk) < split_len:
            chunk = np.pad(chunk, (0, split_len - len(chunk)))
        sdk.run_chunk(chunk, CHUNK)
    sdk.close()
    total_s = time.perf_counter() - t0
    mux(sdk, audio_path, out_path)

    times = sdk.writer.times
    n = len(times)
    # lag of each frame vs. the moment it should be shown if playback started at t0
    lags = [times[k] - (t0 + k / FPS) for k in range(n)]
    q = max(n // 4, 1)
    res = {
        "mode": "online-paced" if paced else "online-max-speed",
        "audio_s": round(dur, 2), "frames": n,
        "model_load_s": round(load_s, 2), "avatar_setup_s": round(setup_s, 2),
        "first_frame_latency_s": round(times[0] - t0, 3) if n else None,
        "lag_median_s": round(statistics.median(lags), 3) if n else None,
        "lag_p95_s": round(sorted(lags)[int(0.95 * (n - 1))], 3) if n else None,
        "lag_first_quarter_s": round(statistics.median(lags[:q]), 3) if n else None,
        "lag_last_quarter_s": round(statistics.median(lags[-q:]), 3) if n else None,
        "total_s": round(total_s, 2), "video": os.path.basename(out_path),
    }
    if n:
        drift = res["lag_last_quarter_s"] - res["lag_first_quarter_s"]
        res["keeps_up_real_time"] = bool(drift < 0.5)
        if not paced:
            res["fps"] = round(n / total_s, 1)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=os.path.join(HERE, "../../web/static/vikatakavi.png"))
    ap.add_argument("--clips", default="te,en")
    ap.add_argument("--pytorch", action="store_true", help="also benchmark the PyTorch model")
    args = ap.parse_args()

    image = os.path.abspath(args.image)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    os.chdir(DITTO_DIR)                       # Ditto loads some assets relative to its repo
    from stream_pipeline_offline import StreamSDK as OfflineSDK
    from stream_pipeline_online import StreamSDK as OnlineSDK

    ck = os.path.join(DITTO_DIR, "checkpoints")
    trt = trt_dir()
    variants = [
        ("trt", OfflineSDK, f"{ck}/ditto_cfg/v0.4_hubert_cfg_trt.pkl", trt, "offline", {}),
        ("trt", OnlineSDK, f"{ck}/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl", trt, "online", {}),
    ]
    if args.pytorch:
        pt_cfg, pt_root = f"{ck}/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl", f"{ck}/ditto_pytorch"
        variants += [
            ("pytorch", OfflineSDK, pt_cfg, pt_root, "offline", {}),
            ("pytorch", OnlineSDK, pt_cfg, pt_root, "online", {"online_mode": True}),
        ]

    report = {"gpu": gpu_name(), "portrait": os.path.basename(image), "runs": []}
    print("GPU:", report["gpu"])
    for clip in args.clips.split(","):
        audio = os.path.join(HERE, "audio", f"{clip}.wav")
        if not os.path.exists(audio):
            sys.exit(f"missing {audio} - run make_test_audio.py first")
        for backend, cls, cfg, root, mode, kw in variants:
            out = os.path.join(HERE, "results", f"{clip}_{backend}_{mode}.mp4")
            label = f"{clip} / {backend} / {mode}"
            print(f"\n=== {label} ===")
            try:
                if mode == "offline":
                    r = run_offline(cls, cfg, root, image, audio, out)
                else:
                    r = run_online(cls, cfg, root, image, audio, out, paced=True, setup_kwargs=kw)
                r.update(clip=clip, backend=backend)
                print(json.dumps(r, indent=2))
            except Exception as e:  # keep going so one failure doesn't hide the rest
                r = {"clip": clip, "backend": backend, "mode": mode, "error": repr(e)[:400]}
                print("FAILED:", r["error"])
            report["runs"].append(r)

    path = os.path.join(HERE, "results", "results.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved {path}")
    print("\nSummary:")
    for r in report["runs"]:
        if "error" in r:
            print(f"  {r['clip']:>2} {r['backend']:>7} {r['mode']:<8} ERROR")
        elif r["mode"] == "offline":
            print(f"  {r['clip']:>2} {r['backend']:>7} offline  {r['fps']} fps, RTF {r['real_time_factor']}")
        else:
            print(f"  {r['clip']:>2} {r['backend']:>7} online   first frame {r['first_frame_latency_s']} s, "
                  f"lag p95 {r['lag_p95_s']} s, keeps up: {r.get('keeps_up_real_time')}")


if __name__ == "__main__":
    main()
