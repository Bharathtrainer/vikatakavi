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


# Framing / motion presets for VikataKavi's portrait (shaved crown + tuft + ears).
PRESETS = {
    # Ditto defaults: face-sized crop, full head pose. Crown/ears stay frozen.
    "default": {},
    # Bigger crop moved up: the whole head (crown, tuft, ears) moves together.
    "wholehead": {"crop_scale": 3.0, "crop_vy_ratio": -0.2, "max_size": 1024},
    # Head pose fixed; only lips/eyes/brows move. No sliding, but no nods.
    "exp_only": {"use_d_keys": ["exp"], "max_size": 1024},
    # Lively storyteller: whole head moving, stronger expressions, breathing shoulders + sway.
    "lively": {"crop_scale": 3.0, "crop_vy_ratio": -0.2, "max_size": 1024,
               "use_d_keys": {"exp": 1.25, "pitch": 0.8, "yaw": 0.8, "roll": 0.8, "t": 0.8},
               "body_motion": 1.0},
    # Calmer version of the same.
    "lively_soft": {"crop_scale": 3.0, "crop_vy_ratio": -0.2, "max_size": 1024,
                    "use_d_keys": {"exp": 1.1, "pitch": 0.5, "yaw": 0.5, "roll": 0.5, "t": 0.5},
                    "body_motion": 0.6},
    # Whole head + gentler motion (head pose at 50%).
    "wholehead_soft": {"crop_scale": 3.0, "crop_vy_ratio": -0.2, "max_size": 1024,
                       "use_d_keys": {"exp": 1.0, "pitch": 0.5, "yaw": 0.5, "roll": 0.5, "t": 0.5}},
}


def trt_dir() -> str:
    env = os.path.expanduser("~/.ditto_bench_env")
    if os.path.exists(env):
        for line in open(env):
            if line.startswith("TRT_DIR="):
                return line.strip().split("=", 1)[1]
    return os.path.join(DITTO_DIR, "checkpoints/ditto_trt_Ampere_Plus")


class BodyMotion:
    """Adds breathing (shoulders rise/fall) and a gentle body sway on top of Ditto's frames.

    Ditto animates only the head crop, so shoulders/torso would be frozen. This warps each
    finished frame with a smooth field: shoulders follow a ~15 breaths/min cycle, the whole
    figure sways slightly, and both grow a little while VikataKavi is speaking.
    Uses the browser-face rig (chin/face position) to know where the shoulders are.
    """

    def __init__(self, audio, rig_path, strength=1.0):
        import cv2  # noqa: F401  (opencv is in the ditto env)
        self.strength = strength
        rig = json.load(open(rig_path))
        self.rig = rig
        # speech envelope per video frame (0..1)
        hop = SR // FPS
        n = int(math.ceil(len(audio) / hop))
        rms = np.array([np.sqrt(np.mean(audio[i * hop:(i + 1) * hop] ** 2) + 1e-9) for i in range(n)])
        rms = rms / (np.percentile(rms, 95) + 1e-9)
        env = np.clip(rms, 0, 1)
        k = np.ones(9) / 9                                   # ~0.36 s smoothing
        self.env = np.convolve(env, k, mode="same")

    def _prepare(self, h, w):
        import cv2
        s = w / self.rig["w"]
        f = self.rig["face"]
        chin = f["chinY"] * s
        fh = f["fh"] * s
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        # body weight: 0 above the chin, ramps to 1 over the neck into the shoulders
        t = np.clip((yy - (chin + 0.05 * fh)) / (0.45 * fh), 0, 1)
        self.body_w = (t * t * (3 - 2 * t)).astype(np.float32)
        self.xx, self.yy = xx, yy
        self.h, self.w, self.fh = h, w, fh
        self.pivot = (w / 2.0, h * 1.15)                     # sway pivots below the frame (hips)
        self._cv2 = cv2

    def __call__(self, img, k):
        h, w = img.shape[:2]
        if not hasattr(self, "xx") or self.h != h or self.w != w:
            self._prepare(h, w)
        cv2 = self._cv2
        t = k / FPS
        e = float(self.env[min(k, len(self.env) - 1)]) if len(self.env) else 0.0
        st = self.strength
        # breathing: shoulders lift ~0.5% of frame height, a bit more when talking
        breath = math.sin(2 * math.pi * t / 4.0) * (0.004 + 0.002 * e) * self.h * st
        # sway: slow idle drift + livelier rhythm while speaking (radians)
        ang = (0.0035 * math.sin(2 * math.pi * t / 6.5) + 0.006 * e * math.sin(2 * math.pi * t / 2.2)) * st
        shift_x = (0.003 * math.sin(2 * math.pi * t / 7.3 + 1.0)) * self.w * st
        zoom = 1.025                                         # hides the edges while moving
        px, py = self.pivot
        ca, sa = math.cos(-ang), math.sin(-ang)
        # inverse map: for each output pixel, where to sample in the source
        x0 = (self.xx - self.w / 2) / zoom + self.w / 2
        y0 = (self.yy - self.h / 2) / zoom + self.h / 2
        dx, dy = x0 - px - shift_x, y0 - py
        sx = px + dx * ca - dy * sa
        sy = py + dx * sa + dy * ca + breath * self.body_w
        return cv2.remap(img, sx.astype(np.float32), sy.astype(np.float32),
                         interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


class TimedWriter:
    """Wraps Ditto's video writer to timestamp every finished frame."""

    def __init__(self, inner, body=None):
        self.inner, self.times, self.body = inner, [], body

    def __call__(self, img, fmt="rgb"):
        self.times.append(time.perf_counter())
        if self.body is not None:
            img = self.body(img, len(self.times) - 1)
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


RIG = os.path.join(HERE, "../../web/static/vikatakavi.rig.json")


def _split(setup_kwargs):
    kw = dict(setup_kwargs or {})
    body = kw.pop("body_motion", 0)
    return kw, float(body or 0)


def run_offline(sdk_cls, cfg, root, image, audio_path, out_path, setup_kwargs=None) -> dict:
    setup_kwargs, body = _split(setup_kwargs)
    t = time.perf_counter()
    sdk = sdk_cls(cfg, root)
    load_s = time.perf_counter() - t

    t = time.perf_counter()
    sdk.setup(image, out_path, **(setup_kwargs or {}))
    setup_s = time.perf_counter() - t

    audio, _ = librosa.load(audio_path, sr=SR)
    dur = len(audio) / SR
    sdk.setup_Nd(N_d=math.ceil(dur * FPS))
    sdk.writer = TimedWriter(sdk.writer, BodyMotion(audio, RIG, body) if body else None)

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
    setup_kwargs, body = _split(setup_kwargs)
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
    sdk.writer = TimedWriter(sdk.writer, BodyMotion(audio, RIG, body) if body else None)

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
    ap.add_argument("--modes", default="offline,online", help="offline,online")
    ap.add_argument("--preset", default="default", choices=sorted(PRESETS),
                    help="framing/motion preset (see PRESETS in bench.py)")
    ap.add_argument("--setup", default="", help='extra setup kwargs as JSON, e.g. \'{"crop_scale": 3.0, "emo": 3, "body_motion": 1.5}\'')
    args = ap.parse_args()
    extra = dict(PRESETS[args.preset])
    if args.setup:
        extra.update(json.loads(args.setup))
    if isinstance(extra.get("use_d_keys"), list):
        extra["use_d_keys"] = tuple(extra["use_d_keys"])
    tag = args.preset if not args.setup else args.preset + "-custom"

    image = os.path.abspath(args.image)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    os.chdir(DITTO_DIR)                       # Ditto loads some assets relative to its repo
    from stream_pipeline_offline import StreamSDK as OfflineSDK
    from stream_pipeline_online import StreamSDK as OnlineSDK

    ck = os.path.join(DITTO_DIR, "checkpoints")
    trt = trt_dir()
    for need in (f"{ck}/ditto_cfg/v0.4_hubert_cfg_trt.pkl", trt):
        if not os.path.exists(need):
            sys.exit(f"Missing {need}\nThe model download did not finish - run `bash setup.sh` again until it says 'Setup complete'.")
    variants = [
        ("trt", OfflineSDK, f"{ck}/ditto_cfg/v0.4_hubert_cfg_trt.pkl", trt, "offline", dict(extra)),
        ("trt", OnlineSDK, f"{ck}/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl", trt, "online", dict(extra)),
    ]
    if args.pytorch:
        pt_cfg, pt_root = f"{ck}/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl", f"{ck}/ditto_pytorch"
        variants += [
            ("pytorch", OfflineSDK, pt_cfg, pt_root, "offline", dict(extra)),
            ("pytorch", OnlineSDK, pt_cfg, pt_root, "online", {**extra, "online_mode": True}),
        ]
    modes = set(args.modes.split(","))
    variants = [v for v in variants if v[4] in modes]

    report = {"gpu": gpu_name(), "portrait": os.path.basename(image), "preset": args.preset,
              "setup_kwargs": {k: list(v) if isinstance(v, tuple) else v for k, v in extra.items()},
              "runs": []}
    print("GPU:", report["gpu"])
    for clip in args.clips.split(","):
        audio = os.path.join(HERE, "audio", f"{clip}.wav")
        if not os.path.exists(audio):
            sys.exit(f"missing {audio} - run make_test_audio.py first")
        for backend, cls, cfg, root, mode, kw in variants:
            out = os.path.join(HERE, "results", f"{clip}_{backend}_{mode}_{tag}.mp4")
            label = f"{clip} / {backend} / {mode}"
            print(f"\n=== {label} ===")
            try:
                if mode == "offline":
                    r = run_offline(cls, cfg, root, image, audio, out, setup_kwargs=kw)
                else:
                    r = run_online(cls, cfg, root, image, audio, out, paced=True, setup_kwargs=kw)
                r.update(clip=clip, backend=backend)
                print(json.dumps(r, indent=2))
            except Exception as e:  # keep going so one failure doesn't hide the rest
                import traceback
                traceback.print_exc()
                detail = f"{e!r} {getattr(e, 'filename', '') or ''}".strip()
                r = {"clip": clip, "backend": backend, "mode": mode, "error": detail[:400]}
                print("FAILED:", r["error"])
            report["runs"].append(r)

    path = os.path.join(HERE, "results", f"results_{tag}.json")
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
