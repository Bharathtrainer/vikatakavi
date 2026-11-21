# Ditto benchmark: can VikataKavi get a neural face on our own GPU?

This kit tests [Ditto](https://github.com/antgroup/ditto-talkinghead), a real-time audio-driven talking
head built on LivePortrait, on a rented GPU, using **VikataKavi's portrait** and his **Sarvam Telugu
voice**. It answers three questions:

| Question | What the kit produces |
|---|---|
| Does it look good, including Telugu lip shapes? | `results/te_trt_offline.mp4`, `results/en_trt_offline.mp4` |
| Is it fast enough? | offline frames per second and real-time factor |
| Can it run live in a call? | online mode fed at real speech speed: first-frame latency, lag, "keeps up" |

It takes about 1 hour of GPU time, most of it the first-time install.

## 1. Rent a GPU

Choose **Ubuntu 22.04 with an NVIDIA driver/CUDA image**, and one of:

- **NVIDIA L4 (24 GB)**: best price for this job
- **A10 / A10G (24 GB)** or **A100**: also fine
- T4 works too, but slower. The setup script detects it and builds its own TensorRT engines (adds about 10 minutes).

Indian providers keep data in India: E2E Networks, Yotta (Shakti Cloud), NxtGen. Any cloud works for a benchmark.

## 2. Install (one time, about 20–30 minutes)

```bash
git clone https://github.com/Bharathtrainer/vikatakavi.git
cd vikatakavi/gpu/ditto-benchmark
bash setup.sh
```

## 3. Run

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate ditto
export SARVAM_API_KEY=xxxxxxxx          # type it here; never commit it
python make_test_audio.py               # Telugu + English clips in VikataKavi's voice
python bench.py                         # add --pytorch to also test the PyTorch model
```

## 4. Send back

- `results/results.json`
- the MP4s in `results/` (download with `scp`, or the provider's file browser)

## How to read the numbers

| Metric | Good for live calls |
|---|---|
| `online` → `keeps_up_real_time` | `true` |
| `online` → `first_frame_latency_s` | under ~0.8 s |
| `online` → `lag_p95_s` | under ~0.6 s, and not growing (first vs. last quarter) |
| `offline` → `real_time_factor` | under 1.0 (lower is faster than real time) |

## 5. Stop the GPU

Delete or stop the VM when you're done, so you aren't billed for idle hours.

## Licence note (before any commercial launch)

Ditto's code is Apache-2.0, but its checkpoints include **InsightFace** detector models
(`insightface_det`, `landmark106`), which are licensed for **non-commercial research only**. A benchmark is
fine. For a paid product, we'd replace those detectors (for example with MediaPipe, which our browser
face already uses) before going live.
