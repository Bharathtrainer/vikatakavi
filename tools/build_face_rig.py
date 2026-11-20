"""Build the talking-portrait rig for VikataKavi (Anvaya AI Lab).

Runs MediaPipe Face Mesh once on the portrait and writes a JSON "rig" that the
browser uses to animate the face (lip sync, blinks, gaze, head sway).

Usage (needs Python 3.9-3.12):
    pip install "mediapipe==0.10.14" "numpy<2" pillow scipy
    python tools/build_face_rig.py web/static/vikatakavi.png web/static/vikatakavi.rig.json

Re-run it whenever you change the portrait.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict

import numpy as np
from PIL import Image
from scipy.spatial import Delaunay

from mediapipe.python.solutions import face_mesh as fm

# --- MediaPipe landmark index groups -----------------------------------------
UPPER_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
LOWER_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
UPPER_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
LOWER_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379,
             378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
             162, 21, 54, 103, 67, 109]
EYES = {
    # image-left eye (subject's right)
    "r": dict(upper=[33, 246, 161, 160, 159, 158, 157, 173, 133],
              lower=[33, 7, 163, 144, 145, 153, 154, 155, 133],
              brow=[46, 53, 52, 65, 55], iris=468),
    # image-right eye (subject's left)
    "l": dict(upper=[362, 398, 384, 385, 386, 387, 388, 466, 263],
              lower=[362, 382, 381, 380, 374, 373, 390, 249, 263],
              brow=[285, 295, 282, 283, 276], iris=473),
}

T_FACE, T_MOUTH, T_EYE, T_BACK = 0, 1, 2, 3


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def interp_y(contour_pts, x):
    """y of a polyline (sorted by x) at x."""
    pts = sorted(contour_pts, key=lambda p: p[0])
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return float(np.interp(x, xs, ys))


def point_in_poly(pt, poly):
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xin = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < xin:
                inside = not inside
    return inside


def detect(img: Image.Image) -> np.ndarray:
    arr = np.asarray(img.convert("RGB"))
    with fm.FaceMesh(static_image_mode=True, refine_landmarks=True, max_num_faces=1) as mesh:
        res = mesh.process(arr)
    if not res.multi_face_landmarks:
        raise SystemExit("No face found in the portrait.")
    h, w = arr.shape[:2]
    return np.array([[p.x * w, p.y * h] for p in res.multi_face_landmarks[0].landmark])


def face_triangles() -> list[tuple[int, int, int]]:
    adj = defaultdict(set)
    for a, b in fm.FACEMESH_TESSELATION:
        adj[a].add(b)
        adj[b].add(a)
    tris = set()
    for a, b in fm.FACEMESH_TESSELATION:
        for c in adj[a] & adj[b]:
            tris.add(tuple(sorted((a, b, c))))
    return sorted(tris)


def main(src: str, dst: str) -> None:
    img = Image.open(src)
    W, H = img.size
    L = detect(img)                          # 478 x 2 (pixels)
    nface = 468
    P = [tuple(map(float, L[i])) for i in range(nface)]

    cx = float((L[10][0] + L[152][0]) / 2)
    cy = float((L[10][1] + L[152][1]) / 2)
    Fh = float(np.linalg.norm(L[10] - L[152]))
    Fw = float(np.linalg.norm(L[234] - L[454]))
    chin_y = float(L[152][1])
    mouth_cx = float((L[13][0] + L[14][0]) / 2)
    mouth_y = float((L[13][1] + L[14][1]) / 2)
    mouth_hw = float(abs(L[291][0] - L[61][0]) / 2)

    # --- triangles: face mesh minus anything inside the mouth opening ----------
    inner_poly = [P[i] for i in UPPER_INNER] + [P[i] for i in LOWER_INNER[::-1][1:-1]]
    eye_sets = {k: set(v["upper"]) | set(v["lower"]) for k, v in EYES.items()}
    tris: list[list[int]] = []
    for a, b, c in face_triangles():
        cen = ((P[a][0] + P[b][0] + P[c][0]) / 3, (P[a][1] + P[b][1] + P[c][1]) / 3)
        inner = set(UPPER_INNER) | set(LOWER_INNER)
        if {a, b, c} <= inner and point_in_poly(cen, inner_poly):
            continue
        t = T_EYE if any({a, b, c} <= s for s in eye_sets.values()) else T_FACE
        tris.append([a, b, c, t])

    # mouth interior strip between upper and lower inner lip
    for i in range(len(UPPER_INNER) - 1):
        u0, u1 = UPPER_INNER[i], UPPER_INNER[i + 1]
        l0, l1 = LOWER_INNER[i], LOWER_INNER[i + 1]
        for tri in ((u0, u1, l1), (u0, l1, l0)):
            if len(set(tri)) == 3:
                tris.append([*tri, T_MOUTH])

    # --- eyes: fan from the iris centre to the lid contour (gaze + blink) ------
    verts = [list(p) for p in P]
    iris_v = {}
    for k, e in EYES.items():
        vi = len(verts)
        verts.append([float(L[e["iris"]][0]), float(L[e["iris"]][1])])
        iris_v[k] = vi
        ring = e["upper"] + e["lower"][::-1][1:-1]
        for j in range(len(ring)):
            tris.append([vi, ring[j], ring[(j + 1) % len(ring)], T_EYE])

    # --- outer region (head outline, hair, ears, body, background) -------------
    oval = [P[i] for i in FACE_OVAL]
    extra = []
    for s, n in ((1.06, 36), (1.25, 36), (1.6, 32), (2.2, 28), (3.2, 24)):
        for k in range(n):
            ang = 2 * np.pi * k / n
            x = cx + np.cos(ang) * 0.5 * Fw * s
            y = cy + np.sin(ang) * 0.5 * Fh * s
            if 0 < x < W and 0 < y < H:
                extra.append((x, y))
    nb = 14
    for k in range(nb + 1):
        t = k / nb
        extra += [(t * W, 0), (t * W, H), (0, t * H), (W, t * H)]
    extra = list({(round(x, 1), round(y, 1)) for x, y in extra})
    base = len(verts)
    verts += [list(p) for p in extra]

    # Background mesh covers the WHOLE image (including under the face) and is
    # drawn first, so any sliver gaps in the face mesh can never show through.
    pts = np.array(oval + extra)
    dt = Delaunay(pts)
    oval_idx = FACE_OVAL
    back = []
    for s in dt.simplices:
        ids = [oval_idx[i] if i < len(oval) else base + (i - len(oval)) for i in s]
        back.append([*map(int, ids), T_BACK])
    tris = back + tris

    V = np.array(verts)
    n = len(V)

    # --- per-vertex motion weights --------------------------------------------
    def lip_profile(x):
        d = np.clip(np.abs(x - mouth_cx) / mouth_hw, 0, 1)
        return np.sqrt(np.clip(1 - d * d, 0, 1))

    def jaw_profile(x):
        return 1 - smoothstep(0.28 * Fw, 0.46 * Fw, np.abs(x - cx))

    jaw = np.zeros(n)
    up = np.zeros(n)
    lower_lip = set(LOWER_INNER) | set(LOWER_OUTER)
    upper_lip = set(UPPER_INNER) | set(UPPER_OUTER)
    for i, (x, y) in enumerate(V):
        if i in upper_lip:
            up[i] = 0.25 * lip_profile(x)
            continue
        if i < nface and i in lower_lip or y > mouth_y:
            t = np.clip((y - mouth_y) / max(chin_y - mouth_y, 1), 0, 1)
            f = (1 - t) * lip_profile(x) + t * jaw_profile(x)
            if y > chin_y:
                f = jaw_profile(x) * (1 - smoothstep(chin_y, chin_y + 0.3 * Fh, y))
            jaw[i] = f
        elif mouth_y - 0.08 * Fh < y <= mouth_y:
            up[i] = 0.25 * lip_profile(x) * smoothstep(mouth_y - 0.08 * Fh, mouth_y, y)
    for i in (61, 291, 78, 308):        # corners barely move
        jaw[i] *= 0.15

    # rounding ("o"/"u"): mouth corners pull toward the centre
    rnd = np.zeros(n)
    for i, (x, y) in enumerate(V):
        for c in (61, 291):
            d = np.hypot(x - L[c][0], y - L[c][1])
            rnd[i] += np.exp(-(d / (0.07 * Fw)) ** 2) * np.sign(mouth_cx - L[c][0])

    # blink: per-vertex y displacement at full closure
    blink = np.zeros(n)
    eyes_meta = {}
    for k, e in EYES.items():
        up_c = [P[i] for i in e["upper"]]
        lo_c = [P[i] for i in e["lower"]]
        br_c = [P[i] for i in e["brow"]]
        x0 = min(p[0] for p in up_c)
        x1 = max(p[0] for p in up_c)
        poly = up_c + lo_c[::-1][1:-1]
        for i, (x, y) in enumerate(V[:nface]):
            if not (x0 <= x <= x1):
                continue
            hf = smoothstep(0, 0.25, min(x - x0, x1 - x) / (x1 - x0))
            ly = interp_y(lo_c, x)
            uy = interp_y(up_c, x)
            by = interp_y(br_c, x)
            if i in e["lower"]:
                continue
            if i in e["upper"] or point_in_poly((x, y), poly):
                w = 1.0
            elif by < y < uy:
                w = float(smoothstep(by, uy, y))
            else:
                continue
            # close to ~70% of the way down (real blinks meet low)
            target = uy + 0.85 * (ly - uy)
            blink[i] = (target - y) * w * hf if y < target else 0
        vi = iris_v[k]
        x, y = V[vi]
        target = interp_y(up_c, x) + 0.85 * (interp_y(lo_c, x) - interp_y(up_c, x))
        blink[vi] = max(target - y, 0)
        iris = L[e["iris"]]
        r = float(np.mean([np.linalg.norm(L[e["iris"] + j] - iris) for j in range(1, 5)]))
        eyes_meta[k] = {"cx": float(iris[0]), "cy": float(iris[1]), "r": r, "v": iris_v[k]}

    # head: rigid sway weight
    head = np.zeros(n)
    for i, (x, y) in enumerate(V):
        r = np.hypot((x - cx) / (0.5 * Fw), (y - cy) / (0.5 * Fh))
        w = 1 - smoothstep(1.15, 1.9, r)
        if y > chin_y:
            w *= 1 - smoothstep(chin_y, chin_y + 0.35 * Fh, y)
        head[i] = w
    for i in range(base, n):           # image border never moves
        x, y = V[i]
        if x <= 0.5 or y <= 0.5 or x >= W - 0.5 or y >= H - 0.5:
            head[i] = 0

    mv = np.zeros(n)
    for i in LOWER_INNER:
        mv[i] = 1.0
    for i in (78, 308):
        mv[i] = 0.5

    rig = {
        "image": src.split("/")[-1],
        "w": W, "h": H,
        "face": {"cx": cx, "cy": cy, "fw": Fw, "fh": Fh, "chinY": chin_y,
                 "mouthX": mouth_cx, "mouthY": mouth_y, "mouthHW": mouth_hw},
        "eyes": eyes_meta,
        "verts": np.round(V, 2).tolist(),
        "tris": tris,
        "jaw": np.round(jaw, 3).tolist(),
        "up": np.round(up, 3).tolist(),
        "round": np.round(rnd, 3).tolist(),
        "blink": np.round(blink, 2).tolist(),
        "head": np.round(head, 3).tolist(),
        "mv": mv.tolist(),
    }
    with open(dst, "w") as f:
        json.dump(rig, f, separators=(",", ":"))
    print(f"rig: {n} vertices, {len(tris)} triangles -> {dst}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
