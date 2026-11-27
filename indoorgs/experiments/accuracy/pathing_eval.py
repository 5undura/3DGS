"""동선 -> 위치 제약(priors) 정확도. 사람이 복도 + 방(ㄱ자)을 걷는 것을 흉내 내고, 손으로 그린 선·고정점으로
build_priors 를 돌려 프레임별 위치 오차를 잰다.

장면 (도면 좌표 m): 복도 y=0 (x 0~14, 폭 2.4), 문 (10, 1.2), 방 x 7~13, y 1.2~7.
걷기: (0.5,0) -> 복도 중간 2초 멈춤(고정점 없음) -> 모퉁이 (10,0) 1초 멈춤 -> 문 (10,1.2) 0.5초 -> 방 (10,4.5) 1초 -> (7.5,4.5).
속도 0.8 m/s x (0.7~1.3 천천히 변함), 모퉁이는 반경 0.4m 로 둥글게 돎. 카메라 높이 1.2~1.5m (+출렁임 2cm).
그린 선: 실제 길 + 옆으로 최대 0.3m 천천히 변하는 어긋남 + 손떨림(σ 3cm), GUI 와 같은 clean_stroke(eps 0.05).
고정점: 시작·모퉁이·문·방 모퉁이·끝. 클릭 위치 σ 0.15m, 시간 오차 ±0.5s (균등), 멈춘 곳이면 멈춘 동안 아무 때.

python experiments/accuracy/pathing_eval.py [--runs 300]
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np

from common import OUT, fmt, stats
from indoorgs import schemas
from indoorgs.pathing import PathModel, build_priors, clean_stroke

FPS = 30.0
CORNERS = [(0.5, 0.0), (10.0, 0.0), (10.0, 1.2), (10.0, 4.5), (7.5, 4.5)]
STOPS = {1: 1.0, 2: 0.5, 3: 1.0}          # 꼭짓점 번호 -> 멈춤(초)
PAUSE_X = 5.0                            # 복도 중간 2초 멈춤 (고정점 안 찍음)
PAUSE_S = 2.0


def rounded_path(pts, r=0.4, step=0.02):
    """꼭짓점을 반경 r 원호로 둥글린 선을 step 간격 점으로. 각 꼭짓점에 가장 가까운 점 번호도."""
    P = [np.array(p, float) for p in pts]
    out, idx = [P[0]], [0]
    for i in range(1, len(P) - 1):
        a, b, c = P[i - 1], P[i], P[i + 1]
        d1 = (b - a) / np.linalg.norm(b - a); d2 = (c - b) / np.linalg.norm(c - b)
        rr = min(r, np.linalg.norm(b - a) / 2.2, np.linalg.norm(c - b) / 2.2)
        p1, p2 = b - d1 * rr, b + d2 * rr
        q0 = out[-1]
        for u in np.linspace(0, 1, max(2, int(np.linalg.norm(p1 - q0) / step)))[1:]:
            out.append((1 - u) * q0 + u * p1)
        # 2차 베지어로 모퉁이
        for u in np.linspace(0, 1, 12)[1:]:
            out.append((1 - u) ** 2 * p1 + 2 * u * (1 - u) * b + u ** 2 * p2)
        idx.append(len(out) - 6)
    q0 = out[-1]
    for u in np.linspace(0, 1, max(2, int(np.linalg.norm(P[-1] - q0) / step)))[1:]:
        out.append((1 - u) * q0 + u * P[-1])
    idx.append(len(out) - 1)
    return np.array(out), idx


def cumlen(X):
    return np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(X, axis=0), axis=1))])


def smooth_noise(rng, n_terms, periods, amp, x):
    v = np.zeros_like(x, float)
    for _ in range(n_terms):
        T = rng.uniform(*periods); ph = rng.uniform(0, 2 * np.pi)
        v += np.sin(2 * np.pi * x / T + ph)
    return amp * v / max(np.abs(v).max(), 1e-9)


def simulate_walk(rng, speed=0.8, var=0.3):
    """-> (시간 배열, 위치 배열 (N,2), 높이 배열, 꼭짓점 도착/출발 시간 [(t_in, t_out)], 실제 길 X, 꼭짓점 번호)"""
    X, vidx = rounded_path(CORNERS)
    S = cumlen(X)
    s_v = [S[i] for i in vidx]
    s_pause = float(np.interp(PAUSE_X, X[: vidx[1], 0], S[: vidx[1]]))
    stops = sorted([(s_v[k], d, k) for k, d in STOPS.items()] + [(s_pause, PAUSE_S, -1)])
    dt = 1.0 / FPS
    t, s, ts, ss = 0.0, 0.0, [], []
    nxt = 0
    events = {0: (0.0, 0.0)}
    # 속도 변화: 시간에 대한 매끈한 잡음
    tt = np.arange(0, 200, dt)
    mult = 1 + smooth_noise(rng, 4, (3, 10), var, tt)
    i = 0
    while s < S[-1]:
        ts.append(t); ss.append(s)
        if nxt < len(stops) and s >= stops[nxt][0]:
            s0, d, k = stops[nxt]
            t_in = t
            for _ in range(int(round(d / dt))):
                t += dt; i += 1; ts.append(t); ss.append(s)
            if k >= 0:
                events[k] = (t_in, t)
            nxt += 1
        s = min(S[-1], s + speed * mult[i] * dt)
        t += dt; i += 1
    ts.append(t); ss.append(S[-1])
    events[len(CORNERS) - 1] = (t, t)
    ts, ss = np.array(ts), np.array(ss)
    P = np.stack([np.interp(ss, S, X[:, 0]), np.interp(ss, S, X[:, 1])], 1)
    h0 = rng.uniform(1.2, 1.5)
    z = h0 + 0.02 * np.sin(2 * np.pi * ts / 0.55 + rng.uniform(0, 6))
    return ts, P, z, events, X


def draw_stroke(rng, X, dev=0.3, jitter=0.03):
    S = cumlen(X)
    T = np.gradient(X, axis=0); T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-9)
    Nn = np.stack([-T[:, 1], T[:, 0]], 1)
    off = smooth_noise(rng, 3, (4, 9), rng.uniform(0.5, 1.0) * dev, S)
    D = X + Nn * off[:, None]
    # 마우스 표본: 5cm 간격 + 손떨림
    Sd = cumlen(D)
    q = np.arange(0, Sd[-1], 0.05)
    pts = np.stack([np.interp(q, Sd, D[:, 0]), np.interp(q, Sd, D[:, 1])], 1)
    pts += rng.normal(0, jitter, pts.shape)
    pts = np.vstack([pts, D[-1]])
    return clean_stroke([tuple(p) for p in pts], eps=0.05)


def make_anchors(rng, events, use=(0, 1, 2, 3, 4), click=0.15, terr=0.5, extra_pause=None, two_per_stop=False):
    out = []
    for k in use:
        t_in, t_out = events[k]
        if two_per_stop and t_out > t_in and 0 < k < len(CORNERS) - 1:
            # CONTRACT: 멈춤 = 같은 위치 고정점 두 개 (멈춘 시작·끝, 각각 시간 오차)
            p = np.array(CORNERS[k]) + rng.normal(0, click, 2)
            for tt in (t_in, t_out):
                out.append({"t": round(float(tt + rng.uniform(-terr, terr)), 3), "point": [float(p[0]), float(p[1])]})
            continue
        t_true = rng.uniform(t_in, t_out) if t_out > t_in else t_in
        t = t_true + rng.uniform(-terr, terr)
        if k == 0:
            t = 0.0
        if k == len(CORNERS) - 1:
            t = max(t, out[-1]["t"] + 0.1)
        p = np.array(CORNERS[k]) + rng.normal(0, click, 2)
        out.append({"t": round(float(max(t, 0.0)), 3), "point": [float(p[0]), float(p[1])]})
    if extra_pause is not None:
        t, p = extra_pause
        out.append({"t": t, "point": p})
    return out


def run_once(rng, *, dev=0.3, jitter=0.03, click=0.15, terr=0.5, var=0.3, use=(0, 1, 2, 3, 4),
             anchor_pause=False, step_frames=15, two_per_stop=False):
    ts, P, z, events, X = simulate_walk(rng, var=var)
    stroke = draw_stroke(rng, X, dev, jitter) if dev is not None else [tuple(p) for p in X[::5]] + [tuple(X[-1])]
    extra = None
    if anchor_pause:
        # 멈춘 곳(x=5)을 지나는 동안 찍은 고정점 = 멈춘 시간 가운데
        k = np.where(np.all(np.isclose(np.diff(P, axis=0), 0), axis=1))[0]
        sel = k[(P[k, 0] > PAUSE_X - 0.1) & (P[k, 0] < PAUSE_X + 0.1) & (np.abs(P[k, 1]) < 0.5)]
        if len(sel):
            tp = float(np.median(ts[sel])) + rng.uniform(-terr, terr)
            pp = P[sel[0]] + rng.normal(0, click, 2)
            extra = (round(tp, 3), [float(pp[0]), float(pp[1])])
            if two_per_stop:
                t0, t1 = float(ts[sel].min()), float(ts[sel].max())
                extra = [(round(t0 + rng.uniform(-terr, terr), 3), extra[1]), (round(t1 + rng.uniform(-terr, terr), 3), extra[1])]
    anchors = make_anchors(rng, events, use, click, terr, None if isinstance(extra, list) else extra, two_per_stop)
    if isinstance(extra, list):
        anchors += [{"t": t, "point": p} for t, p in extra]
    cam = schemas.parse_path({"version": 1, "stroke": [list(p) for p in stroke], "anchors": anchors})
    fidx = np.arange(0, len(ts), step_frames)
    names = [f"f_{i + 1:05d}_{int(round(ts[j] * FPS)):06d}.jpg" for i, j in enumerate(fidx)]
    pri, warns = build_priors(cam, names, FPS)
    xyz = np.array([f["xyz"] for f in pri["frames"]])
    # build_priors 는 이름순 정렬 -> fidx 순서와 같음 (번호가 커지는 순)
    exy = xyz[:, :2] - P[fidx]
    ez = xyz[:, 2] - z[fidx]
    return exy, ez, warns


def scenario(name, runs, seed, **kw):
    rng = np.random.default_rng(seed)
    exy, ez, nwarn = [], [], 0
    for _ in range(runs):
        a, b, w = run_once(rng, **kw)
        exy.append(a); ez.append(b); nwarn += bool(w)
    exy = np.concatenate(exy); ez = np.concatenate(ez)
    r = np.linalg.norm(exy, axis=1)
    res = {"scenario": name, "xy": stats(r), "z": stats(np.abs(ez)),
           "rms_axis_xy": round(float(np.sqrt((exy ** 2).mean())), 3), "rms_z": round(float(np.sqrt((ez ** 2).mean())), 3),
           "frac_xy_gt_1m": round(float((r > 1.0).mean()), 4),
           "frac_3d_gt_1m": round(float((np.sqrt(r ** 2 + ez ** 2) > 1.0).mean()), 4),
           "runs_with_warning": nwarn}
    print(f"{name:34s} xy[{fmt(res['xy'])}] rms/axis={res['rms_axis_xy']} z_rms={res['rms_z']} "
          f">1m(3D)={res['frac_3d_gt_1m']*100:.1f}% warn_runs={nwarn}")
    return res


def edge_cases():
    res = {}
    line = [[0, 0], [10, 0], [10, 5]]
    base = [{"t": 0, "point": [0, 0]}, {"t": 10, "point": [10, 0]}, {"t": 15, "point": [10, 5]}]
    names = [f"f_{i + 1:05d}_{i * 30:06d}.jpg" for i in range(16)]

    def err(cam_json, truth):
        cam = schemas.parse_path(cam_json)
        pri, w = build_priors(cam, names, 30.0)
        xy = np.array([f["xyz"][:2] for f in pri["frames"]])
        return float(np.abs(xy - truth).max()), w

    tr = np.array([(min(t, 10), max(0, t - 10)) for t in range(16)], float)
    e, w = err({"version": 1, "stroke": line, "anchors": base}, tr)
    res["normal"] = {"max_err": round(e, 3), "warnings": w}
    e, w = err({"version": 1, "stroke": line, "anchors": base[::-1]}, tr)
    res["anchors_listed_out_of_order"] = {"max_err": round(e, 3), "warnings": w}
    e, w = err({"version": 1, "stroke": line[::-1], "anchors": base}, tr)
    res["stroke_reversed"] = {"max_err": round(e, 3), "warnings": w}
    late = [dict(a) for a in base]; late[-1]["t"] = 40
    e, w = err({"version": 1, "stroke": line, "anchors": late}, tr)
    res["last_anchor_beyond_video(t=40,video=15s)"] = {"max_err": round(e, 3), "warnings": w}
    # 왕복: 복도 끝까지 갔다가 같은 길로 돌아옴 (방이 막다른 곳이면 흔함). 돌아오는 길은 0.3m 옆.
    ob = [[0, 0], [10, 0], [10, 0.3], [0, 0.3]]
    oba = [{"t": 0, "point": [0, 0]}, {"t": 10, "point": [10, 0]}, {"t": 15, "point": [5, 0.3]},
           {"t": 20, "point": [0, 0.3]}]
    names2 = [f"f_{i + 1:05d}_{i * 30:06d}.jpg" for i in range(21)]
    cam = schemas.parse_path({"version": 1, "stroke": ob, "anchors": oba})
    pri, w = build_priors(cam, names2, 30.0)
    xy = np.array([f["xyz"][:2] for f in pri["frames"]])
    tr2 = np.array([(t, 0) if t <= 10 else (10 - (t - 10), 0.3) for t in range(21)], float)
    # 0.3m 꺾는 구간 길이 무시한 대략 정답 -> 오차 0.3 근처면 정상
    res["out_and_back_same_corridor"] = {"max_err": round(float(np.abs(xy - tr2).max()), 3), "warnings": w,
                                         "xy_at_t13": [round(v, 2) for v in xy[13]], "truth_t13": list(tr2[13])}
    # 방 한 바퀴 돌고 문(시작점) 옆 off m 로 나옴. 클릭 σ0.15. t=24 의 위치 오차 (정답 (1, off))
    rng = np.random.default_rng(0)
    for off in (0.1, 0.3, 0.6):
        es, ws = [], 0
        for _ in range(300):
            st = [[0, 0], [0, 5], [5, 5], [5, off], [0.0, off]]
            p = np.array([2.5, off]) + rng.normal(0, 0.15, 2)
            q = np.array([0.0, off]) + rng.normal(0, 0.15, 2)
            m = PathModel(schemas.parse_path({"version": 1, "stroke": st, "anchors": [
                {"t": 0, "point": [0, 0]}, {"t": 15, "point": [5, 5]}, {"t": 22.5, "point": list(p)},
                {"t": 25, "point": list(q)}]}))
            x, y, _ = m.position(24)
            es.append(math.hypot(x - 1, y - off)); ws += bool(m.warnings)
        res[f"room_loop_exit_{off}m_beside_entry"] = {"max_err": round(max(es), 3), "median_err": round(float(np.median(es)), 3),
                                                     "warnings": [f"{ws}/300 runs warned"]}
    for k, v in res.items():
        print(f"edge {k:42s} max_err={v['max_err']} warnings={len(v['warnings'])} {v['warnings'][:1]}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=300)
    a = ap.parse_args()
    R = a.runs
    res = [
        scenario("ideal(stroke=truth,t exact,no var)", R, 1, dev=None, click=0.0, terr=0.0, var=0.0),
        scenario("speed var only", R, 2, dev=None, click=0.0, terr=0.0),
        scenario("stroke dev only", R, 3, click=0.0, terr=0.0, var=0.0),
        scenario("anchor time err only", R, 4, dev=None, click=0.0, var=0.0),
        scenario("ALL (5 anchors)", R, 5),
        scenario("ALL, no door anchor (4)", R, 6, use=(0, 1, 3, 4)),
        scenario("ALL + anchor at 2s pause (6)", R, 7, anchor_pause=True),
        scenario("ALL, start+end only (2)", R, 8, use=(0, 4)),
        scenario("ALL, every stop+pause = 2 anchors (10)", R, 9, anchor_pause=True, two_per_stop=True),
    ]
    out = {"scenarios": res, "edge_cases": edge_cases()}
    OUT.mkdir(exist_ok=True)
    (OUT / "pathing.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
