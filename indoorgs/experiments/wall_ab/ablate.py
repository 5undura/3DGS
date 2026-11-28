"""wall_ab 한 가지씩 바꾸기: snap 하위 동작, cleanup 단계별, repair 옵션이 각각 얼마나 돕거나 해치는지.

  python ablate.py --out <S> --variant B [--planes fit|oracle]
결과 <S>/<v>_abl[_oracle]/abl.json + 로그 한 줄씩. run.py prep/train/post 뒤에 돌림.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import run as R
import scene


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--variant", default="B")
    ap.add_argument("--planes", default="fit", choices=["fit", "oracle"])
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--only", default="", help="쉼표로 실험 이름 (빈 값 = 전부)")
    a = ap.parse_args(argv)
    import torch
    torch.set_num_threads(2)
    from indoorgs.walls import snap, cleanup, repair, facetex
    S = Path(a.out)
    planes, trusted, plan, model, cams = R.load_ctx(S)
    tex_dir = S / "tex"
    if a.planes == "oracle":
        planes = trusted = scene.true_planes()
        plan = scene.true_plan()
        tex_dir = S / "tex_oracle"
    D = S / (f"{a.variant}_abl" + ("_oracle" if a.planes == "oracle" else ""))
    D.mkdir(exist_ok=True)
    raw = S / a.variant / R.FILES["raw"]
    snapped = S / (a.variant + ("_oracle" if a.planes == "oracle" else "")) / R.FILES["snap"]
    clean = S / (a.variant + ("_oracle" if a.planes == "oracle" else "")) / R.FILES["clean"]

    def sn(**kw):
        return lambda src, dst: snap.snap_file(src, dst, trusted, plan, **{"band": 0.03, "color_tol": 0.12, **kw})

    def cl(steps, **kw):
        return lambda src, dst: cleanup.cleanup_file(src, dst, planes=planes, plan=plan, cameras=cams,
                                                     model=kw.pop("model", model), steps=steps, **kw)

    def rp(tex=False, **kw):
        def f(src, dst):
            t = facetex.load_textures(tex_dir, trusted) if tex else None
            return repair.repair_file(src, dst, trusted, cameras=cams, texture=t,
                                      **{"min_cover": 0.5, "max_hole_m2": 4.0, **kw})
        return f

    exps = [
        # snap 하위 동작 (raw 에서)
        ("snap_default", raw, sn()),
        ("snap_no_remove", raw, sn(remove_behind=False)),
        ("snap_remove_only", raw, sn(move=False, flatten=False)),
        ("snap_remove_nocap", raw, sn(move=False, flatten=False, max_remove_frac=None, max_face_frac=None)),
        ("snap_move_only", raw, sn(remove_behind=False, flatten=False)),
        ("snap_flatten_only", raw, sn(remove_behind=False, move=False)),
        ("snap_no_color", raw, sn(color_tol=None)),
        ("snap_count_color", raw, sn(color_weighting="count")),          # 예전 색 기준 (버그 1) 비교용
        ("snap_tex_color", raw, lambda src, dst: sn(color_texture=facetex.load_textures(tex_dir, trusted))(src, dst)),
        ("snap_move_only_count_color", raw, sn(remove_behind=False, flatten=False, color_weighting="count")),
        # cleanup 단계 하나씩 (raw 에서, snap 없이)
        ("clean_floaters", raw, cl(("floaters",))),
        ("clean_floaters_nomodel", raw, cl(("floaters",), model=None)),
        ("clean_huge", raw, cl(("huge",))),
        ("clean_needles", raw, cl(("needles",))),
        ("clean_sh", raw, cl(("sh",))),
        ("clean_all_raw", raw, cl(("floaters", "huge", "needles", "sh"))),
        ("clean_noair_raw", raw, cl(("floaters", "huge", "needles", "sh"), floaters={"air_gap": 1e9})),
        # repair 옵션 (raw 에서 바로 - snap/cleanup 영향 빼고)
        ("repair_raw", raw, rp()),
        ("repair_tex_raw", raw, rp(tex=True)),
        ("repair_tex_zero_raw", raw, rp(tex=True, texture_rest="zero")),
        ("repair_raw_noray", raw, rp(ray_check=False)),                  # 예전 덮임 (버그 5) 비교용
        ("repair_tex_raw_noray", raw, rp(tex=True, ray_check=False)),
    ]
    if a.only:
        keep = set(a.only.split(","))
        exps = [e for e in exps if e[0] in keep]
    test, novel, gt = R.eval_views(S, a.res)
    tp = scene.true_planes()
    out_f = D / "abl.json"
    res = json.loads(out_f.read_text(encoding="utf-8")) if out_f.exists() else {}
    for name, src, fn in exps:
        dst = D / f"{name}.ply"
        t = time.time()
        info = fn(src, dst)
        tsec = time.time() - t
        e = R.evaluate_ply(dst, test, novel, gt, tp, a.res)
        e.pop("views")
        brief = {k: info.get(k) for k in ("n_removed", "n_moved", "n_flattened", "n_added") if k in info}
        if "floaters" in info:
            brief["floaters"] = info["floaters"]["removed_by"]
        e["post_info"] = brief
        e["post_seconds"] = round(tsec, 1)
        res[name] = R.jsonable(e)
        R.log(f"{a.variant}/{a.planes} {name:24s} test {e['test']['psnr']:.2f} novel {e['novel']['psnr']:.2f} "
              f"graze {e['graze_views']['psnr']:.2f} gpx {e['novel']['graze_px_psnr']:.2f} leak {e['novel']['leak'] * 100:.2f}% "
              f"thick {e['flat']['thick_wall_cm']:.2f} n {e['stats']['count']} tr {e['stats']['translucent_ratio']:.3f} "
              f"nd {e['stats']['needle_ratio']:.3f} {brief}")
        out_f.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
