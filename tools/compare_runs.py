"""학습 결과 PLY 여러 개를 tools/ply_stats.py 지표로 비교해 표로 출력
사용: python tools/compare_runs.py 기준=F:\\...\\1c_brush\\export_30000.ply LFS=F:\\...\\1c_lichtfeld\\x.ply
      (이름= 을 빼면 파일 이름을 씀)
옵션: --md 파일   마크다운 표를 파일에도 씀 (3_train/README.md 에 붙여 넣을 용도; 이미 있으면 뒤에 덧붙임)
      --base 이름  이 이름을 기준으로 스플랫 수·반투명 비율 차이(%p)를 함께 표시 (기본: 첫 번째)
출력: 콘솔 표 + 마크다운 표 (스플랫 수, 바늘형 >5 / >10, 긴 축 상위 1%, 반투명 <0.5 / <0.2)
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ply_stats import load, stats  # noqa: E402


def man(n):
    """1,640,000 -> '164만', 15,000 -> '1.5만' (README 표 표기)"""
    if n >= 1_000_000:
        return f"{n / 10000:.0f}만"
    return f"{n / 10000:.1f}만" if n >= 10000 else f"{n:,}"


def width(s):
    # 콘솔 정렬용: 한글은 2칸
    return sum(2 if ord(c) > 0x1100 else 1 for c in s)


def pad(s, w):
    return s + " " * (w - width(s))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("plys", nargs="+", help="이름=경로 또는 경로")
    ap.add_argument("--md", help="마크다운 표를 덧붙일 파일")
    ap.add_argument("--base", help="기준 이름 (기본: 첫 번째)")
    a = ap.parse_args()

    rows = []
    for arg in a.plys:
        name, path = arg.split("=", 1) if "=" in arg else (os.path.splitext(os.path.basename(arg))[0], arg)
        if not os.path.isfile(path):
            raise SystemExit(f"파일 없음: {path}")
        rows.append((name, path, stats(load(path))))
    base = rows[0] if not a.base else next((r for r in rows if r[0] == a.base), None)
    if base is None:
        raise SystemExit(f"--base {a.base} 가 목록에 없음")
    b = base[2]

    head = ["이름", "스플랫 수", "바늘형 >5", "바늘형 >10", "긴 축 상위1%", "반투명 <0.5", "반투명 <0.2", f"기준({base[0]}) 대비"]
    table = []
    for name, path, r in rows:
        if r is b:
            vs = "기준"
        else:
            vs = (f"수 {(r['n'] / b['n'] - 1) * 100:+.0f}%, "
                  f"반투명 {r['transl05'] - b['transl05']:+.1f}%p")
        table.append([name, f"{r['n']:,}", f"{r['needle5']:.1f}%", f"{r['needle10']:.1f}%",
                      f"{r['long_p99']:.3f}", f"{r['transl05']:.1f}%", f"{r['transl02']:.1f}%", vs])

    ws = [max(width(x) for x in col) for col in zip(head, *table)]
    print("  ".join(pad(h, w) for h, w in zip(head, ws)))
    print("  ".join("-" * w for w in ws))
    for t in table:
        print("  ".join(pad(x, w) for x, w in zip(t, ws)))

    md = ["| 버전 | 스플랫 수 | 바늘형(>10배) | 큰 스플랫 상위 1% 길이 | 반투명(<0.5) | 반투명(<0.2) | 파일 |",
          "|---|---|---|---|---|---|---|"]
    for name, path, r in rows:
        md.append(f"| {name} | {man(r['n'])} | {r['needle10']:.1f}% | {r['long_p99']:.2f} | "
                  f"{r['transl05']:.0f}% | {r['transl02']:.0f}% | `{os.path.basename(path)}` |")
    print("\n마크다운 (3_train/README.md 용):\n")
    print("\n".join(md))
    if a.md:
        with open(a.md, "a", encoding="utf-8") as f:
            f.write("\n" + "\n".join(md) + "\n")
        print(f"\n덧붙임: {a.md}")


if __name__ == "__main__":
    main()
