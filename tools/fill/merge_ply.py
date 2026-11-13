"""여러 3DGS PLY 를 하나로 합침 (속성 이름 기준, 첫 파일의 속성 순서를 따름)
사용: python merge_ply.py out.ply a.ply b.ply ...
"""
import sys, numpy as np

def read(path):
    with open(path, "rb") as f:
        names, n, head = [], 0, []
        while True:
            l = f.readline().decode("ascii").strip(); head.append(l)
            if l.startswith("element vertex"): n = int(l.split()[-1])
            elif l.startswith("property"): names.append(l.split()[-1])
            elif l == "end_header": break
        offset = f.tell()
    # 파일 객체에서 바로 읽으면 버퍼 위치가 어긋나 0개가 읽히는 경우가 있음 -> 헤더 길이로 다시 엶
    a = np.fromfile(path, dtype="<f4", count=n * len(names), offset=offset)
    if a.size != n * len(names):
        raise SystemExit(f"{path}: {a.size} 값만 읽힘 (필요 {n * len(names)}). 구글 드라이브 파일이면 PC 에 내려받아졌는지 확인")
    return names, a.reshape(n, len(names)), [h for h in head if h.startswith("comment")]

out, srcs = sys.argv[1], sys.argv[2:]
names, first, comments = read(srcs[0]); parts = [first]
for s in srcs[1:]:
    nm, a, _ = read(s)
    b = np.zeros((len(a), len(names)), "<f4")
    for i, k in enumerate(names):
        if k in nm: b[:, i] = a[:, nm.index(k)]
    parts.append(b)
A = np.concatenate(parts)
with open(out, "wb") as f:
    h = "ply\nformat binary_little_endian 1.0\n" + "".join(c + "\n" for c in comments)
    h += "element vertex %d\n" % len(A) + "".join("property float %s\n" % k for k in names) + "end_header\n"
    f.write(h.encode("ascii")); A.astype("<f4").tofile(f)
print(out, f"{len(A):,} splats", [len(p) for p in parts])
