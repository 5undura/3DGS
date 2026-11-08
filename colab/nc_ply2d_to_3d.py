"""2DGS PLY -> 표준 3DGS PLY (SuperSplat / Unity 에서 보이게)

2DGS 는 스플랫이 납작한 디스크라 scale 이 2개(scale_0, scale_1)뿐이고,
3DGS 뷰어는 scale_0~2 를 모두 기대함. 세 번째 크기를 아주 작은 값으로 채워서 "두께 거의 0" 으로 만듦.
(scale 은 로그 값으로 저장돼 있음)

numpy 만 있으면 됨 (PC 에서도 실행 가능).
사용: python nc_ply2d_to_3d.py in.ply out.ply [두께_로그값(기본 -14)]
"""
import sys
import numpy as np


def read_ply(path):
    with open(path, "rb") as f:
        names, count = [], 0
        while True:
            line = f.readline().decode("ascii").strip()
            if line.startswith("format") and "binary_little_endian" not in line:
                raise SystemExit("binary_little_endian PLY 만 지원")
            if line.startswith("element vertex"):
                count = int(line.split()[-1])
            elif line.startswith("property"):
                _, t, n = line.split()
                if t != "float":
                    raise SystemExit(f"float 이 아닌 속성: {n} ({t})")
                names.append(n)
            elif line == "end_header":
                break
        data = np.frombuffer(f.read(count * len(names) * 4), dtype="<f4").reshape(count, len(names))
    return names, data


def main():
    src, dst = sys.argv[1], sys.argv[2]
    thick = float(sys.argv[3]) if len(sys.argv) > 3 else -14.0   # exp(-14) = 8e-7

    names, data = read_ply(src)
    if "scale_2" in names:
        print("이미 scale_2 가 있음 (3DGS 형식). 그대로 복사")
        out_names, out = names, data
    else:
        i = names.index("scale_1") + 1          # scale_1 바로 뒤에 끼워 넣음 (표준 순서 유지)
        out_names = names[:i] + ["scale_2"] + names[i:]
        out = np.concatenate([data[:, :i],
                              np.full((len(data), 1), thick, dtype="<f4"),
                              data[:, i:]], axis=1)

    with open(dst, "wb") as f:
        head = "ply\nformat binary_little_endian 1.0\n"
        head += f"element vertex {len(out)}\n"
        head += "".join(f"property float {n}\n" for n in out_names)
        head += "end_header\n"
        f.write(head.encode("ascii"))
        f.write(np.ascontiguousarray(out, dtype="<f4").tobytes())
    print(f"{dst}: {len(out):,} splats (두께 로그값 {thick})")


if __name__ == "__main__":
    main()
