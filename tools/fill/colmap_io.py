"""COLMAP binary reader/writer (cameras, images, points3D) - numpy only"""
import struct, numpy as np

def read_cameras(path):
    cams = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        NP = {0:3,1:4,2:4,3:5,4:8,5:8,6:12,7:5,8:4,9:5,10:12}
        for _ in range(n):
            cid, model, w, h = struct.unpack("<iiQQ", f.read(24))
            p = struct.unpack("<%dd" % NP[model], f.read(8 * NP[model]))
            cams[cid] = dict(model=model, w=w, h=h, params=np.array(p))
    return cams

def qvec2rot(q):
    w, x, y, z = q
    return np.array([[1-2*y*y-2*z*z, 2*x*y-2*w*z, 2*x*z+2*w*y],
                     [2*x*y+2*w*z, 1-2*x*x-2*z*z, 2*y*z-2*w*x],
                     [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x*x-2*y*y]])

def read_images(path, keep_points=False):
    """returns list of dict(id, R, t, cam, name[, xy, pid])"""
    out = []
    data = open(path, "rb").read()
    off = 0
    n = struct.unpack_from("<Q", data, off)[0]; off += 8
    for _ in range(n):
        iid, = struct.unpack_from("<i", data, off); off += 4
        q = struct.unpack_from("<4d", data, off); off += 32
        t = struct.unpack_from("<3d", data, off); off += 24
        cid, = struct.unpack_from("<i", data, off); off += 4
        e = data.index(b"\0", off); name = data[off:e].decode(); off = e + 1
        np2, = struct.unpack_from("<Q", data, off); off += 8
        d = dict(id=iid, q=np.array(q), R=qvec2rot(q), t=np.array(t), cam=cid, name=name)
        if keep_points:
            arr = np.frombuffer(data, dtype=np.dtype([("x","<f8"),("y","<f8"),("p","<i8")]), count=np2, offset=off)
            d["xy"] = np.stack([arr["x"], arr["y"]], 1); d["pid"] = arr["p"].copy()
        off += np2 * 24
        out.append(d)
    return out

def read_points3D(path):
    data = open(path, "rb").read()
    off = 0
    n = struct.unpack_from("<Q", data, off)[0]; off += 8
    ids = np.empty(n, np.int64); xyz = np.empty((n, 3)); rgb = np.empty((n, 3), np.uint8)
    err = np.empty(n); tl = np.empty(n, np.int64)
    tracks = []
    for i in range(n):
        pid, x, y, z, r, g, b, e, L = struct.unpack_from("<Q3d3BdQ", data, off); off += 43 + 8 + 8 - 8
        ids[i] = pid; xyz[i] = (x, y, z); rgb[i] = (r, g, b); err[i] = e; tl[i] = L
        tracks.append(data[off:off + 8 * L]); off += 8 * L
    return dict(ids=ids, xyz=xyz, rgb=rgb, err=err, tl=tl, tracks=tracks)

def write_points3D(path, ids, xyz, rgb, err, tracks):
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(ids)))
        for i in range(len(ids)):
            tr = tracks[i]
            f.write(struct.pack("<Q3d3BdQ", int(ids[i]), *map(float, xyz[i]), *map(int, rgb[i]), float(err[i]), len(tr) // 8))
            f.write(tr)
