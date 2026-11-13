import numpy as np, pickle, os, shutil
from scipy.spatial import cKDTree
from colmap_io import read_points3D, write_points3D
b=np.load("backdrop.npz"); P=b["P"]; N=b["N"]; K=b["K"]; Pw=b["Pw"]; Nw=b["Nw"]; col=b["col"].copy(); SP=float(b["SP"])
f=np.load("frame2.npz"); ax1,ax2,up=f["ax1"],f["ax2"],f["up"]
# 색 없는 샘플은 가장 가까운 색 있는 샘플로 채움
ok=~np.isnan(col[:,0]); _,j=cKDTree(Pw[ok]).query(Pw[~ok]); col[~ok]=col[ok][j]
OUT="/mnt/user-data/outputs/fill"; os.makedirs(OUT+"/sparse/0",exist_ok=True)

# ---- 1) 바탕층 스플랫 PLY (Brush 와 같은 속성 순서) ----
OFF=0.012
pos=Pw-Nw*OFF                       # 면 바로 뒤(바깥쪽)에 깔아서 학습된 스플랫이 앞에 오게
def basis(n):
    # 바닥/천장: 복도축 기준, 벽: 위쪽 기준
    t1=np.where(np.abs(n@up)[:,None]>0.9, np.tile(ax1,(len(n),1)), np.tile(up,(len(n),1)))
    t1=t1-(np.einsum('ij,ij->i',t1,n))[:,None]*n; t1/=np.linalg.norm(t1,axis=1)[:,None]
    t2=np.cross(n,t1); return t1,t2
t1,t2=basis(Nw)
Rm=np.stack([t1,t2,Nw],2)           # columns
def rot2quat(R):
    w=np.sqrt(np.maximum(0,1+R[:,0,0]+R[:,1,1]+R[:,2,2]))/2
    x=np.sqrt(np.maximum(0,1+R[:,0,0]-R[:,1,1]-R[:,2,2]))/2*np.sign(R[:,2,1]-R[:,1,2])
    y=np.sqrt(np.maximum(0,1-R[:,0,0]+R[:,1,1]-R[:,2,2]))/2*np.sign(R[:,0,2]-R[:,2,0])
    z=np.sqrt(np.maximum(0,1-R[:,0,0]-R[:,1,1]+R[:,2,2]))/2*np.sign(R[:,1,0]-R[:,0,1])
    q=np.stack([w,x,y,z],1); return q/np.linalg.norm(q,axis=1)[:,None]
q=rot2quat(Rm)
assert np.allclose(np.einsum('nij,nj->ni',Rm,np.tile([0,0,1.],(len(q),1))),Nw,atol=1e-5)
SH0=0.28209479177387814
names=(["f_dc_0","f_dc_1","f_dc_2"]+["f_rest_%d"%i for i in sorted(range(24),key=str)]+
       ["opacity","rot_0","rot_1","rot_2","rot_3","scale_0","scale_1","scale_2","x","y","z"])
n=len(pos); A=np.zeros((n,len(names)),np.float32)
def setc(k,v): A[:,names.index(k)]=v
for i in range(3): setc("f_dc_%d"%i,(col[:,i]-0.5)/SH0)
setc("opacity",np.log(0.99/0.01))
for i in range(4): setc("rot_%d"%i,q[:,i])
sig=np.log(SP*0.7); setc("scale_0",sig); setc("scale_1",sig); setc("scale_2",np.log(2e-4))
for i,k in enumerate("xyz"): setc(k,pos[:,i])
def write_ply(path,A,names,comment=True):
    with open(path,"wb") as fh:
        h="ply\nformat binary_little_endian 1.0\n"
        if comment: h+="comment Exported from Brush\ncomment Vertical axis: y\ncomment SH degree: 2\n"
        h+="element vertex %d\n"%len(A)+"".join("property float %s\n"%k for k in names)+"end_header\n"
        fh.write(h.encode()); fh.write(np.ascontiguousarray(A,"<f4").tobytes())
write_ply(OUT+"/backdrop_corridor1.ply",A,names)
print("backdrop splats",n, os.path.getsize(OUT+"/backdrop_corridor1.ply")/1e6,"MB")

# ---- 2) 시작점 보강: 원래 points3D + 면 위 샘플 점 ----
S="/mnt/user-data/uploads/Sample/6/b/1b/data/sparse/0/"
Pt=read_points3D(S+"points3D.bin")
start=int(Pt["ids"].max())+1
ids=np.concatenate([Pt["ids"],np.arange(start,start+len(Pw))])
xyz=np.concatenate([Pt["xyz"],Pw]); rgb=np.concatenate([Pt["rgb"],(np.clip(col,0,1)*255).astype(np.uint8)])
err=np.concatenate([Pt["err"],np.ones(len(Pw))]); tracks=Pt["tracks"]+[b""]*len(Pw)
write_points3D(OUT+"/sparse/0/points3D.bin",ids,xyz,rgb,err,tracks)
for fn in ["cameras.bin","images.bin","frames.bin","rigs.bin"]: shutil.copy(S+fn,OUT+"/sparse/0/"+fn)
# 검증: 다시 읽어서 개수 확인
Q=read_points3D(OUT+"/sparse/0/points3D.bin"); print("points",len(Pt["ids"]),"->",len(Q["ids"]),"tracks ok",Q["tl"][:len(Pt['ids'])].sum()==Pt["tl"].sum())
