import numpy as np, pickle
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
d=pickle.load(open("scene.pkl","rb")); X=d["xyz"]; C=d["C"]; good=d["err"]<2
a,b,up,o=np.load("frame.npy")
rng=np.random.default_rng(0)
def ransac(P, n_iter=2000, thr=0.02, normal_hint=None, max_angle=25):
    best=(0,None)
    for _ in range(n_iter):
        s=P[rng.choice(len(P),3,replace=False)]
        n=np.cross(s[1]-s[0],s[2]-s[0]); nn=np.linalg.norm(n)
        if nn<1e-9: continue
        n/=nn
        if normal_hint is not None:
            if abs(n@normal_hint)<np.cos(np.radians(max_angle)): continue
        dd=np.abs((P-s[0])@n); k=(dd<thr).sum()
        if k>best[0]: best=(k,(n,s[0]))
    n,p0=best[1]; inl=np.abs((P-p0)@n)<thr
    # refine by SVD
    Q=P[inl]; c=Q.mean(0); _,_,vt=np.linalg.svd(Q-c,full_matrices=False); n=vt[2]
    if n@normal_hint<0: n=-n
    return n,c,inl.mean(),np.abs((P-c)@n)
# keep points near the camera path (within 4 units) to drop outliers
from scipy.spatial import cKDTree
dist,_=cKDTree(C).query(X); near=good&(dist<3)
Xn=X[near]; h=(Xn-o)@up
low=Xn[h<-0.4]
n_f,c_f,frac,res=ransac(low,normal_hint=up)
print("floor normal",n_f,"angle to cam-up",np.degrees(np.arccos(abs(n_f@up))),"inlier frac",frac)
print("residual pct",np.percentile(res,[10,50,90]))
up2=n_f
a2=np.cross(up2,b); a2/=np.linalg.norm(a2)
# corridor axis via PCA of camera centers in horizontal plane
Cc=C-C.mean(0); Cc-=np.outer(Cc@up2,up2); _,_,vt=np.linalg.svd(Cc,full_matrices=False); ax1=vt[0]; ax2=np.cross(up2,ax1)
def F(p): q=p-c_f; return np.stack([q@ax1,q@ax2,q@up2],-1)
Xf=F(Xn); Cf=F(C)
print("cam height above floor pct",np.percentile(Cf[:,2],[5,50,95]))
print("along range",Cf[:,0].min(),Cf[:,0].max(),"lateral cam",np.percentile(Cf[:,1],[5,95]))
print("pt height pct",np.percentile(Xf[:,2],[1,5,50,95,99,99.5]))
fig,axs=plt.subplots(3,1,figsize=(18,14))
s=rng.choice(len(Xf),min(200000,len(Xf)),replace=False); col=d["rgb"][near][s]/255
axs[0].scatter(Xf[s,0],Xf[s,1],s=0.2,c=col); axs[0].plot(Cf[:,0],Cf[:,1],'r-'); axs[0].set_aspect('equal'); axs[0].set_title('top (along x lateral)')
axs[1].scatter(Xf[s,0],Xf[s,2],s=0.2,c=col); axs[1].plot(Cf[:,0],Cf[:,2],'r-'); axs[1].set_aspect('equal'); axs[1].set_title('side (along x height)')
m=np.abs(Xf[:,0]-np.median(Cf[:,0]))<1.5
axs[2].scatter(Xf[m,1],Xf[m,2],s=0.3,c=d["rgb"][near][m]/255); axs[2].set_aspect('equal'); axs[2].set_title('cross-section at middle (lateral x height)')
plt.tight_layout(); plt.savefig("diag2.png",dpi=70)
np.savez("frame2.npz",ax1=ax1,ax2=ax2,up=up2,c=c_f)
