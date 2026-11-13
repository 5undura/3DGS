import numpy as np, pickle, cv2, os, glob, time
from scipy import ndimage as ndi
d=pickle.load(open("scene.pkl","rb")); cams=d["cams"]; ims=d["ims"]
f=np.load("frame2.npz"); ax1,ax2,up,c0=f["ax1"],f["ax2"],f["up"],f["c"]
ft=np.load("foot.npz"); g=ft["g"]; mn=ft["mn"]; cell=float(ft["cell"])
g=ndi.binary_opening(g,np.ones((5,5)))
H=1.60; SP=0.025          # 천장 높이, 샘플 간격 (COLMAP 단위, 1 ≈ 2m)
Mw=np.stack([ax1,ax2,up],1)   # local->world rotation columns
def W(p): return c0+p@Mw.T
# ---------- samples (local coords) + normals (pointing into the room) ----------
xs=np.arange(mn[0],mn[0]+g.shape[1]*cell,SP); ys=np.arange(mn[1],mn[1]+g.shape[0]*cell,SP)
XX,YY=np.meshgrid(xs,ys); ii=((YY-mn[1])/cell).astype(int); jj=((XX-mn[0])/cell).astype(int)
inside=g[np.clip(ii,0,g.shape[0]-1),np.clip(jj,0,g.shape[1]-1)]
P=[];N=[];K=[]
fx=XX[inside]; fy=YY[inside]
P.append(np.stack([fx,fy,np.zeros_like(fx)],1)); N.append(np.tile([0,0,1.],(len(fx),1))); K.append(np.zeros(len(fx),int))
P.append(np.stack([fx,fy,np.full_like(fx,H)],1)); N.append(np.tile([0,0,-1.],(len(fx),1))); K.append(np.ones(len(fx),int))
# walls: boundary edges of footprint grid
zs=np.arange(SP/2,H,SP)
wp=[];wn=[]
for di,dj,nrm in [(0,1,(-1,0)),(0,-1,(1,0)),(1,0,(0,-1)),(-1,0,(0,1))]:
    nb=np.zeros_like(g); 
    sh=np.roll(g,(-di,-dj),(0,1)); edge=g&~sh   # inside cell whose neighbour in (di,dj) is outside
    r,c=np.nonzero(edge)
    cx=mn[0]+(c+0.5)*cell+dj*cell/2; cy=mn[1]+(r+0.5)*cell+di*cell/2
    # sample along edge (length cell) at SP
    ts=np.arange(-cell/2+SP/4,cell/2,SP/1.0)
    for t in ts:
        ex=cx+(t if di!=0 else 0); ey=cy+(t if dj!=0 else 0)
        for zz in zs:
            wp.append(np.stack([ex,ey,np.full_like(ex,zz)],1)); wn.append(np.tile([nrm[0],nrm[1],0.],(len(ex),1)))
P.append(np.concatenate(wp)); N.append(np.concatenate(wn)); K.append(np.full(len(P[-1]),2))
P=np.concatenate(P); N=np.concatenate(N); K=np.concatenate(K)
print("samples floor/ceil/wall", [(K==k).sum() for k in range(3)])
Pw=W(P); Nw=N@Mw.T
# ---------- color by projecting into images ----------
cam=cams[1]; fxp,fyp,cxp,cyp=cam["params"]; R4=4
names={os.path.basename(p):p for p in glob.glob("/mnt/user-data/uploads/Sample/6/b/1b/data/images/*.jpg")}
views=[im for im in ims if im["name"] in names]
print("views",len(views))
best=np.full((len(Pw),6),np.inf)   # keep top-6 scores (lower better)
cols=np.zeros((len(Pw),6,3),np.float32)
t0=time.time()
for vi,im in enumerate(views):
    img=cv2.imread(names[im["name"]],cv2.IMREAD_REDUCED_COLOR_4)
    h,w=img.shape[:2]
    Xc=Pw@im["R"].T+im["t"]; zc=Xc[:,2]
    ok=zc>0.05
    u=(fxp*Xc[:,0]/np.maximum(zc,1e-6)+cxp)/R4; v=(fyp*Xc[:,1]/np.maximum(zc,1e-6)+cyp)/R4
    ok&=(u>2)&(u<w-3)&(v>2)&(v<h-3)
    Cw=-im["R"].T@im["t"]; vd=Cw-Pw; dist=np.linalg.norm(vd,axis=1); cosang=np.einsum('ij,ij->i',vd,Nw)/dist
    ok&=cosang>0.15
    idx=np.nonzero(ok)[0]
    if len(idx)==0: continue
    score=dist[idx]/cosang[idx]**0.5        # 가깝고 정면일수록 좋음
    c=img[v[idx].astype(int),u[idx].astype(int)][:,::-1].astype(np.float32)/255
    # insert into top-6
    worst=np.argmax(best[idx],1); better=score<best[idx,worst]
    ii2=idx[better]; ww=worst[better]
    best[ii2,ww]=score[better]; cols[ii2,ww]=c[better]
    if vi%30==0: print(vi,time.time()-t0)
have=np.isfinite(best).sum(1)
print("coverage: no view",(have==0).mean(),"by kind",[ (have[K==k]==0).mean() for k in range(3)])
# median over available views (robust to occluders)
cm=np.where(np.isfinite(best)[...,None],cols,np.nan)
col=np.nanmedian(cm,1)
col=np.where(np.isnan(col),np.nan,col)
np.savez("backdrop.npz",P=P,N=N,K=K,Pw=Pw,Nw=Nw,col=col,have=have,H=H,SP=SP)
