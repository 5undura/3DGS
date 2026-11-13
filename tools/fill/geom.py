import numpy as np, pickle
from scipy.spatial import cKDTree
from scipy import ndimage as ndi
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
d=pickle.load(open("scene.pkl","rb")); X=d["xyz"]; C=d["C"]; good=d["err"]<2
f=np.load("frame2.npz"); ax1,ax2,up,c0=f["ax1"],f["ax2"],f["up"],f["c"]
def F(p): q=p-c0; return np.stack([q@ax1,q@ax2,q@up],-1)
dist,_=cKDTree(C).query(X); near=good&(dist<3)
Xf=F(X[near]); Cf=F(C)
z=Xf[:,2]
h,e=np.histogram(z[(z>1.0)&(z<2.2)],bins=120)
for i in np.argsort(h)[-8:][::-1]: print("z %.3f n %d"%(e[i],h[i]))
print("pct>1.0", np.percentile(z[z>1.0],[50,80,90,95,98,99]))
# footprint from floor inliers
fl=np.abs(z)<0.03
cell=0.04
xy=Xf[fl,:2]; mn=xy.min(0)-0.5; mx=xy.max(0)+0.5
W=np.ceil((mx-mn)/cell).astype(int)
g=np.zeros(W[::-1],bool)
ij=((xy-mn)/cell).astype(int); g[ij[:,1],ij[:,0]]=True
# also camera footprint
cij=((Cf[:,:2]-mn)/cell).astype(int); g[cij[:,1],cij[:,0]]=True
g2=ndi.binary_closing(g,np.ones((7,7)),iterations=2); g2=ndi.binary_fill_holes(g2)
lab,n=ndi.label(g2); sizes=ndi.sum(g2,lab,range(1,n+1)); g3=lab==(1+np.argmax(sizes))
g3=ndi.binary_opening(g3,np.ones((3,3)))
print("grid",g.shape,"raw",g.sum(),"final",g3.sum(),"area units2",g3.sum()*cell*cell)
np.savez("foot.npz",g=g3,mn=mn,cell=cell)
fig,axs=plt.subplots(2,1,figsize=(16,8))
axs[0].imshow(g,origin='lower'); axs[0].set_title('floor points occupancy')
axs[1].imshow(g3,origin='lower'); axs[1].plot(cij[:,0],cij[:,1],'r.',ms=1); axs[1].set_title('footprint')
plt.savefig("foot.png",dpi=70)
