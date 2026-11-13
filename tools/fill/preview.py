import numpy as np, pickle, cv2, glob, os
d=pickle.load(open("scene.pkl","rb")); ims=d["ims"]; cam=d["cams"][1]
b=np.load("backdrop.npz"); Pw=b["Pw"]; col=b["col"]; ok=~np.isnan(col[:,0])
fxp,fyp,cxp,cyp=cam["params"]; R4=4
names={os.path.basename(p):p for p in glob.glob("/mnt/user-data/uploads/Sample/6/b/1b/data/images/*.jpg")}
views=[im for im in ims if im["name"] in names]
rows=[]
for im in [views[i] for i in (20,70,120,160)]:
    img=cv2.imread(names[im["name"]],cv2.IMREAD_REDUCED_COLOR_4); h,w=img.shape[:2]
    Xc=Pw[ok]@im["R"].T+im["t"]; z=Xc[:,2]; m=z>0.05
    u=(fxp*Xc[m,0]/z[m]+cxp)/R4; v=(fyp*Xc[m,1]/z[m]+cyp)/R4; c=col[ok][m]; zz=z[m]
    o=np.argsort(-zz); u,v,c,zz=u[o],v[o],c[o],zz[o]
    canvas=np.zeros_like(img)
    r=np.clip((6/zz).astype(int),1,6)
    for k in range(len(u)):
        if 0<=u[k]<w and 0<=v[k]<h:
            cv2.circle(canvas,(int(u[k]),int(v[k])),int(r[k]),(c[k,::-1]*255).tolist(),-1)
    rows.append(np.concatenate([img,canvas],1))
cv2.imwrite("preview.jpg",np.concatenate(rows,0)[::2,::2])
