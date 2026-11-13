import numpy as np, pickle, time
from colmap_io import *
S="/mnt/user-data/uploads/Sample/6/b/1b/data/sparse/0/"
t=time.time()
cams=read_cameras(S+"cameras.bin"); print(cams)
ims=read_images(S+"images.bin"); print(len(ims), ims[0]["name"], time.time()-t)
P=read_points3D(S+"points3D.bin"); print(len(P["ids"]), time.time()-t)
C=np.array([-im["R"].T@im["t"] for im in ims])
pickle.dump(dict(cams=cams, ims=ims, xyz=P["xyz"], rgb=P["rgb"], err=P["err"], tl=P["tl"], ids=P["ids"], C=C), open("scene.pkl","wb"))
