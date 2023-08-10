import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
get_ipython().magic('matplotlib nbagg')

hdul_img=fits.open('set096/o8010g0344o.1300829.ch.1992525.XY36.p00.fits')
data_img=hdul_img[0].data
header=hdul_img[0].header
hdul_img.close()

hdul_img2=fits.open('set096/o8010g0362o.1300847.ch.1992543.XY36.p00.fits')
data_img2=hdul_img2[0].data
header2=hdul_img2[0].header
hdul_img2.close()

fig,axs=plt.subplots(1,2,sharey=True,sharex=True)
axs[0].imshow(data_img,cmap='gray',vmin=0,vmax=500)
axs[1].imshow(data_img2,cmap='gray',vmin=0,vmax=500)
ax=axs[0]

hdul_xy=fits.open('o8010g0344o.1300829.ch.1992525.XY36.p00.axy')
hdul_xy.info()
srcs=[]
for src in hdul_xy[1].data:
    srcs+=[[src[0],src[1]]]
srcs=np.array(srcs)
hdul_xy.close()

for ax in axs:
    ax.plot(srcs[:,0],srcs[:,1],'ro',mfc='None',ms=5)

hdul_objs=fits.open('o8010g0344o.1300829.ch.1992525.XY36.p00-indx.xyls')
hdul_objs.info()
objs=[]
for obj in hdul_objs[1].data:
    objs+=[[obj[0],obj[1]]]
objs=np.array(objs)
hdul_objs.close()

ax.plot(objs[:,0],objs[:,1],'gs',mfc='None',ms=7)

hdul_crd=fits.open('o8010g0344o.1300829.ch.1992525.XY36.p00.rdls')
hdul_crd.info()
crds=[]
for i,crd in enumerate(hdul_crd[1].data):
    crds+=["%.5f,%.5f"%(crd[0]/15,crd[1])]
crds=np.array(crds)
hdul_crd.close()

for i in range(len(crds)):
    t=ax.text(objs[i,0],objs[i,1],crds[i],color='b',fontsize=10,zorder=100,rotation=45,ha='left',va='bottom')

header

ra=(23+37/60.0+49.518/3600.0)*15
dec=-(4+5/60.0+38.469/3600.0)
print(ra,dec)

np.sign(-3)

hdul_psf=fits.open("default.psf")

hdul_psf.info()

hdul_psf[1].data

hdul_cat=fits.open("sources.cat")
hdul_cat.info()

hdul_cat[1].header



