import numpy as np
import sys
from matplotlib import pylab as plt
get_ipython().magic('matplotlib inline')

sys.path.append("..")
import fastcat as fc
pz = fc.photoz.PhotoZHist('./pzdist.txt')

arr = np.array([0, 1, 0, 107])
arr.dtype=[('z','<f8'), ('iz','<i8'), ('itype','<i8'), ('imag','<i8')]
indice = pz.tup2id(1,0,107) #indice=470

arr=np.array(ztrues,dtype=[('z',np.float32)])
arr = pz.applyPhotoZ(arr)

ax=plt.subplot(111)
dummy = ax.hist(smp,100, normed=True);
#grrr where does factor 3 comes from...
ax.plot(pz.dz, pz.dataset[indice,3:]*3/np.diff(dummy[1])[0])
ax.set_title(pz.dataset[indice,:3])
ax.set_xlabel('dz')
ax.set_xlim(-0.1,3);

#ztrues=np.array([0.7, 0.1])
ntot=100
ztrues = 1.+np.random.rand(ntot)*1.9

arr=np.array(ztrues,dtype=[('z',np.float32)])
arr = pz.applyPhotoZ(arr)

nsamples=1000
photoz=[]
indices = pz.tup2id(arr['iz'], arr['itype'], arr['imag'])
photoz_pdfs = pz.getpdf(arr)
mask=photoz_pdfs.sum(axis=1)!=0
masked_pdfs=photoz_pdfs[mask]
print masked_pdfs.shape
[plt.plot(pz.dz,pdf) for pdf in masked_pdfs];
cumsum = np.cumsum(masked_pdfs, axis=1)
#print photoz_pdfs
#print mask

nsamples=1000

pz_samples = pz.drawPhotoZ(arr, nsamples)    
ax2=plt.subplot(111)
i=80
zt=arr[i]['z']
pdf = pz.getpdf(arr[i])
smp = pz_samples[i]
dummy = ax2.hist(smp,100, normed=True);
ax2.plot(pz.dz+zt, pdf/np.diff(dummy[1])[0])
ax2.set_xlabel('z+dz')
ax2.set_xlim(zt-0.5,zt+0.5);

a=pz.PofZ(arr, 0.6, 1)
b=pz.cPofZ(arr, 1.5)
print a, b





