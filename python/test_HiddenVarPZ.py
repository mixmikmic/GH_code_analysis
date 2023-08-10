import numpy as np
import matplotlib.pyplot as plt
import numpy as np
get_ipython().magic('matplotlib inline')
import sys
sys.path.append("..")
import fastcat as fc
pz=fc.photoz.PhotoZHiddenVar(0.03)

ztrue=np.array([0.05,0.58,0.8])
zar=np.linspace(0,1.5,1000)

F=pz.Fofz(zar)
plt.plot(zar,F)

arr=np.array(ztrue,dtype=[('z',np.float32)])

arra=pz.applyPhotoZ(arr,addErrors=False)

# plot p(z) for some zs.
plt.figure(figsize=(10,10))
p=np.array([pz.PofZ(arra,float(z),0.1) for z in zar])
for i,z in enumerate(ztrue):
    plt.plot (zar,p[:,i],label="ztrue=%f"%(z))
# Let's overplot main Gaussian at z=0.8 to see if it makes sense
sig08=pz.sigma*(1+0.8)
plt.plot(zar,0.1*np.exp(-(zar-0.8)**2/(2*sig08**2)),'k-')
plt.legend(loc='lower right')

## and now the same for cumulative p
p=np.array([pz.cPofZ(arra,float(z)) for z in zar])
for i,z in enumerate(ztrue):
    plt.plot (zar,p[:,i],label="ztrue=%f"%(z))
# Let's overplot main Gaussian at z=0.8 to see if it makes sense
plt.legend(loc='lower right')



