get_ipython().magic('matplotlib inline')
from gpgLabs.Seismic.SeismicRefraction import *

plotWavelet()

fig, ax = plt.subplots(1, 3, figsize=(15,6))
ax[0].set_title('Expected Arrival Times')
ax[1].set_title('Clean Data')
ax[2].set_title('Noisy Data')
ax[0]=viewTXdiagram(x0=1., dx=8, v1=400., v2=1000., v3=1500., z1=5., z2=15., ax=ax[0])
ax[1]=plotWiggleTX(x0=1., dx=8, v1=400., v2=1000., v3=1500., z1=5., z2=15., ax=ax[1])
ax[2]=plotWiggleTX(x0=1., dx=8, v1=400., v2=1000., v3=1500., z1=5., z2=15., ax=ax[2], noise=True)
plt.show()

makeinteractSeisRefracSurvey()

makeinteractTXwigglediagram()



