import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
d=np.loadtxt("Metolius_MOD15_LAI.txt")
plt.figure(figsize=(10,10))
x = np.arange( 166 )
plt.plot ( x, d[:, 2], 'o' )
plt.vlines ( x, d[:, 2] - d[:, 3], d[:, 2] + d[:, 3] )
plt.xlabel("Observation time[-]")
plt.ylabel(r'LAI $m^2m^{-2}$')



