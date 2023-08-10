import numpy as np
import matplotlib.pyplot as plt
import twpca
from twpca import TWPCA
get_ipython().magic('matplotlib inline')

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

_, _, data = twpca.datasets.jittered_neuron()
model = TWPCA(data, n_components=1, warpinit='identity')

np.all(np.isclose(model.params['warp'], np.arange(model.shared_length), atol=1e-5, rtol=2))

np.nanmax(np.abs(model.transform() - data)) < 1e-5

model = TWPCA(data, n_components=1, warpinit='shift')

plt.imshow(np.squeeze(model.transform()))

