get_ipython().magic('matplotlib inline')

import keras.backend as K
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

sns.set_context('talk')

activations = {'sigmoid': K.sigmoid, 'tanh': K.tanh, 'ReLU': K.relu}

x = K.placeholder(ndim=1)

plt.figure(figsize=(10,12))
idx = 0

for name, activation in activations.items():
    f = K.function([x], activation(x))
    y = f([np.linspace(-5, 5)])
    plt.subplot(3,1,idx+1)
    plt.plot(np.linspace(-5, 5), y)
    plt.xlabel(name)
    idx += 1
    plt.ylim([np.min(y)-0.025, np.max(y)+0.025])

model.add(Dense(100, activation='sigmoid'))



