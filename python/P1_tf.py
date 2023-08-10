from matplotlib import pyplot as plt
get_ipython().magic('matplotlib inline')
import pandas as pd
import numpy as np

import tensorflow as tf
print(tf.__version__)

plt.style.use('ggplot')

figsize=(20, 5)

from singen import SinP1Gen

p1g = SinP1Gen(timesteps=100)

x, y = p1g.batch()

(x.shape, x.squeeze().shape)

p = pd.DataFrame({"x": x.squeeze()})

p.plot(figsize=figsize)

import basic_tf_p1 as b

breadth = 2
depth = 4

m1 = b.TSModel(name='foo', timesteps=b.lstm_timesteps, breadth=breadth, depth=depth)

losses1 = b.train(m1, epochs=1, lr=b.default_lr, epere=10, verbose=True)

m1.close()

breadth = 4
depth = 2

m2 = b.TSModel(name='foo', timesteps=b.lstm_timesteps, breadth=breadth, depth=depth)

losses2 = b.train(m2, epochs=1, lr=b.default_lr, epere=10, verbose=True)

m2.close()

p = pd.DataFrame({"m1": losses1, "m2": losses2})

p.plot(figsize=figsize)



