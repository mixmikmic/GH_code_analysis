import numpy as np

m = 123.45   # Slope of our line (weight)
b = -67.89   # Intercept of our line (bias)

numDataPoints = 100  # Let's just have 100 total data points

X = np.random.rand(numDataPoints, 1)  # Let's generate a vector X with numDataPoints random numbers

noiseScale = 1.2  # The larger this value, the noisier the data.

trueLine = m*X + b  # Let's generate a vector Y based on a linear model of X 
y = trueLine + noiseScale * np.random.randn(numDataPoints, 1)  # Let's add some noise so the line is more like real data.

from neon.data import ArrayIterator
from neon.backends import gen_backend

gen_backend(backend='gpu', batch_size=2)  # Change to 'gpu' if you have gpu support 

train = ArrayIterator(X=X, y=y, make_onehot=False)

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

plt.figure(figsize=(10,7))
plt.scatter(X, y, alpha=0.7, color='g')
plt.plot(X, trueLine, alpha=0.5, color='r')
plt.title('Raw data is a line with slope (m) of {} and intercept (b) of {}'.format(m, b), fontsize=14);
plt.grid('on');
plt.legend(['True line', 'Raw data'], fontsize=18);

from neon.initializers import Gaussian
from neon.optimizers import GradientDescentMomentum
from neon.layers import Linear, Bias
from neon.layers import GeneralizedCost
from neon.transforms import SumSquared
from neon.models import Model
from neon.callbacks.callbacks import Callbacks

init_norm = Gaussian(loc=0.0, scale=1)

layers = [Linear(1, init=init_norm), # Linear layer with 1 unit
          Bias(init=init_norm)]      # Bias layer

model = Model(layers=layers)

# Loss function is the squared difference
cost = GeneralizedCost(costfunc=SumSquared())

optimizer = GradientDescentMomentum(0.1, momentum_coef=0.9)

# Execute the model
model.fit(train, 
          optimizer=optimizer, 
          num_epochs=11, 
          cost=cost,
          callbacks=Callbacks(model))

# print weights
slope = model.get_description(True)['model']['config']['layers'][0]['params']['W'][0][0]
print ("calculated slope = {:.3f}, true slope = {:.3f}".format(slope, m))

bias_weight = model.get_description(True)['model']['config']['layers'][1]['params']['W'][0][0]
print ("calculated bias = {:.3f}, true bias = {:.3f}".format(bias_weight, b))

plt.figure(figsize=(10,7))
plt.plot(X, slope*X+bias_weight, alpha=0.5, color='b', marker='^')
plt.scatter(X, y, alpha=0.7, color='g')
plt.plot(X, trueLine, '--', alpha=0.5, color='r')

plt.title('How close is our predicted model?', fontsize=18);
plt.grid('on');
plt.legend(['Predicted Line', 'True line', 'Raw Data'], fontsize=18);





