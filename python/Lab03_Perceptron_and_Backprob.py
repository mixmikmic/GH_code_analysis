from random import choice
from numpy import array, dot, random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

activation = lambda x: 0 if x < 0 else 1

training_data = np.array([
    [0,0,1],
    [1,0,1],
    [0,1,1],
    [1,1,1]
])
target_data = np.array([
    0, 
    1, 
    1, 
    1 
])
w = random.rand(3)
#learning rate
eta = 0.2
n = 100

#loop training
for i in range(n):
    train_index = choice(range(0,4))
    x, expected = (training_data[train_index],target_data[train_index])
    #feed forward
    result = dot(x, w)
    result = activation(result)
    #error estimation
    error = expected - result
    #back prob
    w += eta * error * x
#end loop

print(w)

for x in training_data:
    result = dot(x, w)
    print("{}: {} -> {}".format(x[:2], result, activation(result)))

import utils
import json
plt = utils.plot_logic(plt,training_data,target_data)
plt = utils.plot_space(plt,w)
plt.show()

