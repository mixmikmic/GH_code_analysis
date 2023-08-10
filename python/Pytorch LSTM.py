from matplotlib import pyplot as plt
get_ipython().magic('matplotlib inline')
import pandas as pd
import numpy as np

import torch
print(torch.__version__)

import basic_pytorch as b
from basic_pytorch import pt_input

m = b.TSModel(b.lstm_timesteps, b.lstm_batchsize)

m.load('2017-07-09::11:41:36.pt')

m.eval()

from singen import SinGen

g = SinGen(timesteps=b.lstm_timesteps)

x, y = g.batch()

y_ = m(pt_input(x))

y_ = y_.data.numpy()

y_.shape

res = pd.DataFrame({"predict": y_.squeeze(), "actual": y.squeeze()})

plt.style.use('ggplot')

res.plot()

x, y = g.batch()
y_ = m(pt_input(x)).data.numpy()
res = pd.DataFrame({"predict": y_.squeeze(), "actual": y.squeeze()})

res.plot()

m

figsize=(20, 5)

x, y = g.batch()

def gen_future(count):
    xs = []
    ys = []
    for _ in range(count):
        tx, ty = g.batch()
        xs += [i for i in tx.squeeze()]
        ys += [i for i in ty.squeeze()]
    return xs, ys

def pred_future(xs, count):
    timesteps = xs.shape[1]
    ys = []
    # Each time through this predict loop we get one future element
    for _ in range(count * timesteps):
        xs = m(pt_input(xs)).data.numpy()
        xs = xs.reshape(xs.shape + (1,))
        ys += [[i for i in xs.squeeze()][-1]]  # The last one is the only thing new
    return ys

fx, fy = gen_future(2)

pd.DataFrame({'predict': pfy, 'actual': fx}).plot(figsize=figsize)

showpoints=10

pd.DataFrame({'predict y':pfy[:showpoints], 'gen x':fx[:showpoints]})



