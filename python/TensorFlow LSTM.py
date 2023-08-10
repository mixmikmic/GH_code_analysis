from matplotlib import pyplot as plt
get_ipython().magic('matplotlib inline')
import pandas as pd
import numpy as np

import tensorflow as tf
print(tf.__version__)

import basic_tf as b

plt.style.use('ggplot')
figsize=(20, 5)

def plot_lr(lr):
    m = b.TSModel(name="lrsearch%f" % lr, timesteps=b.lstm_timesteps)
    h = b.train(m, 4, lr, verbose=False)
    return h

plots = {                       "1e-1": plot_lr(1e-1),
         "5e-2": plot_lr(5e-2), "1e-2": plot_lr(1e-2),
         "5e-3": plot_lr(5e-3), "1e-3": plot_lr(1e-3),
         "5e-4": plot_lr(5e-4), "1e-4": plot_lr(1e-4)}

npplots = {}
for k in plots.keys():
    npplots[k] = np.array(plots[k])

for k in npplots.keys():
    plots[k] = npplots[k].clip(max=1.0)

plot = pd.DataFrame(plots)

plot.plot(figsize=figsize)

filename = "tflstm/iter64_1e-2"

m = b.TSModel(filename, timesteps=b.lstm_timesteps)

m.load(filename)

print(m)

from singen import SinGen

g = SinGen(timesteps=b.lstm_timesteps)

x, y = g.batch()

y_ = m.predict(x)

res = pd.DataFrame({"predict": y_.squeeze(), "actual": y.squeeze()})

plt.style.use('ggplot')

res.plot()

x, y = g.batch()
y_ = m.predict(x)
res = pd.DataFrame({"predict": y_.squeeze(), "actual": y.squeeze()})

res.plot()

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
        xs = m.predict(xs)
        ys += [[i for i in xs.squeeze()][-1]]  # The last one is the only thing new
    return ys

fx, fy = gen_future(2)

pfy = pred_future(x, 2)

pd.DataFrame({'predict': pfy, 'actual': fx}).plot(figsize=figsize)

showpoints=10

pd.DataFrame({'predict y':pfy[:showpoints], 'gen x':fx[:showpoints]})



