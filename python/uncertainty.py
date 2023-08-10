get_ipython().magic('matplotlib inline')

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import trackpy as tp
import pims

# optional, just for plot style:
mpl.style.use('http://tiny.cc/leheny-style-sans-serif/raw') 

class SimulatedFrame(object):
    
    def __init__(self, shape, dtype=np.uint8):
        self.image = np.zeros(shape, dtype=dtype)
        self._saturation = np.iinfo(dtype).max
        self.shape = shape
        self.dtype =dtype
        
    def add_spot(self, pos, amplitude, r, ecc=0):
        "Add a Gaussian spot to the frame."
        x, y = np.meshgrid(*np.array(list(map(np.arange, self.shape))) - np.asarray(pos))
        spot = amplitude*np.exp(-((x/(1 - ecc))**2 + (y*(1 - ecc))**2)/(2*r**2)).T
        self.image += np.clip(spot, 0, self._saturation).astype(self.dtype)
        
    def with_noise(self, noise_level, seed=0):
        "Return a copy with noise."
        rs = np.random.RandomState(seed)
        noise = rs.randint(-noise_level, noise_level, self.shape)
        noisy_image = np.clip(self.image + noise, 0, self._saturation).astype(self.dtype)
        return noisy_image
    
    def add_noise(self, noise_level, seed=0):
        "Modify in place with noise."
        self.image = self.with_noise(noise_level, seed=seed)

fig, axes = plt.subplots(2, 2)

frame = SimulatedFrame((20, 30))
frame.add_spot((10, 15), 200, 2.5)

for ax, noise_level in zip(axes.ravel(), [1, 20, 40, 90]):
    noisy_copy = frame.with_noise(noise_level)
    features = tp.locate(noisy_copy, 13, topn=1, engine='python')
    tp.annotate(features, noisy_copy, plot_style=dict(marker='x'), imshow_style=dict(vmin=0, vmax=255), ax=ax)
    dx, dy, ep = features[['x', 'y', 'ep']].iloc[0].values - [16, 10, 0]
    ax.set(xticks=[5, 15, 25], yticks=[5, 15])
    ax.set(title=r'Signal/Noise = {signal}/{noise}'.format(
              signal=200, noise=noise_level))
    ax.text(0.5, 0.1, r'$\delta x={dx:.2}$  $\delta y={dy:.2}$'.format(
                dx=abs(dx), dy=abs(dy)), ha='center', color='white', transform=ax.transAxes)
    ax.text(0.05, 0.85, r'$\epsilon={ep:.2}$'.format(ep=abs(ep)), ha='left', color='white',
            transform=ax.transAxes)
fig.subplots_adjust()

good_results = []
bad_results = []
steps = np.linspace(0, 1, num=10, endpoint=True)

for s in steps:
    frame = SimulatedFrame((20, 30))
    frame.add_spot((10, 15 + s), 200, 2.5)
    feature = tp.locate(frame.image, 13, topn=1, preprocess=False, engine='python')
    good_results.append(feature)
    feature = tp.locate(frame.image, 9, topn=1, preprocess=False, engine='python')
    bad_results.append(feature)
    
good_df = pd.DataFrame([pd.concat(good_results)['x'].reset_index(drop=True) - 15, pd.Series(steps, name='true x')]).T
bad_df = pd.DataFrame([pd.concat(bad_results)['x'].reset_index(drop=True) - 15, pd.Series(steps, name='true x')]).T

fig, ax = plt.subplots()
ax.plot([-0.1, 1.1], [-0.1, 1.1], color='gray')
ax.plot(bad_df['true x'], bad_df['x'], marker='s', lw=0, mfc='r', mec='r', mew=1, label='diameter=9')
ax.plot(good_df['true x'], good_df['x'], marker='o', lw=0, mfc='k', mec='k', mew=1, label='diameter=13')
ax.set(xlabel=r'ground truth $x$ mod 1', ylabel=r'measured $x$ mod 1')
ax.set(xlim=(-0.1, 1.1), ylim=(-0.1, 1.1))
ax.legend(loc='best');

N = 10000
traj = pd.DataFrame(np.random.randn(N, 2).cumsum(0), columns=['x', 'y'])
noisy_traj = traj + np.random.randn(N, 2)
traj['frame'] = np.arange(N)
noisy_traj['frame'] = np.arange(N)

fig, ax = plt.subplots()
ax.plot(tp.msd(traj, 1, 1)['msd'], label='random walk')
ax.plot(tp.msd(noisy_traj, 1, 1)['msd'], label='random walk + random noise')
ax.legend()
ax.set(xscale='log', yscale='log')
ax.set(xlabel=r'$t$ [s]', ylabel=r'$\langle \Delta r^2 \rangle$ [\textmu m$^2$]');

