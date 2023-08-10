get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import numpy  as np
import matplotlib.pyplot as plt
import skimage.io
from scipy.io import loadmat
import numpy as np
import pandas as pd

from saliency import utils
from saliency.model import  IttyKoch
from saliency.data import load_video

video_id = 2

ids = [6,12,17]

fname_save = "Clip{}.csv".format(ids[video_id])

X, y = load_video(video_id)
print("Loaded video with {} frames of resolution {}x{}px".format(*X.shape))
print("Loaded {} timesteps of groundtruth data from {} subjects".format(*y.shape[1:]))

import h5py

with h5py.File('/home/stes/media/saliency/icf-compressed.hdf5', 'r') as ds:
    S_icf = ds[str(video_id)][:,::9,::9,0]

with h5py.File('/home/stes/media/saliency/deepgaze-compressed.hdf5', 'r') as ds:
    S_deep = ds[str(video_id)][:,::9,::9,0]
    
with h5py.File('/home/stes/media/saliency/ittykoch.hdf5', 'r') as ds:
    S_ittykoch = ds[str(video_id)][...]
    
S_icf.shape, S_deep.shape, S_ittykoch.shape

mean   = np.nanmean(y, axis=-1)
median = np.nanmedian(y, axis=-1)
std    = np.nanstd(y, axis=-1)

t = np.arange(len(mean.T))

plt.plot(t, median.T)
plt.fill_between(t, median[0]-std[0], median[0] + std[0], label="y-Direction")
plt.fill_between(t, median[1]-std[1], median[1] + std[1], label="x-Direction")
plt.title("Human Fixation Average")
plt.legend()
plt.show()

from saliency import  saliency

didc = slice(50+39,-41,30)
idc = slice(50+40,-40,30)

n_examples = len(X[idc])

fig, axes = plt.subplots(n_examples,4,figsize=(20,20))

titles = ['Input', 'IttyKoch', 'DeepGaze', 'ICF', 'combined']

for ax, imgs in zip(axes, zip(X[idc], S_ittykoch[idc], S_deep[idc], S_icf[idc])):
    for a, i in zip(ax,imgs):
        a.imshow(i, cmap="coolwarm")
        a.grid("off")
        a.axis("off")

for a, t in zip(axes[0], titles):
    a.set_title(t)
    
plt.show()

didc = slice(0,-1)
idc = slice(1,None)

diff = lambda x : np.exp(x[idc]) * abs(np.exp(x[idc])-np.exp(x[didc]))

dS_ittykoch = diff(S_ittykoch)
dS_icf      = diff(S_icf)
dS_deep     = diff(S_deep)

S_all = (sum(s for s in [dS_icf, dS_deep]))

from saliency import  saliency

didc = slice(50+39,-41,30)
idc = slice(50+40,-40,30)

n_examples = len(X[idc])

fig, axes = plt.subplots(n_examples,5,figsize=(20,20))

titles = ['Input', 'IttyKoch', 'DeepGaze', 'ICF', 'combined']

for ax, imgs in zip(axes, zip(X[idc], dS_ittykoch[idc], dS_deep[idc], dS_icf[idc], S_all[idc])):
    for a, i in zip(ax,imgs):
        a.imshow(i, cmap="coolwarm")
        a.grid("off")
        a.axis("off")

for a, t in zip(axes[0], titles):
    a.set_title(t)
    
plt.savefig("report/fig/clip{}.pdf".format(ids[video_id]), bbox_inches="tight")

S_all_log = np.log(1e-2+S_all)

yy, xx = utils.argmax2d(S_all_log)
yy *= 9
xx *= 9

fig, (ax1, ax2) = plt.subplots(1,2,figsize=(20,8))

ax1.imshow(X[0])
ax1.plot(median[0], median[1], alpha=.8, label="ground truth")
ax1.plot(xx,yy, alpha=.8, label="predicted")
ax1.legend()
ax1.axis("off")
ax1.set_title("Fixation Trajectory")

t = np.arange(y.shape[1])
ax2.imshow(X[100])
for i in range(y.shape[-1]):
    ax2.scatter(y[0,:,i], y[1,:,i], c="blue", alpha=.5)
ax2.scatter(xx,yy,color="orange", label="predicted")
ax2.set_title("Fixation Points")
ax2.legend()
ax2.axis("off")

plt.show()

for s, t in zip([S_ittykoch, S_icf, S_deep, S_all], ["Itty Koch", "ICF", "DeepGaze", "Combined"]):

    nss = utils.nss(s,y/9,normalized=False)[1]
    plt.plot(nss,label=t)
    plt.legend()
plt.show()

from saliency import utils

names  = ['IttyKoch-BAS', 'DeepGaze-BAS', 'ICF-BAS', 'IttyKoch-DIFF',
          'DeepGaze-DIFF', 'ICF-DIFF', 'combined-DIFF']
output = [S_ittykoch, S_deep, S_icf, dS_ittykoch, dS_deep, dS_icf]


print("Clip {}".format(ids[video_id]))

results = {}

for name, S in zip(names, output):

    if len(S) < y.shape[1]:
        S = np.concatenate([S[0:],S_all[0:1]], axis=0)

    mse,_ = utils.mse(S, y/9, per_frame=True)
    nss,_ = utils.nss(S, y/9, normalized=False, per_frame=True)
    nssl,_ = utils.nss(np.log(S-S.min() + 1e-2), y/9, normalized=False, per_frame=True)

    print("{} \t MSE : {:10.2f}, \t NSS : {:.2f} \t NSS log : {:.2f}".format(name, mse, nss, nssl))
    
    
    yy, xx = utils.argmax2d(S)
    results['{}_x'.format(name)] = xx
    results['{}_y'.format(name)] = yy

df = pd.DataFrame(data=results, index=np.arange(len(X)))
df.to_csv(fname_save)

