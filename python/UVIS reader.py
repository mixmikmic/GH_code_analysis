get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

from pyuvis import QUBE

lbl = '/Volumes/Data/backup/data/uvis/EUV2010_137_23_20.LBL'

qube = QUBE(lbl)

from IPython.html.widgets import interactive

def show_slice(i):
    im = plt.imshow(qube.data[i], aspect='auto', interpolation='nearest',
           cmap=plt.cm.spectral)
    plt.colorbar(im, ax=plt.gca())
interactive(show_slice, i=(0,qube.shape[0]))

plt.imshow(qube.data.mean(axis=0), aspect='auto', interpolation='nearest',
           cmap=plt.cm.rainbow)

qube.line_range

qube.band_range

qube.data1D

qube.data.shape

qube

