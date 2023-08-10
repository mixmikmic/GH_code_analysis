import numpy as np
import rasterio

filepath = '/home/sravya/data/satellite/nepal/usgs/worldview/29APR15WV010500015APR29062253-P1BS-500308331010_01_P059.ntf'

dataset = rasterio.open(filepath)

dataset.transform

dataset.name

dataset.mode

dataset.closed

dataset.count

dataset.width

dataset.height

{i: dtype for i, dtype in zip(dataset.indexes, dataset.dtypes)}

dataset.dtypes

dataset.bounds

dataset.transform

dataset.crs

dataset.indexes

band1 = dataset.read(1)

band1.shape

band1

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

plt.imshow(band1[:1000, :1000])

plt.imshow(band1[:5000, :5000], cmap='gray')

im = band1[:1000, :1000]

from tools2 import plots, plot

plot(im, f=24)

im = band1[:5000, :5000]
plot(im, f=24)

import smopy
map = smopy.Map((42., -1., 55., 3.), z=4)
x, y = map.to_pixels(48.86151, 2.33474)
ax = map.show_mpl(figsize=(8, 6))
ax.plot(x, y, 'or', ms=10, mew=2);

import smopy
# https://tools.wmflabs.org/geohack/geohack.php?pagename=Nepal&params=27_42_N_85_19_E_type:city
la, lo = 27.7, 85.316667
n = 1.0
map = smopy.Map((la-n, lo-n, la+n, lo+n), z=10)
ax = map.show_mpl(figsize=(8, 8))

kath = map.fetch()

type(kath)

kath_np = np.array(kath)

kath_np.shape

plot(kath_np, f=16)

import time

import numpy as np
import rasterio
import smopy
from tools2 import plots, plot
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

filepath = '../data/worldview1_ntf/29APR15WV010500015APR29062253-P1BS-500308331010_01_P059.ntf'
dataset = rasterio.open(filepath)
band1 = dataset.read(1)

k = 500
im = band1[:k, :k]
plot(im, f=4)

la, lo = 27.7, 85.316667
n = 1.0
box = (la-n, lo-n, la+n, lo+n)
ims = []
for zoom in (6, 7, 8):
    osm = smopy.Map(box, z=zoom)
    npl = osm.fetch()
    npl_np = np.array(npl)
    ims.append(npl_np)
    time.sleep(1)

plot(ims, r=len(ims), f=36, t=['zoom = ' + str(i) for i in (6, 7, 8)])

la, lo = 27.7, 85.316667
n = 1.0
box = (la-n, lo-n, la+n, lo+n)
zoom = 6
osm = smopy.Map(box, z=zoom)
npl = osm.fetch()
npl_np = np.array(npl)
plot(npl_np)

la, lo = 27.7, 85.316667
n = 1.0
box = (la-n, lo-n, la+n, lo+n)
zoom = 5
osm = smopy.Map(box, z=zoom)
npl = osm.fetch()
npl_np = np.array(npl)
plot(npl_np)

dataset.bounds

dataset.shape

import tifffile as tiff

type(dataset)

band1.shape

band1.nbytes / 1024**3

tiff.imsave('../data/worldview1_ntf/29APR15WV010500015APR29062253-P1BS-500308331010_01_P059.tif', band1)

band1.dtype

loaded = tiff.imread('../data/worldview1_ntf/29APR15WV010500015APR29062253-P1BS-500308331010_01_P059.tif')

tiff.imshow(loaded[:1000, :1000])









