import astropy.io.ascii as ascii
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import astropy.visualization as viz
import matplotlib.pyplot as plt
import astropy.io.fits as fits
import astropy.wcs as WCS # you need this astropy module for processing fits files
import matplotlib as mpl
import numpy as np
from tqdm import tqdm
from astropy.table import Table, join
get_ipython().run_line_magic('matplotlib', 'inline')
from scipy.ndimage import interpolation
from skimage.transform import rescale, resize, downscale_local_mean
import h5py

data = np.load('/home/thouis/xorastro/big-data/imgcube-uint8-all.npy', mmap_mode='r')
h5 = h5py.File('imgcube.h5', 'w')
chunked = h5.create_dataset('imgcube', shape=data.shape, dtype=data.dtype, chunks=(data.shape[0], data.shape[1], 1024), compression='lzf')

num_images = data.shape[2]
for idx in tqdm(range(0, num_images, 1024)):
    end = min(idx + 1024, num_images)
    chunked[:, :, idx:end] = data[:, :, idx:end]
h5.close()

h5 = h5py.File('imgcube.h5', 'r')
data = h5['imgcube']
h5new = h5py.File('imgcube2.h5', 'w')
chunked = h5new.create_dataset('imgcube', shape=data.shape, dtype=data.dtype, chunks=(data.shape[0], data.shape[1], 1), compression='lzf')
num_images = data.shape[2]
for idx in tqdm(range(0, num_images, 64)):
    end = min(idx + 64, num_images)
    chunked[:, :, idx:end] = data[:, :, idx:end]
h5.close()
h5new.close()

import h5py
d = h5py.File('big-data/imgcube-uint8-all.h5')['imgcube']
d.shape

from astropy.table import Table
metadata = Table.read('/home/thouis/xorastro/big-data/joined_meta.tsv', format='ascii.tab')
color = np.array(metadata[:]['g'] - metadata[:]['i'])

color.shape

hist(np.clip(color[:int(247590* 0.8)], 0, 3), 'auto');

hist(np.clip(color[int(247590* 0.8):int(247590* 0.9)], 0, 3), 'auto');

hist(np.clip(color[int(247590* 0.9):], 0, 3), 'auto');



