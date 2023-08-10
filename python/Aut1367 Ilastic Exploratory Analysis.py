## Following https://github.com/neurodata/ndio-demos/blob/master/Getting%20Started.ipynb
import ndio
ndio.version

import ndio.remote.neurodata as neurodata
nd = neurodata()

TOKEN = "Aut1367"
CHANNEL = "Aut1367_stitched"

TOKEN in public_tokens # Should *definitely* be true

## I see it in ndviz, so hopefully this is ok..  (it did work, although it's not in public_tokens, which makes sense)

## RUN 2: (RUN 1 had dimensions that were much too narrow)

query = {
    'token': TOKEN,
    'channel': CHANNEL,
    'x_start': 10000,
    'x_stop': 15000,
    'y_start': 10000,
    'y_stop': 15000,
    'z_start': 500,
    'z_stop': 505,
    'resolution': 0
}

aut_1367 = nd.get_cutout(**query)

get_ipython().magic('matplotlib inline')

print type(aut_1367)

from PIL import Image
print aut_1367.shape

import numpy as np
import scipy.misc

## if we have (i, j, k), we want (k, j, i)  (converts nibabel format to sitk format)
new_im = aut_1367.swapaxes(0,2) # just swap i and k

plane = 0;
for plane in (0, 1, 2, 3, 4, 5):
    output = np.asarray(new_im[plane])
    ## Save as TIFF for Ilastik
    scipy.misc.toimage(output).save('RAWoutfile' + 'aut1367_' + str(plane) + '.tiff')



