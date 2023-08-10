get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
from flowdec.nb import utils as nbutils 
from flowdec import data as fd_data

# Load volume downsampled to 25%
acq = fd_data.bars_25pct()
acq.shape()

nbutils.plot_rotations(acq.data)

import tensorflow as tf
from flowdec import restoration as fd_restoration

res = fd_restoration.richardson_lucy(
    acq, niter=25, 
    # Disable GPUs for this since on windows at least, repeat runs of TF graphs
    # on GPUs in a jupyter notebook do not go well (crashes with mem allocation errors)
    session_config=tf.ConfigProto(device_count={'GPU': 0})
)
res.shape

nbutils.plot_rotations(res)

