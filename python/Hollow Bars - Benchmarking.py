get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from flowdec.nb import utils as nbutils 
from flowdec import data as fd_data
from flowdec import restoration as fd_restoration

acq = fd_data.load_bars()
acq.shape()

niter = 25
config = tf.ConfigProto(device_count={'GPU': 1})

get_ipython().run_cell_magic('timeit', '-n 10', 'import tensorflow as tf\nres = fd_restoration.richardson_lucy(acq, niter=niter, session_config=config)')

algo = fd_restoration.RichardsonLucyDeconvolver(acq.data.ndim).initialize()

get_ipython().run_cell_magic('timeit', '-n 10', 'res = algo.run(acq, niter=niter, session_config=config)')

nbutils.plot_rotations(algo.run(acq, niter=niter, session_config=config).data)

