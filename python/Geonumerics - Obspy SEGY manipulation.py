get_ipython().magic('cd fatiando_dev')
get_ipython().magic('cd ..')
from fatiando.seismic import io
import numpy as np
from scipy import ndimage
get_ipython().magic('cd gngdata')
get_ipython().magic('cd OtherSegy')

cloudtraces, bheader = io.readSEGY("cloudpspin635.segy")
dt = bheader.binary_file_header.sample_interval_in_microseconds*0.001
nsamples, ntrac = np.shape(cloudtraces)
# apply moving average over trace samples size 10 samples
# create a trace and append it to stream
cloudtraces_smooth = ndimage.convolve(cloudtraces, np.ones((10,10))/(10*10))
#cloudtraces_smooth = ndimage.gaussian_filter(cloudtraces, 10)
io.writeSEGY(cloudtraces_smooth.astype(np.float32), 'cloudpspin635_smooth.segy')    

cloudtraces_smooth, _ = io.readSEGY("cloudpspin635_smooth.segy")

import matplotlib.pylab as pylab
pylab.rcParams['figure.figsize'] = 18, 13  # that's default image size for this interactive session
fig = figure()
ax = fig.add_subplot(1,2,1)
ax.imshow(cloudtraces, interpolation='bicubic', cmap=pylab.cm.gray, vmin=-20, vmax=20, aspect=0.5, origin='upper', extent=[0,ntrac*10,0,nsamples*dt])
ax.set_title('Cloudpsin',fontsize=10)
ax = fig.add_subplot(1,2,2)
ax.imshow(cloudtraces_smooth, interpolation='bicubic', cmap=pylab.cm.gray, vmin=-20, vmax=20, aspect=0.5, origin='upper', extent=[0,ntrac*10,0,nsamples*dt])
ax.set_title('Cloudpsin moving average',fontsize=10)
pylab.rcParams['figure.figsize'] = 4, 4  # that's default image size for this interactive session

pylab.rcParams['figure.figsize'] = 12, 6  # that's default image size for this interactive session
from fatiando.vis import mpl
from obspy.segy.core import readSEGY
# Marmousi 240 shots
# 96 receivers 750 samples per trace
#%!ls
get_ipython().magic('cd gngdata')
get_ipython().magic('cd Marmousi')

from obspy.segy.core import readSEGY

traces = readSEGY("marmousi_shots.segy")
mpl.seismic_wiggle(traces[:96], ranges=(0,960), scale=0.001)

import numpy as np
import sys
from obspy import Trace, Stream
from obspy.core.trace import Stats

arraytraces = np.require(np.random.random((20,20)), dtype=np.float32)
stream = Stream()
traceheader = Stats()
traceheader.delta = 0.004
for array in arraytraces:
    trace = Trace(data=array, header=traceheader)
    stream.append(trace)
    
# additionally for setting non defaults segy (ascii and binary headers)
from obspy.core import AttribDict
from obspy.segy.segy import SEGYBinaryFileHeader
stream.stats = AttribDict()
stream.stats.textual_file_header = 'Made by me!'
stream.stats.binary_file_header = SEGYBinaryFileHeader()
stream.stats.binary_file_header.number_of_samples_per_data_trace = 20

get_ipython().system('pwd')
get_ipython().magic('cd gngdata')
stream.write("random.sgy", format="SEGY", data_encoding=1,
             byteorder=sys.byteorder)



