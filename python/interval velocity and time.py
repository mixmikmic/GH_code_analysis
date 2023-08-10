get_ipython().run_line_magic('pylab', 'inline')

from matplotlib import pylab
import numpy as np

ax = pylab.subplot(111)
ax.plot(1480.+numpy.arange(0, 3000)*0.008)
ax.plot(1491.+numpy.zeros(3000))
pylab.show()

ax = pylab.subplot(111)
z = numpy.arange(0, 3000)
tz_average=z/1491.
ax.plot(tz_average)
tz_integrated_linear = (1./0.008)*np.log(np.abs(0.008*z+1480)/(np.abs(1480)))
ax.plot(tz)
pylab.show()

ax = pylab.subplot(111)
ax.plot(np.abs(tz_average-tz_integrated_linear)*1000)
pylab.ylabel('time difference (ms)')



