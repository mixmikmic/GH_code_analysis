import sys
print(sys.version)

import numpy
#import pylab
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

from algorithms import gamma_evaluation 

# Reference data with (2, 1) mm resolution
reference = numpy.random.random((128, 256))
#reference = numpy.abs(reference)
reference /= reference.max()
reference *= 100
reference -= 50

# Sample data with a %3 shift on the reference
sample = reference * 1.03

# Perform gamma evaluation at 4mm, 2%, resoution x=2, y=1
gamma_map = gamma_evaluation(sample, reference, 4., 2., (2, 1), signed=True)

plt.imshow(gamma_map, cmap='RdBu_r', aspect=2, vmin=-2, vmax=2)
plt.colorbar()
plt.show()

# Two subplots, unpack the axes array immediately
f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)

ax1.imshow(reference)
ax1.set_title('reference')

ax2.imshow(sample)
ax2.set_title('sample')

ax3.imshow(gamma_map, cmap='RdBu_r', aspect=2, vmin=-2, vmax=2)
ax3.set_title('gamma_map')
plt.show()



