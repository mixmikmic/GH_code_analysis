#-- load python packages
import numpy as np
from matplotlib import pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

#-- create x-values
x1 = np.arange(0,8,1)
x2 = np.arange(100)

#-- create y-values
data   = np.arange(1,40,5)
linear = np.arange(100)
square = [v * v for v in np.arange(0,10,0.1)]

plt.plot(x1, data,   "ob")         # marker o=open circle, b=black
plt.plot(x2, linear, "g")          # no marker, g=green
plt.plot(x2, square, "+r")         # marker +=cross, r=red

plt.legend(('data','linear','square'), loc='upper left')

plt.title('Title string')
plt.xlabel('x-axis label')
plt.ylabel('y-axis label')

plt.show()

