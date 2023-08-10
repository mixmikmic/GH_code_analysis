import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

import numpy as np

x = np.linspace(0,10,100)

#Setting the background of the graphs
plt.style.use('seaborn-whitegrid')

plt.plot(x, np.sin(x), color='yellow')
plt.plot(x, np.sin(x - 1), color='green')
plt.plot(x, np.sin(x - 2), color='red')
plt.plot(x, np.sin(x - 3), color='blue')

plt.plot(x, x + 0, linestyle='solid')
plt.plot(x, x + 1, linestyle='dashed')
plt.plot(x, x + 2, linestyle='dashdot')
plt.plot(x, x + 3, linestyle='dotted');

plt.plot(x, x + 0, '-g')  # solid green
plt.plot(x, x + 1, '--c') # dashed cyan
plt.plot(x, x + 2, '-.k') # dashdot black
plt.plot(x, x + 3, ':r');  # dotted red

plt.plot(x, np.sin(x), label='sin')
plt.plot(x, np.cos(x), label='cos')
plt.xlim(-5,11)
plt.ylim(-1.5, 1.5)
plt.xlabel("x-axis information")
plt.ylabel("y-axis information")
plt.legend()



