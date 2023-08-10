# Adapted from http://matplotlib.org/users/pyplot_tutorial.html

import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

mu, sigma = 100, 15
x = mu + sigma * np.random.randn(10000)

# the histogram of the data
n, bins, patches = plt.hist(x, 50, normed=1, facecolor='red', alpha=0.75)


plt.xlabel('Smarts')
plt.ylabel('Probability')
plt.title('Histogram of IQ')
plt.text(60, .025, r'$\mu=100,\ \sigma=15$', fontsize=14)
plt.axis([40, 160, 0, 0.03])
plt.grid(True)
plt.savefig('plot_example.eps', format='eps', dpi=300)
plt.show()

