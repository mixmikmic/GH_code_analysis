get_ipython().magic('matplotlib inline')

import numpy as np
import matplotlib.pyplot as plt

greyhounds = 500
labs = 500

grey_height = 28 + 4 * np.random.randn(greyhounds)
lab_height = 24 + 4 * np.random.randn(labs)

# Note on np.random.nrand()
# distribution of mean 0 and variance 1
# returns number of integers passed
np.random.randn(4)

grey_height[:5]

plt.hist([grey_height, lab_height], stacked=True, color=['r', 'b'])
plt.show()

