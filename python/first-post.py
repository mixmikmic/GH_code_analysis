get_ipython().magic('matplotlib inline')

import random

import matplotlib.pyplot as plt

# Turn on retina mode
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('retina')

random.randint(0,100)

plt.hist([10,5,7,10,1,1,2,3,5]);

