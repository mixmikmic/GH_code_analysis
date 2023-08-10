get_ipython().magic('matplotlib notebook')

import matplotlib.pyplot as plt
import sys
import pandas as pd
import matplotlib

print(u'▶ Python version ' + sys.version)
print(u'▶ Pandas version ' + pd.__version__)
print(u'▶ Matplotlib version ' + matplotlib.__version__)

import numpy as np
values = np.random.rand(100)

df = pd.DataFrame(data=values, columns=['RandomValue'])
df.head(10)

df.plot()

