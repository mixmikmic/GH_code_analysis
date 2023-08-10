get_ipython().magic('matplotlib inline')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.graphics.factorplots import interaction_plot

np.random.seed(12345)
weight = pd.Series(np.repeat(['low', 'hi', 'low', 'hi'], 15), name='weight')
nutrition = pd.Series(np.repeat(['lo_carb', 'hi_carb'], 30), name='nutrition')
days = np.log(np.random.randint(1, 30, size=60))

fig, ax = plt.subplots(figsize=(6, 6))
fig = interaction_plot(x=weight, trace=nutrition, response=days, 
                       colors=['red', 'blue'], markers=['D', '^'], ms=10, ax=ax)

