import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from upper_confidence_bound import *
get_ipython().magic('matplotlib inline')

ads_data = pd.read_csv("Ads_CTR_Optimisation.csv")

ads_data.head()

ads_, total_result = upper_confidence_bound(ads_data.values)

total_result

plt.hist(ads_)
plt.show()

