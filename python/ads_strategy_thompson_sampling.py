import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from thompson_sampling_algorithm import *
get_ipython().magic('matplotlib inline')

ads_data = pd.read_csv("Ads_CTR_Optimisation.csv")

ads_data.head()

ads_, total_result = thompson_sampling(ads_data.values)

total_result

plt.hist(ads_)
plt.title('Thompson sampling algorithm on ads strategy')
plt.xlabel('Different versions of ads')
plt.ylabel('Number of times ads were selected')
plt.show()

