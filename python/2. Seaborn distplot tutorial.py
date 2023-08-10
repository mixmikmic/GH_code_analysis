import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')

num = np.random.randn(150)
sns.distplot(num,color ='green')


label_dist = pd.Series(num,name = " Variable x")
sns.distplot(label_dist,color = "red")

# Plot the distribution with a kenel density. estimate and rug plot:

sns.distplot(label_dist,hist = False,color = "red")

# Plot the distribution with a kenel density estimate and rug plot:

sns.distplot(label_dist,rug = True,hist = False,color = "red")

# Plot the distribution with a histogram and maximum likelihood gaussian distribution fit:

from scipy.stats import norm
sns.distplot(label_dist, fit=norm, kde=False)

sns.distplot(label_dist, vertical =True)



