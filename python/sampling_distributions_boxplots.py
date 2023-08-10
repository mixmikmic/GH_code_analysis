get_ipython().magic('matplotlib inline')


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

import seaborn as sns
sns.set(style="ticks")

sns.set_context("talk")

N = 20
num_reps = 100
v = 1.0

true_mean = np.exp(v/2)  # Lognormal case

outcomes_mean = np.empty(num_reps)
outcomes_midrange = np.empty(num_reps)
outcomes_mle = np.empty(num_reps)

for i in range(num_reps):
    y = norm.rvs(size=N)
    x = np.exp(y)
    mu_hat = y.mean()
    sigma2_hat = y.var()
    outcomes_mle[i] = np.exp(mu_hat + sigma2_hat/2.0)
    outcomes_mean[i] = x.mean()
    outcomes_midrange[i] = 0.5 * (x.max() - x.min())


d = {'sample mean' : outcomes_mean,
        'mle' : outcomes_mle,
        'midrange' : outcomes_midrange}

df = pd.DataFrame(d)

fig, ax = plt.subplots()

sns.boxplot(data=df, orient='h', palette="PRGn")
sns.despine(offset=10, trim=True)

plt.show()





