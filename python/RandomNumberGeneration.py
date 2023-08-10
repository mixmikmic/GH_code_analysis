get_ipython().run_line_magic('matplotlib', 'inline')
import numpy.random as random
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

df = pd.DataFrame(random.rand(10, 5), columns=['A', 'B', 'C', 'D', 'E'])

df

mu, sigma = 0, 0.1
s = random.normal(mu, sigma, 1000)

count, bins, ignored = plt.hist(s, 30, normed=True, alpha=0.7)

w = 1 * random.weibull(2, 2000)
count, bins, ignored = plt.hist(w, 30, normed=True, alpha=0.7)

