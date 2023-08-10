import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import math

get_ipython().run_line_magic('matplotlib', 'inline')

np.random.seed(6)

population_ages1 = stats.poisson.rvs(loc=18, mu=35, size=150000)
population_ages2 = stats.poisson.rvs(loc=18, mu=10, size=100000)
population_ages = np.concatenate((population_ages1, population_ages2))

minnesota_ages1 = stats.poisson.rvs(loc=18, mu=30, size=30)
minnesota_ages2 = stats.poisson.rvs(loc=18, mu=10, size=20)
minnesota_ages = np.concatenate((minnesota_ages1, minnesota_ages2))

print(population_ages.mean())
print(minnesota_ages.mean())

stats.ttest_1samp(a=minnesota_ages, popmean=population_ages.mean())

stats.t.ppf(q=0.025, df=49)

stats.t.ppf(q=0.975, df=49)

stats.t.cdf(x=-2.5742, df=49) * 2

sigma = minnesota_ages.std() / math.sqrt(50)

stats.t.interval(0.95, df=49, loc=minnesota_ages.mean(), scale=sigma)

stats.t.interval(0.99, df=49, loc=minnesota_ages.mean(), scale=sigma)

np.random.seed(12)
wisconsin_ages1 = stats.poisson.rvs(loc=18, mu=33, size=30)
wisconsin_ages2 = stats.poisson.rvs(loc=18, mu=13, size=20)
wisconsin_ages = np.concatenate((wisconsin_ages1, wisconsin_ages2))

print(wisconsin_ages.mean())

stats.ttest_ind(a=minnesota_ages, b=wisconsin_ages, equal_var=False)

np.random.seed(11)

before = stats.norm.rvs(scale=30, loc=250, size=100)

after = before + stats.norm.rvs(scale=5, loc=-1.25, size=100)

weight_df = pd.DataFrame({'weight_before': before,
                          'weight_after': after,
                          'weight_change': after - before})

weight_df.describe().T

stats.ttest_rel(a=before, b=after)

plt.figure(figsize=(12,10))

plt.fill_between(x=np.arange(-4,-2,0.01),
                 y1=stats.norm.pdf(np.arange(-4,-2,0.01)),
                 facecolor='red',
                 alpha=0.35)

plt.fill_between(x=np.arange(-2,2,0.01),
                 y1=stats.norm.pdf(np.arange(-2,2,0.01)),
                 facecolor='white',
                 alpha=0.35)

plt.fill_between(x=np.arange(2,4,0.01),
                 y1=stats.norm.pdf(np.arange(2,4,0.01)),
                 facecolor='red',
                 alpha=0.5)

plt.fill_between(x=np.arange(-4,-2,0.01),
                 y1=stats.norm.pdf(np.arange(-4,-2,0.01), loc=3, scale=2),
                 facecolor='white',
                 alpha=0.35)

plt.fill_between(x=np.arange(-2,2,0.01),
                 y1=stats.norm.pdf(np.arange(-2,2,0.01), loc=3, scale=2),
                 facecolor='blue',
                 alpha=0.35)

plt.fill_between(x=np.arange(2,10,0.01),
                 y1=stats.norm.pdf(np.arange(2,10,0.01), loc=3, scale=2),
                 facecolor='white',
                 alpha=0.35)

plt.text(x=-0.8, y=0.15, s='Null Hypothesis')

plt.figure(figsize=(12,10))

plt.fill_between(x=np.arange(-4,-2,0.01), 
                 y1= stats.norm.pdf(np.arange(-4,-2,0.01)) ,
                 facecolor='red',
                 alpha=0.35)

plt.fill_between(x=np.arange(-2,2,0.01), 
                 y1= stats.norm.pdf(np.arange(-2,2,0.01)) ,
                 facecolor='white',
                 alpha=0.35)

plt.fill_between(x=np.arange(2,4,0.01), 
                 y1= stats.norm.pdf(np.arange(2,4,0.01)) ,
                 facecolor='red',
                 alpha=0.5)

plt.fill_between(x=np.arange(-4,-2,0.01), 
                 y1= stats.norm.pdf(np.arange(-4,-2,0.01),loc=3, scale=2) ,
                 facecolor='white',
                 alpha=0.35)

plt.fill_between(x=np.arange(-2,2,0.01), 
                 y1= stats.norm.pdf(np.arange(-2,2,0.01),loc=3, scale=2) ,
                 facecolor='blue',
                 alpha=0.35)

plt.fill_between(x=np.arange(2,10,0.01), 
                 y1= stats.norm.pdf(np.arange(2,10,0.01),loc=3, scale=2),
                 facecolor='white',
                 alpha=0.35)

plt.text(x=-0.8, y=0.15, s= "Null Hypothesis")
plt.text(x=2.5, y=0.13, s= "Alternative")
plt.text(x=2.1, y=0.01, s= "Type 1 Error")
plt.text(x=-3.2, y=0.01, s= "Type 1 Error")
plt.text(x=0, y=0.02, s= "Type 2 Error")

lower_quantile = stats.norm.ppf(0.025)
upper_quantile = stats.norm.ppf(0.975)

# Area under alternative, to the left the lower cutoff value
low = stats.norm.cdf(lower_quantile, loc=3, scale=2)

# Area under alternative, to the left of the upper cutoff value
upper = stats.norm.cdf(upper_quantile, loc=3, scale=2)

# Area under the alternative, between the upper and lower cutoffs (Type II Error)
upper-low

