# Dependencies
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import math
import seaborn as sns

# Generating a Random Distribution
normal_distribution = stats.gennorm.rvs(1, size=10000)
plt.scatter(np.arange(0, 10000), normal_distribution)
plt.show()

# Visualize Data as Histogram
plt.hist(normal_distribution, 50, facecolor='blue', alpha=0.75)
plt.show()

# Generating a Random Distribution
poisson_distribution = stats.poisson.rvs(loc=10, mu=35, size=10000)
plt.scatter(np.arange(0, 10000), poisson_distribution)
plt.show()

# Visualize as Histogram
plt.hist(poisson_distribution, 50, facecolor='green', alpha=0.75)
plt.show()

# Generating a Random Distribution
exponential_distribution = stats.expon.rvs(size=10000)
plt.scatter(np.arange(0, 10000), exponential_distribution)
plt.show()

# Visualize as Histogram
plt.hist(exponential_distribution, 50, facecolor='green', alpha=0.75)
plt.show()



