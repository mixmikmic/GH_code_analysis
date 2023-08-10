import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

cigarettes = pd.read_table("./data/cigarettes.txt")
cigarettes.head()

# get the correlation
r = np.corrcoef(cigarettes.tar, cigarettes.nicotine)[0][1]
r

# calculate b_1 (slope)
b1 = r * (cigarettes.nicotine.std() / cigarettes.tar.std())
b1

# calculate b_0 (intercept)
b0 = cigarettes.nicotine.mean() - b1 * cigarettes.tar.mean()
b0

# data points
plt.scatter(cigarettes.tar, cigarettes.nicotine)

# regression line
yhat = b0 + b1 * cigarettes.tar
plt.plot(cigarettes.tar, yhat)

plt.xlabel('tar')
plt.ylabel('nicotine')
plt.title('cigarettes - tar vs. nicotine regression line');

residuals = cigarettes.nicotine - yhat
residuals.head()

plt.scatter(cigarettes.tar, residuals)
plt.ylabel("residuals (nicotine)")
plt.xlabel("tar")
plt.title("tar~nicotine regression : residuals");

se = np.sqrt((residuals.pow(2).sum() / (residuals.size - 2)))
se

plt.hist(residuals)
plt.title("residuals histogram");

