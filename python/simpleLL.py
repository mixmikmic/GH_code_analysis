import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

x=np.array([1,2,3,4,5,6])
y=np.array([3,5,8,10,11,14])

slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
fitLine = slope * x + intercept
#print fitLine

plt.scatter(x, y)
plt.plot(x, fitLine, c='r')
plt.show()

slope

intercept

std_err



