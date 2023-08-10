import numpy as np
import matplotlib.pyplot as plt
import my_module
get_ipython().magic('matplotlib inline')

x = np.array([1, 2, 4, 3, 5])
y = np.array([1, 3, 3, 2, 5])

sr = my_module.SimpleRegression()
sr.fit(x, y)
print(sr.intercept_)
print(sr.coef_)

# Plot outputs
plt.scatter(x, y,  color='black')
plt.plot(x, sr.predict(x), color='blue', linewidth=3)
plt.show()

