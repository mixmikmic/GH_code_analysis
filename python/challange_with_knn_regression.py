get_ipython().run_line_magic('matplotlib', 'inline')
# Import libraries (Numpy, Tensorflow, matplotlib)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Create 1000 points following a function y=0.1 * x + 0.4 (i.e. y = W * x + b) with some normal random distribution:
df = pd.read_csv("./UN.csv")
df = df.dropna()
print(df)

# Separate the data point across axises

x_data = df['gdp'].values
x_data = np.reshape(x_data,(193,1))

y_data = df['infant.mortality'].values

# Plot and show the data points in a 2D space
plot.plot(x_data, y_data, 'ro', label='Original data')
plot.legend()
plot.show()

from sklearn import neighbors

# Parameters
n_neighbors = 8
weigths = "uniform" #"distance" 

# Define model
neigh = neighbors.KNeighborsRegressor(n_neighbors=2, weights=weigths)

# Fit regression model
neigh.fit(x_data, y_data) 

# Predict one value
print(neigh.predict([[20]]))

# Predict several examples and plot with original
T = np.linspace(0, 40000, 400)[:, np.newaxis]
ypredict = neigh.predict(T)
plt.figure(figsize=(22,7))
plt.scatter(x_data, y_data, c='r', label='data')
plt.plot(T, ypredict, c='g', label='prediction')
plt.axis('tight')
plt.legend()
plt.show()



