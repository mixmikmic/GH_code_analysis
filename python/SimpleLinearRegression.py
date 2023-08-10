get_ipython().run_line_magic('matplotlib', 'inline')

#Import Libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

points = np.genfromtxt('data.csv', delimiter=',')

#Extract columns
x = np.array(points[:,0])
y = np.array(points[:,1])

#Plot the dataset
plt.scatter(x,y)
plt.xlabel('Hours of study')
plt.ylabel('Test scores')
plt.title('Dataset')
plt.show()

x=x.reshape(-1,1)
print x.shape
print y.shape
reg = linear_model.LinearRegression()
reg.fit(x,y)

plt.scatter(x, y,color='b')
plt.xlabel('Hours of study')
plt.ylabel('Test scores')
plt.title('Dataset')
plt.plot(x, reg.predict(x),color='k')

