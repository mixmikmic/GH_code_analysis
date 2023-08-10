import numpy as np # first load the needed libraries

np.random.seed(1234567) # This is to ensure repeatability in results

x = np.random.random(100)*20 # Generate a random set of 100 input variables
y =  np.array([ 2*v + 5 for v in x]) # Can genereate a new array by modifying each element in x
print(x[:10]) #print the first 10 values
print(y[:10]) #print the first 10 values

import matplotlib.pyplot as plt


plt.plot(x,y, 'b', label='True Line')
plt.legend()
plt.show()

y1 =  np.array([ v + (np.random.random()-0.5)*4 for v in y]) #this introduces some noise into the data
print(y1[:10])

x_train = x[:-30]
x_test = x[-30:]

y_train = y1[:-30]
y_test = y1[-30:]

import matplotlib.pyplot as plt # this is the library we will be using to make a simple 2-D scatter plot of our data

plt.plot(x_train,y_train,'x',label="Original Data")
plt.show()

plt.plot(x_train,y_train,'x',label="Original Data")
plt.plot(x,y, 'b', label='True Line')
plt.legend()
plt.show()

import scipy.stats as scs #the scipy library, and specifically the stats module.

gradient,intercept,r,p,err = scs.linregress(x_train,y_train) #estimate trend line

print("Estimated Gradient (c1) : %3.4f\nEstimated Intercept (c0) : %3.4f"%(gradient,intercept))
print("r^2 value %3.4f"%(r*r))
print("P-values : %3.4f"%(p))
print("Estimated Error : %3.4f"%(err))
estimated = gradient*x_train + intercept # generating predicted values for each sample in train set.
plt.plot(x_train,y_train,'x',label="Train Data")
plt.plot(x_test,y_test,'o',label="Test Data")
plt.plot(x,y, 'b', label='True Line')
plt.plot(x_train,estimated, 'r', label='Regression Line')
plt.legend()
plt.show()

print("Actual Line of Best Fit : y = 2x + 5")
print("Estimated Line of Best Fit : y = %3.4fx + %3.4f"%(gradient,intercept))

import math

mean_sqrd_err = 0.0
est_test = gradient*x_test + intercept # predicting values for each sample in test set.
for i in range(0,len(est_test)) :
    mean_sqrd_err += math.pow((est_test[i]- y_test[i]),2)
mean_sqrd_err = mean_sqrd_err/len(est_test)
print(mean_sqrd_err)



