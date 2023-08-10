import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

get_ipython().run_line_magic('matplotlib', 'inline')

#y=3x

x=[1,2,3,4,5]
y=[3,6,9,12,15]

x = np.array(x)
y = np.array(y)

model = LinearRegression()

model.fit(x.reshape(-1,1),y.reshape(-1,1))

model.predict(6)

plt.plot(x,y)
plt.scatter(x,y)

X = np.arange(1,20)

X

m, c = 1, 2
Y = m*X + c
Y

plt.plot(X,Y)

noise = np.random.random(len(X))
noise

ynoise = m*X + c + 3*noise

plt.scatter(X,ynoise)

model = LinearRegression()

model.fit(X.reshape(-1,1),ynoise.reshape(-1, 1))

# Slope of the model 
m_lreg = model.coef_
print(m_lreg)
print(m)

# Intercept
c_lreg = model.intercept_
print(c_lreg)
print(c)

model.predict(50)

yfit = m_lreg * X + c_lreg
print(yfit.shape)

# original points with noise
# plt.plot(x,)
plt.scatter(X, ynoise)

# Predicted line
plt.plot(X.reshape(-1, 1), yfit.reshape(-1,1),'g')



