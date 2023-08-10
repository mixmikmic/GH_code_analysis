get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np

rng = np.random.RandomState(1)
x = 10 * rng.rand(50)
y = np.sin(x) + 0.1 * rng.randn(50)

plt.scatter(x, y)

### Polynomial Functions

from sklearn.preprocessing import PolynomialFeatures
x = np.array([2,3,4])

#Intend to find a polynomial function of degree 3 representing the dataset
poly = PolynomialFeatures(3, include_bias=False)

poly.fit_transform(x[:,np.newaxis])

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
poly_model = make_pipeline(PolynomialFeatures(7),
                           LinearRegression())

poly_model3 = make_pipeline(PolynomialFeatures(3),
                           LinearRegression())

poly_model100 = make_pipeline(PolynomialFeatures(100),
                           LinearRegression())

#poly_model.fit(x[:, np.newaxis], y)

#xfit = np.linspace(0, 10, 1000)

#yfit = poly_model.predict(xfit[:, np.newaxis])

rng = np.random.RandomState(1)
x = 10 * rng.rand(50)
y = np.sin(x) + 0.1 * rng.randn(50)

poly_model.fit(x[:, np.newaxis], y)
poly_model3.fit(x[:, np.newaxis], y)
poly_model100.fit(x[:, np.newaxis], y)
xfit = np.linspace(0, 10, 1000)

yfit = poly_model.predict(xfit[:, np.newaxis])
yfit3 = poly_model3.predict(xfit[:, np.newaxis])
yfit100 = poly_model100.predict(xfit[:, np.newaxis])

plt.scatter(x, y)
plt.plot(xfit, yfit);
plt.plot(xfit,yfit3)
#plt.plot(xfit,yfit100)

poly_model.get_params()







