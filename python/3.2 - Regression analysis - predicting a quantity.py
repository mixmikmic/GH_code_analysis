from sklearn.datasets import load_boston

boston = load_boston()

print(boston.DESCR)

boston.data[0]

boston.feature_names

import pandas as pd

data = pd.DataFrame(boston.data, columns=boston.feature_names)

data.head()

data['PRICE'] = boston.target

data.head()

X = data[['RM']]  # only one feature first
Y = data['PRICE']

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, Y_train)

Y_prediction = model.predict(X_test)

Y_prediction[0]

Y_test.values[0]

get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt

plt.scatter(Y_test, Y_prediction)
plt.xlabel("Prices: $Y_{test}$")
plt.ylabel("Predicted prices: $Y_{predicted}$")
plt.title("Prices vs Predicted prices")

from sklearn.metrics import mean_squared_error

mse = mean_squared_error(Y_test, Y_prediction)

mse

X = data.drop('PRICE', axis=1)  # all features

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

model = LinearRegression()
model.fit(X_train, Y_train)

Y_prediction = model.predict(X_test)

plt.scatter(Y_test, Y_prediction)
plt.xlabel("Prices: $Y_i$")
plt.ylabel("Predicted prices: $\hat{Y}_i$")
plt.title("Prices vs Predicted prices: $Y_i$ vs $\hat{Y}_i$")

mse = mean_squared_error(Y_test, Y_prediction)

mse



