import numpy as np
np.random.seed(0)
x = np.random.random(size=(15, 1))
y = 3 * x.flatten() + 2 + np.random.randn(15)

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.plot(x, y, 'o')

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x, y)

print("Model intercept is", model.intercept_)
print("Model slope is", model.coef_[0])

x_unseen = 0.78
model.predict(x_unseen)

# create predictions which we will use to generate our line
X_fit = np.linspace(0, 1, 100)[:, np.newaxis]
y_fit = model.predict(X_fit)
# plot the data
plt.plot(x.flatten(), y, 'o')
# plot the line
plt.plot(X_fit, y_fit)

import pandas as pd
df = pd.read_csv("advertising.csv", index_col=0)
df.head()

model = LinearRegression()
# create a copy of the data frame, with a single input variable
x = df[["TV"]]
# fit the model based on the original response variable
model.fit(x,df["Sales"])

print("Model intercept is", model.intercept_)
print("Model slope is", model.coef_[0])

test_x = x[0:5]
model.predict(test_x)

df["Sales"][0:5]

plt.scatter(df["TV"], df["Sales"])
plt.xlabel("TV Budget Spend")
plt.ylabel("Sales")
# add the predictions from regression
plt.plot(df["TV"], model.predict(x), color="red")
plt.show()

np.mean((df["Sales"] - model.predict(x)) ** 2)

# extract the relevant column
x = df[["Newspaper"]]
# build the model
model = LinearRegression()
model.fit(x,df["Sales"])

np.mean((df["Sales"] - model.predict(x)) ** 2)

# separate the training test data - normally we would do this randomly
train_df = df[0:160]
test_df = df[160:200]
train_x = train_df[["Newspaper"]]
test_x = test_df[["Newspaper"]]

# only build a model on the training set
model = LinearRegression()
model.fit(train_x,train_df["Sales"])

model.predict(test_x)

np.mean((test_df["Sales"] - model.predict(test_x)) ** 2)

df = pd.read_csv("advertising.csv", index_col=0)
# we remove the sales column that we are going to predict
x = df.drop("Sales",axis=1)
x.head()

model = LinearRegression()
model.fit(x,df["Sales"])

print("Model intercept is", model.intercept_)
print("Model slope is", model.coef_)

test_x = x[0:1]
print(test_x)
print("Predicted Sales = %.2f" % model.predict(test_x))
print("Actual Sales = %.2f" % df["Sales"].iloc[0])

unseen_X = np.array( [ [ 140.0, 45.3, 70.5 ], [ 70.0, 84.62, 98.95 ] ] )
unseen_X

model.predict( unseen_X )

