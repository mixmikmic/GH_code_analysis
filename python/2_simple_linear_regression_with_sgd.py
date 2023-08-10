import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
get_ipython().run_line_magic('matplotlib', 'inline')

# Load the diabetes dataset
diabetes = datasets.load_diabetes()

# Use only one feature
diabetes_X = diabetes.data[:, np.newaxis, 2]

# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]

NO_OF_EPOCHS = 10000
b0 = 0
b1 = 0
LEARNING_RATE = 0.005

for idx in range(NO_OF_EPOCHS):
    sum_error = 0
    for x_i, y_i in zip(diabetes_X_train, diabetes_y_train):
        y_hat = b0 + b1 * x_i
        error = y_hat - y_i 
        b0 = b0 - LEARNING_RATE * error
        b1 = b1 - LEARNING_RATE * error * x_i
        sum_error += error**2
    if idx % 1000 == 0:
        print('epoch=%d, error=%.3f, coef_0=%.3f, coef_1=%.3f' % (idx, sum_error, b0, b1))

print('coefs', b0, b1)

y_pred = diabetes_X_test * b1 + b0

sse = 0
for y_p, y_o in zip(y_pred, diabetes_y_test):
    sse += (y_p - y_o) ** 2

mse = sse/len(y_pred)

mse

plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
plt.plot(diabetes_X_test, y_pred, color='blue', linewidth=1)

plt.xticks(())
plt.yticks(())

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)

# Make predictions using the testing set
diabetes_y_pred = regr.predict(diabetes_X_test)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(diabetes_y_test, diabetes_y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(diabetes_y_test, diabetes_y_pred))

# Plot outputs
plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=1)

plt.xticks(())
plt.yticks(())



