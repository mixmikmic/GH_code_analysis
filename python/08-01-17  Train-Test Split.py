import pandas as pd

from sklearn.datasets import load_boston
boston = load_boston()
X, y = boston.data, boston.target
print (X.shape, y.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=5)

print (X_train.shape, X_test.shape)

print (y_train.shape, y_test.shape)

pd.DataFrame(X_train).head()

pd.DataFrame(y_train).head()



