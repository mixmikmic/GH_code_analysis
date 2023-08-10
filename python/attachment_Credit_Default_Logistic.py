get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Load Credit Default File
cred_df = pd.read_csv("C:/Users/User/Documents/machineLearning/default.csv")

cred_df.info()

sns.boxplot(x='default', y='balance', data=cred_df)
plt.show()

sns.boxplot(x='default', y='income', data=cred_df)
plt.show()

sns.lmplot(x='balance', y='income', hue = 'default', data=cred_df, aspect=1.5, ci = None, fit_reg = False)
plt.show()

pd.crosstab(cred_df['default'], cred_df['student'], rownames=['Default'], colnames=['Student'])

# Convert Categorical to Numerical
default_dummies = pd.get_dummies(cred_df.default, prefix='default')
default_dummies.drop(default_dummies.columns[0], axis=1, inplace=True)
cred_df = pd.concat([cred_df, default_dummies], axis=1)
cred_df.head()

# Try simple linear regression on the data
sns.lmplot(x='balance', y='default_Yes', data=cred_df, aspect=1.5, ci = None, fit_reg = True)

# Building Linear Regression Model
from sklearn.linear_model import LinearRegression

X = cred_df[['balance']]
y = cred_df['default_Yes']

linreg = LinearRegression()
linreg.fit(X, y)

print(linreg.coef_)
print(linreg.intercept_)

# Building the Logistic Regression Model
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(C=1e42) # Set Large C value for low regularization
logreg.fit(X, y)

print(logreg.coef_)
print(logreg.intercept_)

y_pred = logreg.predict_proba(X)
plt.scatter(X.values, y_pred[:,1])
plt.scatter(X.values, y)
plt.show()

x = X.values
y_hat = linreg.intercept_ + linreg.coef_ * x
sig_y_hat = np.exp(y_hat)/(1+np.exp(y_hat))

plt.plot(x, y_hat, color='red')
plt.scatter(x, y, color = 'blue')
plt.scatter(x, sig_y_hat, color = 'green')
plt.show()

sns.lmplot(x='balance', y='default_Yes', data=cred_df, aspect=1.5, ci = None, fit_reg = True)

#plt.plot(x, sig_y_hat)
x = X
y_hat = logreg.intercept_ + logreg.coef_ * x
sig_y_hat = np.exp(y_hat)/(1+np.exp(y_hat))

plt.scatter(x=x, y=sig_y_hat)
plt.scatter(x=x, y=y)
#plt.scatter(x=x, y=y_hat)
plt.show()

x = np.arange(-10,10,0.1)
sig_x = np.exp(x)/(1+np.exp(x))
df = pd.DataFrame(list(zip(x,sig_x)), columns = ['x', 'sig_x'])
sns.lmplot(x='x', y='sig_x', data=df, aspect=1.5, ci = None, fit_reg = False)
plt.show()

