from sklearn.datasets import load_boston
import pandas as pd, numpy as np
get_ipython().magic('matplotlib inline')
get_ipython().magic('pylab inline')
import seaborn  as sns 
from sklearn.datasets import load_boston
import pylab as pl
import matplotlib.pyplot as plt



from sklearn import datasets, linear_model
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, SGDRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.learning_curve import learning_curve
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import PCA

boston = load_boston()
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df['target'] = boston.target

df.head()

X = df[['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
       'PTRATIO', 'B', 'LSTAT']]
Y=df['target']

#  relative feature importances



X = df[['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
       'PTRATIO', 'B', 'LSTAT']]

Y = np.array(Y).astype(int)



clf = ExtraTreesClassifier()
clf = clf.fit(X, Y)
clf.feature_importances_ 

df_feature = pd.DataFrame(clf.feature_importances_ )
df_feature.index =[['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
       'PTRATIO', 'B', 'LSTAT']]
pd.DataFrame(df_feature).plot(kind='bar',figsize=(20, 10),fontsize=18) 
pyplot.xlabel('features')
pyplot.ylabel('feature_importances')

df.corr()

ax = sns.heatmap(df.corr(), cmap="YlGnBu")

sns.jointplot(df['CRIM'], df['target'],  kind="reg", size=8)

sns.jointplot(df['RM'], df['target'],  kind="reg", size=8)

sns.jointplot(df['AGE'], df['target'],  kind="reg", size=8)

sns.jointplot(df['LSTAT'], df['target'],  kind="reg", size=8)

pd.scatter_matrix(df, figsize=[10, 10], alpha=0.2, diagonal='kde')

boston = load_boston()
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df['target'] = boston.target

X = df[['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
       'PTRATIO', 'B', 'LSTAT']]
Y=df['target']

regr = linear_model.LinearRegression()
regr.fit(X, Y)

print (pd.DataFrame({'predict':regr.predict(X), 'actual':Y, 'error':Y-regr.predict(X)}))



print ('score :', regr.score(X,Y))
print ('intercept_ : ' , regr.fit(X, Y).intercept_, '\n', 'coef_ : ', regr.fit(X, Y).coef_)


# Plot outputs
plt.scatter(regr.predict(X), Y,  color='blue')
plt.plot([0,50],[0,50], 'g-')


plt.show()

# remove possible outliers 

df_ = df[df.target < 50]

X = df_[['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
       'PTRATIO', 'B', 'LSTAT']]
Y=df_['target']

regr = linear_model.LinearRegression()
regr.fit(X, Y)


# Plot outputs
plt.scatter(regr.predict(X), Y,  color='blue')
plt.plot([0,50],[0,50], 'g-')



print ('score :', regr.score(X,Y))
print ('intercept_ : ' , regr.fit(X, Y).intercept_, '\n', 'coef_ : ', regr.fit(X, Y).coef_)

# cross_validation


X = df[['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
       'PTRATIO', 'B', 'LSTAT']]
Y=df['target']
r_scores = []
random_state=range(1,100)
for r in random_state:
    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=r)

    regr.fit(X_train,  y_train)
    r_scores.append(regr.score(X_test,y_test))
    

plt.plot(random_state, r_scores)
plt.xlabel('Value of r for LinearRegression')
plt.ylabel('Cross-Validated Accuracy')


# http://statsmodels.sourceforge.net/devel/examples/notebooks/generated/ols.html
# import statsmodels.api as sm
# ref : http://marcharper.codes/2016-06-14/Linear+Regression+with+Statsmodels+and+Scikit-Learn.html

# import statsmodels.api as sm

import statsmodels.api as sm



model = sm.OLS(Y, X)
results = model.fit()
# Statsmodels gives R-like statistical output
print(results.summary())





