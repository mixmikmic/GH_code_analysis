# print all the outputs in a cell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# imports
import pandas as pd
import matplotlib.pyplot as plt
import xlrd
from math import log

# this allows plots to appear directly in the notebook
get_ipython().magic('matplotlib inline')

#data = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv', index_col=0)
#data1 = pd.read_csv("union.csv", low_memory=False)
#data1 = pd.read_excel("union2.xlsx")
AllCleanedApps = pd.read_csv("CleanedApps.csv")
SeventyFivepercentdata = pd.read_csv("75percent-data.csv")
SeventyFivepercentdata.shape
AllCleanedApps.shape

SeventyFivepercentdataWithoutLogs = SeventyFivepercentdata.copy()
SeventyFivepercentdata.head(10)

#Change the table values to log values
import math
def toLog(x):
    y = x
    if x == 0:
        y = 0.1
    return math.log(y)


AllCleanedApps["Downloads"] = AllCleanedApps.Downloads.apply(toLog)
AllCleanedApps["Size"] = AllCleanedApps.Size.apply(toLog)
#AllCleanedApps["Price"] = AllCleanedApps.Price.apply(toLog)
#AllCleanedApps["AvgR"] = AllCleanedApps.AvgR.apply(toLog)
AllCleanedApps["Raters"] = AllCleanedApps.Raters.apply(toLog)

SeventyFivepercentdata["Downloads"] = SeventyFivepercentdata.Downloads.apply(toLog)
SeventyFivepercentdata["Size"] = SeventyFivepercentdata.Size.apply(toLog)
#SeventyFivepercentdata["Price"] = SeventyFivepercentdata.Price.apply(toLog)
#SeventyFivepercentdata["AvgR"] = SeventyFivepercentdata.AvgR.apply(toLog)
SeventyFivepercentdata["Raters"] = SeventyFivepercentdata.Raters.apply(toLog)

import statsmodels.formula.api as smf


SeventyFivepercentdataWithoutLogsAnswer = smf.ols(formula='Downloads ~ Price + AvgR + Raters + Size', data=SeventyFivepercentdataWithoutLogs).fit()
print("\nCoefficients for 75% android apps without logs\n")
SeventyFivepercentdataWithoutLogsAnswer.params

type(SeventyFivepercentdataWithoutLogsAnswer)

AllCleanedAppsAnswer = smf.ols(formula='Downloads ~ AvgR', data=AllCleanedApps).fit()
print("\nCoefficients for all Cleaned android apps after taking log values\n")
AllCleanedAppsAnswer.params

SeventyFivepercentdataAnswer = smf.ols(formula='Downloads ~ AvgR', data=SeventyFivepercentdata).fit()
print("\nCoefficients for Cleaned 75 % android apps after taking log values\n")
#SeventyFivepercentdataAnswer.params

#print(SeventyFivepercentdataWithoutLogsAnswer.summary())

AllCleanedApps= AllCleanedApps.ix[AllCleanedApps.Downloads > 5, :]
y=AllCleanedApps['Downloads']
X=AllCleanedApps.drop(['Downloads', 'Name','Unnamed: 0', 'Random'], axis=1)
y.shape
X.shape

from sklearn.linear_model import LinearRegression
lr= LinearRegression()

from sklearn.model_selection import train_test_split
X_tr, X_test, y_tr, y_test = train_test_split(X,y,test_size=0.3)

X_tr.shape,y_tr.shape,X_test.shape, y_test.shape
X_tr.head(3)
y_tr.head(3)
#y_test.sort_values(ascending=False)

lr.fit(X_tr,y_tr)
y_pred= lr.predict(X_test)

import sklearn
from sklearn.metrics import mean_squared_error, r2_score
print('mse:',mean_squared_error(y_test,y_pred))
print('r2_score:',r2_score(y_test, y_pred))
y_test.shape
y_pred.shape
y_test.head(5)

from sklearn.model_selection import cross_val_score
print(cross_val_score(LinearRegression(), X, y).mean())

import matplotlib.pylab as plt
get_ipython().magic('matplotlib inline')
plt.figure(figsize=(15, 10))
min(y_test)
max(y_test)
min(y_pred)
max(y_pred)
#y_test = np.linspace(0, y_test.max, 50)
plt.scatter(y_test, y_pred)
plt.title('Predicted vs. Actual')
plt.xlabel('Actual Downloads')
plt.ylabel('Predicted Downloads')

plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)])
#plt.tight_layout()

