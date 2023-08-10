import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('1mii_ML.csv', index_col=0) # set index as 'Frame' (i.e. the time in the simulation)
index = 'Frame'

# df = pd.read_csv('1mii_ML.csv', index_col=3) # set index as 'RMSD' (i.e. the time in the simulation)
# index = 'RMSD'

column_names = df.columns[0:len(df.columns) - 1]
print(column_names)

# goals: minimize RMSD and total energy

df.head()

df.describe()

df.hist(figsize=(15,15))
plt.show()

column_x = df.columns[0:len(df.columns) - 1]
column_x

corr = df[df.columns].corr()

plt.figure(figsize=(16,10))
sns.heatmap(corr, annot = True)
plt.show()

# return the total energy for each frame
all_energy_terms = df.iloc[:,5:13]
total_energy = all_energy_terms.sum(axis=1)

# append total_energy to existing dataframe
df['total_energy'] = total_energy

# create new dataframe of just total energy and rmsd
total_energy_v_rmsd = df.iloc[:,-1]
total_energy_v_rmsd = total_energy_v_rmsd.to_frame()
total_energy_v_rmsd['RMSD'] = df.iloc[:,2]

total_energy_v_rmsd.head()

# how do the other terms effect the total energy?
# which energy terms contribute most to the total energy?
# how are total energy and RMSD correlated?

y = total_energy_v_rmsd.index

plt.scatter(total_energy_v_rmsd['total_energy'], y, s=1) # s describes the size of the marker
plt.xlabel('total_energy')
plt.ylabel('Frame')
plt.show()

y = total_energy_v_rmsd.index

plt.scatter(total_energy_v_rmsd['RMSD'], y, s=1)
plt.xlabel('RMSD')
plt.ylabel('Frame')
plt.show()

# phase portrait
plt.scatter(total_energy_v_rmsd['total_energy'], total_energy_v_rmsd['RMSD'], s=1)
plt.xlabel('Total Energy')
plt.ylabel('RMSD')
plt.show()

# classification of near native structures
# regression of RMSD

df.head()

y = df['RMSD'] # assign targets
X = df.iloc[:,0:22] # assign features

del X['RMSD'] # remove target from features

X.head()

# df.shape

del X['total_energy']
print(X.shape)
X.head()

# X.shape
y.head()

from sklearn.svm import SVR
from sklearn.linear_model import Ridge

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.33,
                                                    random_state=123)

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
# from sklearn.svm import SVR

lr = LinearRegression()
lr.fit(X_train, y_train)
lr_score = lr.score(X_test, y_test)
print("%20s | Accuracy: %0.2f%%" % ('Linear Regression', lr_score*100))

knr = KNeighborsRegressor()
knr.fit(X_train, y_train)
knr_score = knr.score(X_test, y_test)
print("%20s | Accuracy: %0.2f%%" % ('k-Nearest Regressor', knr_score*100))

svr = SVR(kernel='rbf')
svr.fit(X_train, y_train)
svr_score = svr.score(X_test, y_test)
print("%20s | Accuracy: %0.2f%%" % ('SVR-rbf', svr_score*100))

svr = SVR(kernel='linear')
svr.fit(X_train, y_train)
svr_score = svr.score(X_test, y_test)
print("%20s | Accuracy: %0.2f%%" % ('SVR-linear', svr_score*100))

rr = Ridge() # Ridge Regression
rr.fit(X_train, y_train)
rr_score = rr.score(X_test, y_test)
print("%20s | Accuracy: %0.2f%%" % ('Ridge Regression', rr_score*100))

import matplotlib.pyplot as plt

count = 0
features = []
coefs = []

for feature in X_train:
    features.append(feature)
    coefs.append(lr.coef_[count])
    count +=1
    
import pylab as pl

a = range(len(coefs))
xTicks = features
b = coefs
pl.xticks(a, xTicks)
pl.xticks(range(len(coefs)), xTicks, rotation=90) #writes strings with 45 degree angle
pl.plot(a,b,'*')
pl.show()

import pylab as pl

a = range(len(coefs))
xTicks = features
b = coefs
pl.xticks(a, xTicks)
pl.xticks(range(len(coefs)), xTicks, rotation=90) #writes strings with 45 degree angle
pl.plot(a,b,'*')
pl.show()

lr.coef_

# try using Ridge Regression, and compare this to the others 

