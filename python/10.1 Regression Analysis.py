import pandas as pd
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data', header=None, sep='\s+')
df.columns= ['CRIM', 'ZN', 'INDUS', 'CHAS', 
             'NOX', 'RM', 'AGE', 'DIS', 'RAD', 
             'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df.head()

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid', context='notebook')
cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']
sns.pairplot(df[cols], size=2.5)
plt.show()

from IPython.display import Image
Image("/Users/surthi/gitrepos/ml-notes/images/correlationmatrix.jpg")

import numpy as np
cm = np.corrcoef(df[cols].values.T)

sns.set(font_scale=1.5)
hm = sns.heatmap(cm, 
                 cbar=True,
                 annot=True, 
                 square=True,
                 fmt='.2f',
                 annot_kws={'size': 15},
                 yticklabels=cols,
                 xticklabels=cols)
plt.show()

X = df[['RM']].values
X = X.reshape(X.shape[0], -1)
y = df['MEDV'].values
y = y.reshape(y.shape[0], -1)

from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
X_std = sc_x.fit_transform(X).reshape(X.shape[0], -1)
y_std = sc_y.fit_transform(y).reshape(y.shape[0], -1)

lr = LinearRegression()
lr.fit(X_std, y_std)

print X_std.shape, y_std.shape
def lin_regplot(X, y, model):
    plt.scatter(X, y, c='blue')
    plt.plot(X, model.predict(X), color='red')    
    return None

lin_regplot(X_std, y_std, lr)
plt.xlabel('Average number of rooms [RM] (standardized)')
plt.ylabel('Price in $1000\'s [MEDV] (standardized)')
plt.show()


print('Slope: %.3f' % lr.coef_[0])
print('Intercept: %.3f' % lr.intercept_)

