import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np

wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',
                   header = None)
wine.columns = ['Class Label','Alcohol','Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols','Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity','Hue','OD280/OD315 of diluted wines', 'Proline']
wine.head()

from sklearn.model_selection import train_test_split

X, y = wine.iloc[:,1:], wine.iloc[:,0]

#the train test split command
X_train, X_test, y_train, y_test = train_test_split(X, y)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std =  sc.transform(X_test)

xx = np.arange(len(X_train))
plt.scatter(xx, X_train_std[:,2], color = 'b')
plt.scatter(xx, X_train.iloc[:,2], color = 'red')

