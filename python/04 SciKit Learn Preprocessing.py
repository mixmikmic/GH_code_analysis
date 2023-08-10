import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# creating some random data
data = np.random.randint(0,100,(10,2))
data

scaller_model = MinMaxScaler()
type(scaller_model)

scaller_model.fit(data)

scaller_model.transform(data)

# Perform both fit and transfrom
scaller_model.fit_transform(data)

# crating datafram
data = np.random.randint(0,100,(50,4))
df = pd.DataFrame(data=data, columns=['f0', 'f1', 'f2', 'label'])
df.head()

# Separating Data and label
X = df[['f0','f1','f2']]
y = df['label']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 2)

# Checking the train and test shape
X_train.shape, X_test.shape



