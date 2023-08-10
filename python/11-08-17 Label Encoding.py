import pandas as pd
X = pd.DataFrame(data = [['Matt','24','99'],
 ['Owen','22', '98'],
 ['Link', '16','100']])
X.columns = ['name','age','score']
X

train = X[0:2]; train

test = X[1:3]; test

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
X['name'] = le.fit_transform(X['name'])

X

train

train['name'] = le.transform(train['name'])

X

train

test['name'] = le.transform(test['name'])

test

