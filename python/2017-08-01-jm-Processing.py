from scipy.io.arff import loadarff
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd
import scipy as sp

data, meta = loadarff('../data/KDDCup99_full.arff')

meta

data = pd.DataFrame(np.asarray([list(i) for i in data], dtype=str))

cat_cols = [1,2,3,41]

for x in data.columns:
    if x in cat_cols:
        continue
    data[x] = pd.to_numeric(data[x])

data[41].unique()

for x in cat_cols:
    le = LabelEncoder()
    le.fit(data[x])
    data[x] = le.transform(data[x])

labels = data[41]
data.drop([41],inplace = True, axis = 1)

data.columns

labels.unique()

cat_cols = [1,2,3]

ohe = OneHotEncoder(sparse = False)
ohe.fit(data[cat_cols])
transformed = ohe.transform(data[cat_cols])

ohe = OneHotEncoder(sparse = False)
ohe.fit(labels.values.reshape(-1,1))
labels_k = ohe.transform(labels.values.reshape(-1,1))

transformed.shape

labels.shape

labels_k.shape

data.drop(cat_cols, axis = 1, inplace = True)

data = data.values

data = np.concatenate((data, transformed), axis = 1)

#main_msk = np.random.uniform(size = data.shape[0]) < 0.7
#np.save('../data/main_splitting_mask.npy', main_msk)
main_msk = np.load('../data/main_splitting_mask.npy')

main_msk.shape

X_train = data[main_msk, :]
y_train = labels[main_msk]
y_train_k = labels_k[main_msk]
data = data[~main_msk, :]
labels = labels[~main_msk]
labels_k = labels_k[~main_msk]

#sub_msk = np.random.uniform(size = data.shape[0]) < 0.5
#np.save('../data/sub_splitting_mask.npy', sub_msk)
sub_msk = np.load('../data/sub_splitting_mask.npy')

X_val = data[sub_msk, :]
y_val = labels[sub_msk]
y_val_k = labels_k[sub_msk]

np.save('../data/X_train.npy', X_train)
np.save('../data/y_train.npy', y_train)
np.save('../data/y_train_k.npy', y_train_k)
np.save('../data/X_val.npy', X_val)
np.save('../data/y_val.npy', y_val)
np.save('../data/y_val_k.npy', y_val_k)
np.save('../data/X_test.npy', data[~sub_msk,:])
np.save('../data/y_test.npy', labels[~sub_msk])
np.save('../data/y_test_k.npy', labels_k[~sub_msk])

