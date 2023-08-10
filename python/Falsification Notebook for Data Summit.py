import numpy as np
import pandas as pd
from fancyimpute import MICE, SoftImpute, KNN  #if speed desired, KNN or SoftImpute
from sklearn.preprocessing import StandardScaler
import hdbscan

# (Optional) To get a progress bar on processing
from tqdm import tqdm

# Import data
data = pd.read_stata(r"X:\Box\a_dataverse_JJ\IPA Studies\88 - Robinson - Savings Constraints, Kenya\Robinson - Savings Constraints\dataset_savingsAEJ.dta", convert_categoricals=False, convert_dates=False)

data.shape

# ID columns designated as 'other', those that were randomly generated, and those with personal identification / all uniques
cols_other = []
cols_rand = []
cols_uid = []

for c in list(data.columns):
    # consider making others categoricals
    if 'other' in c:
        cols_other.append(c)
    # these two search for 'id' variables, because they don't have meaning and should be excluded
    if 'rand' in c and 'brand' not in c:
        cols_rand.append(c)
    if len(set(data[c])) == len(data[c]):
        cols_uid.append(c)
        
droppable_cols = set(cols_other + cols_rand + cols_uid)

for dc in droppable_cols:
    del data[dc]

data.shape

# cleaning timestamped data in a form that works for training models
for c in list(data.columns):
    if type(data[c][0]) == pd.Timestamp:
        data[c] = data[c].astype(np.int64)

# converting string variable to numeric categoricals
data = pd.get_dummies(data)
data.shape

# Impute missing data: MICE is best choice for statistical accuracy; if speed desired, KNN or SoftImpute
## DO NOT use sklearn's imputation function (introduces bias)

if data.isnull().values.any(): #tests if there are any missing values
    data_imputed = SoftImpute().complete(data.values)
else:
    data_imputed = data
    print('There appears to be no missing values.')

# Normalize the data
data_normalized = pd.DataFrame(StandardScaler().fit_transform(data_imputed))

data_normalized

# Fit and run the model
clusterer = hdbscan.HDBSCAN(min_cluster_size=50) # , allow_single_cluster=True)
cluster_labels = clusterer.fit_predict(data_normalized)

len(np.unique(cluster_labels))

for x in set(list(cluster_labels)):
    print (str(x))
    print (list(cluster_labels).count(x))

list(cluster_labels).count(-1)

