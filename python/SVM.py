get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

import pandas as pd
import numpy as np
import dmc
from process import processed_data

data = processed_data()
train_ids, test_ids = dmc.loading.load_ids('rawMirrored')
train, test = dmc.preprocessing.split_train_test(data, train_ids, test_ids)

# allocate memory
data = train_ids = test_ids = None
####
ensemble = dmc.ensemble.Ensemble(train, test)

for split in ensemble.splits:
    print(len(ensemble.splits[split][1]), split)

def shuffle(df: pd.DataFrame) -> pd.DataFrame:
    return df.reindex(np.random.permutation(df.index))

for split in ensemble.splits:
    ensemble.splits[split] = (shuffle(ensemble.splits[split][0])[:50000], ensemble.splits[split][1])

splits = len(ensemble.splits)
scalers = [dmc.transformation.scale_raw_features] * splits
ignore = [None] * splits
ensemble.transform(scalers=scalers, ignore_features=ignore)

res = []
for k in ensemble.splits:
    Xt, yt = ensemble.splits[k]['train'][0], ensemble.splits[k]['train'][1]
    Xe, ye = ensemble.splits[k]['test'][0], ensemble.splits[k]['test'][1]
    fts, clf = ensemble.splits[k]['features'], dmc.classifiers.SVM
    clf = clf(Xt, yt)
    prec = dmc.evaluation.precision(ye, clf(Xe))
    res.append((k, prec, len(ye)))
    print(k, prec, len(ye))



