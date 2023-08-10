get_ipython().magic('matplotlib notebook')
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
import hdbscan
import seaborn as sns
sns.set_style("dark")

pd.set_option('max_colwidth', 1000)

data = pd.read_csv('data/scorecard_reduced_features.csv')
imputed = pd.read_csv('data/scorecard_imputed.csv')
imputed['UNITID'] = data.UNITID

data.set_index('UNITID', inplace=True)
data.head()

imputed.set_index('UNITID', inplace=True)
imputed.head()

imputed = pd.get_dummies(imputed, columns=['CONTROL', 'LOCALE', 'CCBASIC', 'CCUGPROF', 'CCSIZSET'])

rescaled = MinMaxScaler().fit_transform(imputed)
imputed[imputed.columns] = rescaled

rescaled.shape

from sklearn.metrics.pairwise import cosine_similarity

c = cosine_similarity(rescaled, rescaled)

cosim = pd.DataFrame(c, index=data.index, columns=data.index)
cosim['INSTNM'] = data.INSTNM
cosim['ZIP'] = data.ZIP

cols = cosim.columns.tolist()
cosim = cosim[cols[-2:]+cols[:-2]]

cosim.head()

cosim[cosim.INSTNM.str.contains('Harvard')]['INSTNM']

cosim[['INSTNM', 166027]].sort_values(166027, ascending=False).head(15)

cosim[cosim.INSTNM.str.contains('Truman')]['INSTNM']

cosim[['INSTNM', 178615]].sort_values(178615, ascending=False).head(15)

cosim[cosim.INSTNM.str.contains('University of Utah')]['INSTNM']

cosim[['INSTNM', 230764]].sort_values(230764, ascending=False).head(15)

cosim[cosim.INSTNM.str.contains('University of Texas at Austin')]['INSTNM']

cosim[['INSTNM', 228778]].sort_values(228778, ascending=False).head(15)

cosim[cosim.INSTNM.str.contains('Missouri University of Science and Technology')]['INSTNM']

cosim[['INSTNM', 178411]].sort_values(178411, ascending=False).head(15)

cosim[cosim.INSTNM.str.contains('California Institute of Technology')]['INSTNM']

cosim[['INSTNM', 110404]].sort_values(110404, ascending=False).head(15)

cosim[cosim.INSTNM.str.contains('Stanford')]['INSTNM']

cosim[['INSTNM', 243744]].sort_values(243744, ascending=False).head(15)

cosim[cosim.INSTNM.str.contains('Brigham Young')]['INSTNM']

cosim[['INSTNM', 230038]].sort_values(230038, ascending=False).head(15)

cosim[cosim.INSTNM.str.contains('Calvin College')]

cosim[['INSTNM', 169080]].sort_values(169080, ascending=False).head(15)

cosim[cosim.INSTNM.str.contains('Spark')]

cosim[['INSTNM', 21997601]].sort_values(21997601, ascending=False).head(15)

cosim[cosim.INSTNM.str.contains('Lipscomb')]

cosim[['INSTNM', 219976]].sort_values(219976, ascending=False).head(15)

cosim[['INSTNM', 439279]].sort_values(439279, ascending=False).head(30)

cosim.to_csv('data/similarity_index.csv')

