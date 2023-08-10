get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

from sys import path

generator_dir = '../generators/'
utils_dir = '../data_manager/'
path.append(generator_dir)
path.append(utils_dir)

import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
mush_df = pd.read_csv('../data/mushrooms/mushrooms.data', header=None, sep=" ")
#adult_df = pd.DataFrame(scale(adult_df), columns=['Feature {}'.format(i) for i in range(4)])
print(mush_df.shape)

mush_df.head()

#adult_df.iloc[:,:5].hist()

from csam import SAM
sam = SAM(nh=10, dnh=50, lr=0.01, dlr=.01, batchsize=-1, train_epochs=200, test_epochs=200)
sam.predict(mush_df,categorical_variables=[True for i in range(len(mush_df.columns))], nruns=1, gpus=1)

gdata = sam.generate(mush_df, [True for i in range(len(mush_df.columns))], gpu=True) 

gdata =  [i.data.cpu().numpy() for i in gdata]

gdata[16].fill(1)  # One column 
d1 = [[np.random.choice(len(j[0]),p=np.abs(j[i])) for i in range(j.shape[0])] for j in gdata]

d2 = np.array(d1).transpose()
d2.shape
pd.DataFrame(d2).to_csv("csam_mushrooms.csv",index=False)

onehotdata = []
for i in range(len(mush_df.columns)):
    onehotdata.append(pd.get_dummies(mush_df.iloc[:, i]).as_matrix())

orig_data = [np.array([list(i).index(1) for i in j]) for j in onehotdata]
orig_data = np.stack(orig_data, 1)
orig_data.shape

from metric import *

mush_orig = orig_data
mush_artif = d2

print('Covariance discrepancy: ', cov_discrepancy(mush_orig, mush_artif))
print('Correlation discrepancy: ', corr_discrepancy(mush_orig, mush_artif))
print('Relief divergence: ', relief_divergence(mush_orig, mush_artif))
print('KS_test: ', ks_test(mush_orig, mush_artif))
print('NN discrepancy', nn_discrepancy(mush_orig, mush_artif))
print('BAC metric', bac_metric(mush_orig, mush_artif))

pd.DataFrame(orig_data[:,0:6]).hist()

pd.DataFrame(d2[:,0:6]).hist()



