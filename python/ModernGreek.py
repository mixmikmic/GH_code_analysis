get_ipython().magic('precision 3')

import pandas as pd
import numpy as np
pd.set_option('display.float_format',lambda x : '%.3f'%x)

import entropy

greek = pd.read_table('mgreek.txt',index_col=0)
greek

greek.describe()

np.log2(len(greek.index))

pd.DataFrame([entropy.entropy(greek)],index=['H'])

entropy.entropy(greek).mean()

H = entropy.cond_entropy(greek)
H

pd.DataFrame([H.mean(0)],index=['AVG'])

pd.DataFrame([H.mean(1)],index=['AVG'])

H = H.join(pd.Series(H.mean(1),name='AVG'))
H = H.append(pd.Series(H.mean(0),name='AVG'))
H

print H.to_latex(na_rep='---')

boot = entropy.bootstrap(greek, 999)

boot.mean()

boot[0]

sum(boot <= boot[0]) / 1000.

get_ipython().magic('matplotlib inline')

plot = boot.hist()



