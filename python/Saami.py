get_ipython().magic('precision 3')

import numpy as np
import pandas as pd
pd.set_option('display.float_format',lambda x : '%.3f'%x)

import entropy

saami = pd.read_table('saami.txt', index_col=0)
saami

sing = [c for c in saami.columns if c.endswith('sg')]
plur = [c for c in saami.columns if c.endswith('pl')]
print saami[sing].to_latex()
print saami[plur].to_latex()

saami.describe()

len(set(saami.values.flatten()))

np.log2(len(saami.index))

H = pd.DataFrame([entropy.entropy(saami)], index=['H'])
H

print H[sing].to_latex()
print H[plur].to_latex()

print entropy.entropy(saami).mean()
print 2**entropy.entropy(saami).mean()

H = entropy.cond_entropy(saami)
H

pd.DataFrame([H.mean(0)], index=['AVG'])

print pd.DataFrame([H.mean(0)],index=['AVG'])[sing].to_latex()
print pd.DataFrame([H.mean(0)],index=['AVG'])[plur].to_latex()

pd.DataFrame([H.mean(1)], index=['AVG'])

print pd.DataFrame([H.mean(1)],index=['AVG'])[sing].to_latex()
print pd.DataFrame([H.mean(1)],index=['AVG'])[plur].to_latex()

H = H.join(pd.Series(H.mean(1), name='AVG'))
H = H.append(pd.Series(H.mean(0), name='AVG'))
H

print H[sing].to_latex(na_rep='---')
print H[plur].to_latex(na_rep='---')

boot = entropy.bootstrap(saami, 999)

boot.mean()

len(boot),boot.min()

sum(boot <= boot[0]) / 1000.

get_ipython().magic('matplotlib inline')

import matplotlib.pyplot as plt

plt.style.use('seaborn-white')
plt.rcParams['figure.figsize']= '8, 6'

plot = boot.hist(bins=40)



