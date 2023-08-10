import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from rdkit import Chem
get_ipython().run_line_magic('matplotlib', 'inline')

homedir = os.path.dirname(os.path.realpath('__file__'))
homedir = homedir+"/data/"
df = pd.read_csv(homedir+"tox_niehs_all.csv")

df.head()

size = 0.10
seed = 6
np.random.seed(seed)

msk = np.random.rand(len(df)) < 0.1
df_tv = df[~msk]
df_int = df[msk]

print(df.shape, df_tv.shape, df_int.shape)

df_tv.to_csv(homedir+'tox_niehs_all_trainval.csv', index=False)
df_int.to_csv(homedir+'tox_niehs_all_int.csv', index=False)

import matplotlib.pyplot as plt

task = 'verytoxic'

fig, axes = plt.subplots(nrows=1, ncols=3)

df[task].hist(normed=True, ax=axes[0])
df_tv[task].hist(normed=True, ax=axes[1])
df_int[task].hist(normed=True, ax=axes[2])

task = 'nontoxic'

fig, axes = plt.subplots(nrows=1, ncols=3)

df[task].hist(normed=True, ax=axes[0])
df_tv[task].hist(normed=True, ax=axes[1])
df_int[task].hist(normed=True, ax=axes[2])

task = 'epa'

fig, axes = plt.subplots(nrows=1, ncols=3)

df[task].hist(normed=True, ax=axes[0])
df_tv[task].hist(normed=True, ax=axes[1])
df_int[task].hist(normed=True, ax=axes[2])

task = 'ghs'

fig, axes = plt.subplots(nrows=1, ncols=3)

df[task].hist(normed=True, ax=axes[0])
df_tv[task].hist(normed=True, ax=axes[1])
df_int[task].hist(normed=True, ax=axes[2])

task = 'ld50'

fig, axes = plt.subplots(nrows=1, ncols=3)

df[task].hist(normed=True, ax=axes[0])
df_tv[task].hist(normed=True, ax=axes[1])
df_int[task].hist(normed=True, ax=axes[2])

task = 'logld50'

fig, axes = plt.subplots(nrows=1, ncols=3)

df[task].hist(normed=True, ax=axes[0])
df_tv[task].hist(normed=True, ax=axes[1])
df_int[task].hist(normed=True, ax=axes[2])



