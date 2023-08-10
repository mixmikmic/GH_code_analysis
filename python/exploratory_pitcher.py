import numpy as np
import pandas as pd

pitchers = pd.read_table("pitchers_cleaned.tsv")

pitchers.shape

pitchers.describe()

list(pitchers)

pitchers = pitchers.replace('None', np.nan)

pitchers.isnull()

np.sum(pitchers.isnull())

pd.set_option('display.max_columns', None)
pitchers.loc[pitchers['ibb'].isnull(),:]

pitchers['ibb'] = pitchers['ibb'].fillna(0)

pitchers.loc[pitchers['k'].isnull(),:]

pitchers['cg'] = pitchers['cg'].fillna(0)
pitchers['sho'] = pitchers['sho'].fillna(0)
pitchers['saves'] = pitchers['saves'].fillna(0)

pitchers.loc[pitchers['r'].isnull(),:]

pitchers = pitchers.dropna(subset=['r']) 

pitchers.loc[pitchers['name'] == "John Smoltz",:]

pitchers.loc[pitchers['fip'].isnull(),:]

# remove position
pitchers = pitchers.drop('position', 1)

pitchers.to_csv('pitchers_final.csv', header = True)



