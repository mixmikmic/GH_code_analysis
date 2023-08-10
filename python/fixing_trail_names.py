import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

pkl_file = open('../data/df.pkl', 'rb')
df = pickle.load(pkl_file)
pkl_file.close() 

a = list(df['trail_name'][df['resort'] == 'Monarch'])
b = [x.split() for x in a]
c = [''.join(x) if len(x[0]) == 1 else ' '.join(x) for x in b]
c
for idx,name in enumerate(c):
    for i in range(len(name)-1):
        if name[i].islower() and name[i+1].isupper():
            print(idx,name)
            print
        if name[i].isupper() and name[i+1].isupper():
            print(idx,name)

c[19] = 'Quick Draw'
c[20] = 'KC Cutoff'
c[41] = "Doc's Run"
c[42] = 'Dire Straits'
c[47] = 'Great Divide'
c[53] = "Geno's Meadow"

c

df['trail_name'][df['resort'] == 'Monarch'] = c

df[df['resort'] == 'Monarch'].shape



