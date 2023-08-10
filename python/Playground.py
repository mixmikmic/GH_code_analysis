import pandas as pd
import time # time everything
start = time.time()
first_6th = pd.read_csv("/Users/chineseSamurai/Documents/capstone_Data/all",encoding='iso-8859-1')
end = time.time()
print("op done in %0.2f seconds" % (end - start))

# initial look at data: dimension, first couple of obs
print('dimension of dataset is: '+str(first_6th.shape))
first_6th.head(10)

# summary statistics:
print(first_6th.Source.unique().size)
print(first_6th.Destination.unique().size)
print(first_6th.Protocol.unique().size)

import numpy as np
#np.mean(first_6th.Length)
np.std(first_6th.Length)
print(max(first_6th.Length))
print(min(first_6th.Length))

# filter data, keep only transmission leaving UVA network
### in order to do this, define a function
def filter_data(df, domain):
    """filter df based on first six digits of domain (e.g. '111.111.xxx.xxx').
    return filtered column as a pandas.series obj
    """
    # defining capture group for use in pattern matching
    cap_grp = ("%s%s%s") % ("(",domain,".\d*.\d*)")
    return df['Source'].str.extract(cap_grp, expand=False)

# ‘128.143’, ‘137.54’, or ‘199.111’
test = filter_data(first_6th, '199.111')

test.fillna(filter_data(first_6th, '137.54'), inplace = True)
test.fillna(filter_data(first_6th, '128.143'), inplace = True)

# take 199.111.xxx.xxx as a filter
start = time.time()
first_6th.loc[:,'Source'] = test
end = time.time()
print("op done in %0.2f seconds" % (end - start))
first_6th.head(10)

# now simply take out all obs with Source IP missing.
first_6th.dropna(axis=0, inplace = True)
# and check out dimension after dropping:
print(first_6th.shape)
first_6th.head(10)

print(first_6th.Source.unique().size)

# form unique src-dest IP pairs
start = time.time()
first_6th['Pair'] = first_6th['Source'] +"_"+ first_6th['Destination']
end = time.time()
print("op done in %0.2f seconds" % (end - start))
first_6th.head(10)

# frequency counts, duration, length, etc.


(36-21)/36



