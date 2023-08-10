get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
pd.set_option("display.max_columns",1500)
pd.set_option("display.max_rows",1500)

_new = '/home/bay001/projects/codebase/rbp-maps/examples/se/new.20.RBFOX2-BGHLV26-HepG2-excluded-upon-knockdown.normed_matrix.txt'
new_df = pd.read_table(_new, sep=',', index_col=0)
new_df

# chr1:162560251-162567631
plt.plot(
    new_df.ix[
        '15694\tENSG00000117143.9\tUAP1\tchr1\t+\t162562521\t162562572\t162560112\t162560301\t162567581\t162567648\t15694\t3,9\t61,50\t33,16\t18,16\t150\t100\t2.10287898206e-10\t1.53383992951e-07\t0.032,0.107\t0.55,0.4\t-0.406'
    ]
)

# chr1:53736968-53741352:-
# 2nd and 3rd panel only: chr1:53737775-53738814
region = '32890\tENSG00000157193.10\tLRP8\tchr1\t-\t53738275\t53738314\t53736898\t53737018\t53741302\t53741335\t32890\t0,0\t12,16\t11,5\t10,4\t138\t100\t3.48778205322e-05\t0.00867268714643\t0.0,0.0\t0.444,0.475\t-0.46'
plt.plot(
    new_df.ix[
        region
    ]
)


# old_peak_file = '/home/bay001/projects/codebase/rbp-maps/examples/se/204_01.basedon_204_01.peaks.l2inputnormnew.bed.compressed.RBFOX2-BGHLV26-HepG2-excluded-upon-knockdown.hist'
# new_peak_file = '/home/bay001/projects/codebase/rbp-maps/examples/se/new.RBFOX2-BGHLV26-HepG2-excluded-upon-knockdown.hist.txt'
old_peak_file = '/home/bay001/projects/codebase/rbp-maps/examples/se/original.50.RBFOX2-BGHLV26-HepG2-excluded-upon-knockdown.miso.50.hist'
new_peak_file = '/home/bay001/projects/codebase/rbp-maps/examples/se/new.50.RBFOX2-BGHLV26-HepG2-excluded-upon-knockdown.hist.txt'

old = pd.read_table(
    old_peak_file,
    names=['old']
)
new = pd.read_table(
    new_peak_file,
    names=['new']
)
new.head()

old

new

_new = '/home/bay001/projects/codebase/rbp-maps/examples/se/new.50.RBFOX2-BGHLV26-HepG2-excluded-upon-knockdown.normed_matrix.txt'
new_df = pd.read_table(_new, sep=',', index_col=0)
new_df.head()

# chr11:70266566-70269095
new_df[new_df['247']==1]

series = pd.Series([1,2,3,4,5])
series = pd.Series(series.iloc[::-1], index=reversed(series.index))
series
# series = series.reindex(pd.Index(reversed(series.index)))

pd.Series([s for s in reversed(series)])



