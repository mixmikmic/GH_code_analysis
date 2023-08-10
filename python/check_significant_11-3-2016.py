import pandas as pd
import numpy as np

f2 = pd.read_table('/home/bay001/projects/kes_20160307/permanent_data/9-27-2016/TBBPA_vs_CONTROL_FEMALE/diffexp.csv',sep=',')
m2 = pd.read_table('/home/bay001/projects/kes_20160307/permanent_data/9-27-2016/TBBPA_vs_CONTROL_MALE/diffexp.csv',sep=',')
m1 = pd.read_table('/home/bay001/projects/kes_20160307/permanent_data/10-11-2016/SCCP_vs_CONTROL_MALE/diffexp.csv',sep=',')
f1 = pd.read_table('/home/bay001/projects/kes_20160307/permanent_data/10-11-2016/SCCP_vs_CONTROL_FEMALE/diffexp.csv',sep=',')

# number of 'significant' genes without cutoff for TBBPA vs control affected males
m2[m2['padj']<0.05].shape # tbbpa male

# number of 'significant' genes without cutoff for TBBPA vs control affected females
f2[f2['padj']<0.05].shape # tbbpa female

# number of 'significant' genes without cutoff for SCCP vs control affected males
m1[m1['padj']<0.05].shape # sccp male

# number of 'significant' genes without cutoff for SCCP vs control affected females
f1[f1['padj']<0.05].shape # sccp female

# just look at a few of these genes
m2[m2['padj']<0.05]



