get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from pandas.util.testing import assert_frame_equal

df1 = pd.read_table(
    '/oasis/tscc/scratch/bay001/parp13_20171015/shashank/KO_Arsenite_vs_293_WT_deseq2_results.csv',
    sep=',', index_col=0
)
df1.sort_index(inplace=True)
print(df1.shape)
df1.head()

df2 = pd.read_table(
    '/home/bay001/projects/parp13_ago2_20171015/permanent_data/chang_newdata/deseq2/20171107/ko_vs_wt/counts.Arsenite.multimap.diffexp.txt',
    sep=',', index_col=0
)
print(df2.shape)
df2.sort_index(inplace=True)
df2.head()

df3 = pd.read_table(
    '/home/bay001/projects/parp13_ago2_20171015/permanent_data/chang_newdata/deseq2/20171107/ko_vs_wt_useallcounts/counts.Arsenite.multimap.diffexp.txt',
    sep=',', index_col=0
)
print(df3.shape)
df3.sort_index(inplace=True)
df3.head()

# assert_frame_equal(df1,df2)

pd.merge(df1[['log2FoldChange']], df3[['log2FoldChange']], how='outer', left_index=True, right_index=True).corr()

pd.merge(df1[['padj']], df2[['padj']], how='inner', left_index=True, right_index=True).corr()

df1 = pd.read_table(
    '/oasis/tscc/scratch/bay001/parp13_20171015/shashank/KO_Arsenite_vs_293_WT_deseq2_normalized_counts.csv',
    sep=',', index_col=0
)
df1.sort_index(inplace=True)
print(df1.shape)
df1.head()

df2 = pd.read_table(
    '/home/bay001/projects/parp13_ago2_20171015/permanent_data/chang_newdata/deseq2/20171107/ko_vs_wt_useallcounts/counts.Arsenite.multimap.diffexp.txt.norm_counts',
    sep=',', index_col=0
)
print(df2.shape)
df2.sort_index(inplace=True)
df2.head()

dfx = pd.merge(
    df1[['X293_PARP13_KO_Arsenite_rep1.polyATrim.adapterTrim.rmRep.sorted.rg.bam']], 
    df2[['X293_PARP13_KO_Arsenite_rep1.polyATrim.adapterTrim.rmRep.sorted.rg.bam']],
    left_index=True, right_index=True
)
dfx.corr()





