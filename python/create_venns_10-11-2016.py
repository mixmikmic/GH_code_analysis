get_ipython().magic('matplotlib inline')
from matplotlib_venn import venn2, venn3
import pandas as pd
import numpy as np
import os

wd = '/home/bay001/projects/kes_20160307/permanent_data/9-29-2016'

# SCCP VS CTRL
fc = 1

female = pd.read_table(os.path.join(wd,'SCCP_vs_CONTROL_FEMALE/diffexp.csv'),sep=',',index_col=0)
female = female[(female['padj']<=0.05) &
                (female['log2FoldChange'] >=fc)]
female_genes = set(female.index)

male = pd.read_table(os.path.join(wd,'SCCP_vs_CONTROL_MALE/diffexp.csv'),sep=',',index_col=0)
male = male[(male['padj']<=0.05) &
                (male['log2FoldChange'] >=fc)]
male_genes = set(male.index)

a = pd.read_table(os.path.join(wd,'SCCP_vs_CONTROL/diffexp.csv'),sep=',',index_col=0)
a = a[(a['padj']<=0.05) &
                (a['log2FoldChange'] >=fc)]
all_genes = set(a.index)
venn3([male_genes, female_genes, all_genes],['Male','Female','All'])

female = pd.read_table(os.path.join(wd,'TBBPA_vs_CONTROL_FEMALE/diffexp.csv'),sep=',',index_col=0)
female = female[(female['padj']<=0.05) &
                (female['log2FoldChange'] >=fc)]
female_genes = set(female.index)

male = pd.read_table(os.path.join(wd,'TBBPA_vs_CONTROL_MALE/diffexp.csv'),sep=',',index_col=0)
male = male[(male['padj']<=0.05) &
                (male['log2FoldChange'] >=fc)]
male_genes = set(male.index)

a = pd.read_table(os.path.join(wd,'TBBPA_vs_CONTROL/diffexp.csv'),sep=',',index_col=0)
a = a[(a['padj']<=0.05) &
                (a['log2FoldChange'] >=fc)]
all_genes = set(a.index)
venn3([male_genes, female_genes, all_genes],['Male','Female','All'])

male - female



