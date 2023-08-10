import pandas as pd
import numpy as np
import os

sccp_vs_wt_male = pd.read_table('/home/bay001/projects/kes_20160307/permanent_data/9-27-2016/SCCP_vs_CONTROL_MALE/diffexp.csv',
                               sep=',', index_col=0)
sccp_vs_wt_female = pd.read_table('/home/bay001/projects/kes_20160307/permanent_data/9-27-2016/SCCP_vs_CONTROL_FEMALE/diffexp.csv',
                                  sep=',', index_col=0)
sccp_vs_wt = pd.read_table('/home/bay001/projects/kes_20160307/permanent_data/9-27-2016/SCCP_vs_CONTROL/diffexp.csv',
                           sep=',', index_col=0)

tbbpa_vs_wt_male = pd.read_table('/home/bay001/projects/kes_20160307/permanent_data/9-27-2016/TBBPA_vs_CONTROL_MALE/diffexp.csv',
                                   sep=',', index_col=0)
tbbpa_vs_wt_female = pd.read_table('/home/bay001/projects/kes_20160307/permanent_data/9-27-2016/TBBPA_vs_CONTROL_FEMALE/diffexp.csv',
                                     sep=',', index_col=0)
tbbpa_vs_wt = pd.read_table('/home/bay001/projects/kes_20160307/permanent_data/9-27-2016/TBBPA_vs_CONTROL/diffexp.csv',
                              sep=',', index_col=0)
padj = 0.05
l2fc = 1.5

sccp_vs_wt_male_sig = sccp_vs_wt_male[sccp_vs_wt_male['padj']<=padj]
sccp_vs_wt_male_sig = sccp_vs_wt_male_sig[abs(sccp_vs_wt_male_sig['log2FoldChange'])>=l2fc]
sccp_vs_wt_male_sig.to_csv('/home/bay001/projects/kes_20160307/permanent_data/9-27-2016/SCCP_vs_CONTROL_MALE/significant-genes.csv')
sccp_vs_wt_male_sig

sccp_vs_wt_female_sig = sccp_vs_wt_female[sccp_vs_wt_female['padj']<=padj]
sccp_vs_wt_female_sig = sccp_vs_wt_female_sig[abs(sccp_vs_wt_female_sig['log2FoldChange'])>=l2fc]
sccp_vs_wt_female_sig.to_csv('/home/bay001/projects/kes_20160307/permanent_data/9-27-2016/SCCP_vs_CONTROL_FEMALE/significant-genes.csv')
sccp_vs_wt_female_sig

sccp_vs_wt_sig = sccp_vs_wt[sccp_vs_wt['padj']<=padj]
sccp_vs_wt_sig = sccp_vs_wt_sig[abs(sccp_vs_wt_sig['log2FoldChange'])>=l2fc]
sccp_vs_wt_sig.to_csv('/home/bay001/projects/kes_20160307/permanent_data/9-27-2016/SCCP_vs_CONTROL/significant-genes.csv')
sccp_vs_wt_sig

tbbpa_vs_wt_male_sig = tbbpa_vs_wt_male[tbbpa_vs_wt_male['padj']<=padj]
tbbpa_vs_wt_male_sig = tbbpa_vs_wt_male_sig[abs(tbbpa_vs_wt_male_sig['log2FoldChange'])>=l2fc]
tbbpa_vs_wt_male_sig.to_csv('/home/bay001/projects/kes_20160307/permanent_data/9-27-2016/TBBPA_vs_CONTROL_MALE/significant-genes.csv')
tbbpa_vs_wt_male_sig

tbbpa_vs_wt_female_sig = tbbpa_vs_wt_female[tbbpa_vs_wt_female['padj']<=padj]
tbbpa_vs_wt_female_sig = tbbpa_vs_wt_female_sig[abs(tbbpa_vs_wt_female_sig['log2FoldChange'])>=l2fc]
tbbpa_vs_wt_female_sig.to_csv('/home/bay001/projects/kes_20160307/permanent_data/9-27-2016/TBBPA_vs_CONTROL_FEMALE/significant-genes.csv')
tbbpa_vs_wt_female_sig

tbbpa_vs_wt_sig = tbbpa_vs_wt[tbbpa_vs_wt['padj']<=padj]
tbbpa_vs_wt_sig = tbbpa_vs_wt_sig[abs(tbbpa_vs_wt_sig['log2FoldChange'])>=l2fc]
tbbpa_vs_wt_sig.to_csv('/home/bay001/projects/kes_20160307/permanent_data/9-27-2016/TBBPA_vs_CONTROL/significant-genes.csv')
tbbpa_vs_wt_sig



