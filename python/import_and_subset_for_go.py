get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import seaborn as sns

wd = '/home/bay001/projects/kes_20160307/permanent_data/10-11-2016/'

# lets do sccp_vs_control first

sccp_vs_control = pd.read_table(os.path.join(wd,'SCCP_vs_CONTROL/diffexp.annotated.csv'),sep=',')
sccp_vs_control_female = pd.read_table(os.path.join(wd,'SCCP_vs_CONTROL_FEMALE/diffexp.annotated.csv'),sep=',')
sccp_vs_control_male = pd.read_table(os.path.join(wd,'SCCP_vs_CONTROL_MALE/diffexp.annotated.csv'),sep=',')

tbbpa_vs_control = pd.read_table(os.path.join(wd,'TBBPA_vs_CONTROL/diffexp.annotated.csv'),sep=',')
tbbpa_vs_control_female = pd.read_table(os.path.join(wd,'TBBPA_vs_CONTROL_FEMALE/diffexp.annotated.csv'),sep=',')
tbbpa_vs_control_male = pd.read_table(os.path.join(wd,'TBBPA_vs_CONTROL_MALE/diffexp.annotated.csv'),sep=',')

conditions = {'sccp_vs_control':sccp_vs_control, 'sccp_vs_control_female':sccp_vs_control_female, 'sccp_vs_control_male':sccp_vs_control_male, 
              'tbbpa_vs_control':tbbpa_vs_control, 'tbbpa_vs_control_female':tbbpa_vs_control_female, 'tbbpa_vs_control_male':tbbpa_vs_control_male}
             

padj_cutoff = 0.05
fold_change_cutoff = 0
genelist = {}
for condition, df in conditions.iteritems():
    filtered = df[(df['padj'].replace('-',1).astype(float)< padj_cutoff) & 
                  (abs(df['log2FoldChange'].replace('-',0).astype(float)) > fold_change_cutoff)]
    filtered_genes = set(filtered[filtered['Unnamed: 0'].str.contains("ENSGALG")]['Unnamed: 0'])
    filtered_all = set(filtered['Unnamed: 0'])
    print("cond: {}, number of named genes: {}, number of all genes: {}".format(condition, len(filtered_genes), len(filtered_all)))
    genelist[condition] = filtered_genes
    pd.DataFrame(pd.Series(list(filtered_genes))).to_csv(os.path.join(wd,"GO_ANALYSIS/{}.go.nocutoff.txt".format(condition)),
                                                        sep='\t',index=None,header=None) # ¯\_(ツ)_/¯

analyzed = get_ipython().getoutput('ls $wd/GO_ANALYSIS/*male*.analyzed.txt')
dfx = pd.DataFrame()
for analyze in analyzed:
    df = pd.read_table(analyze,skiprows=11,index_col=0)
    dfy = pd.DataFrame(df['upload_1 (fold Enrichment)'])
    dfy.columns = [os.path.basename(analyze).replace('.go.nocutoff.analyzed.txt','')]
    dfx = pd.merge(dfx,dfy,how='outer',left_index=True,right_index=True)
dfx.fillna(0)

colors = (sns.color_palette("BuPu",4))
sns.heatmap(dfx)
plt.title("GO Enriched Terms\n(Fold change over expected background)")

dfx



