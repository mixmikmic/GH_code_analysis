import pandas as pd

ens2gene = pd.read_table('/home/bay001/projects/kes_20160307/org/00_data/references/biomart/chicken_biomart.txt',
                        index_col=0).fillna('-')
ens2gene.head(2)

# uses an old directory convention. These should now be in: /home/bay001/projects/kes_20160307/org/03_output/differential_expression/current
date = "10-11-2016"
sccp = '/home/bay001/projects/kes_20160307/permanent_data/{}/SCCP_vs_CONTROL/diffexp.csv'.format(date)
sccp_m = '/home/bay001/projects/kes_20160307/permanent_data/{}/SCCP_vs_CONTROL_MALE/diffexp.csv'.format(date)
sccp_f = '/home/bay001/projects/kes_20160307/permanent_data/{}/SCCP_vs_CONTROL_FEMALE/diffexp.csv'.format(date)

tbbpa = '/home/bay001/projects/kes_20160307/permanent_data/{}/TBBPA_vs_CONTROL/diffexp.csv'.format(date)
tbbpa_m = '/home/bay001/projects/kes_20160307/permanent_data/{}/TBBPA_vs_CONTROL_MALE/diffexp.csv'.format(date)
tbbpa_f = '/home/bay001/projects/kes_20160307/permanent_data/{}/TBBPA_vs_CONTROL_FEMALE/diffexp.csv'.format(date)

tomap = [sccp,sccp_m,sccp_f,tbbpa,tbbpa_m,tbbpa_f]
for m in tomap:
    df = pd.read_table(m,sep=',',index_col=0).fillna('-')
    X = pd.merge(ens2gene,df,how='right',left_index=True,right_index=True).sort_values(['pvalue','log2FoldChange'])
    X.to_csv(m.replace('.csv','.annotated.csv'))





