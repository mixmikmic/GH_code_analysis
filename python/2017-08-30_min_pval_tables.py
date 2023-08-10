import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

bigtable = pd.read_csv('../../output/concat.txt', sep='\t', index_col=0)

bigtable['motif_hits'] = 0
grp=bigtable.groupby(['motif_FBgn','target_gene','source','#hits'])

agg=grp.agg({'min_pval': ['min'], 'motif_hits': 'count'})

agg.columns = ['min_pval', 'motif_hits'] 

agg2 = agg.reset_index()

agg2.columns = ['motif_FBgn','target_gene','source','pos_hits','min_pval', 'motif_hits']

agg2.head()

#agg2['sum_pos_hits'] = 0
#agg2['sum_motif_hits'] = 0 
grp2 = agg2.groupby(['motif_FBgn','target_gene'])

agg3 = grp2.agg({'min_pval':['min'], 'pos_hits':['sum'],'motif_hits':['sum']})

agg3.columns = ['sum_pos_hits','min_pval', 'sum_motif_hits'] 

agg4 = agg3.reset_index()

#add column for motif_symbol
symbolmap = pd.read_table('/data/LCDB/lcdb-references/dmel/r6-11/gtf/dmel_r6-11.SYMBOL.csv', sep=',', na_values='NA', keep_default_na=False) 
newmap = symbolmap.drop_duplicates(subset='ENSEMBL', keep='first')
update = agg4.merge(newmap, left_on='motif_FBgn', right_on='ENSEMBL', how='left')
update = update.rename(columns={'SYMBOL': 'motif_symbol'})
trim = update[['motif_FBgn','motif_symbol','target_gene','min_pval','sum_motif_hits','sum_pos_hits']].copy()
trim.head()

#add column for gene_symbol
update2 = trim.merge(newmap, left_on='target_gene', right_on='ENSEMBL', how='left')
update2 = update2.rename(columns={'SYMBOL': 'gene_symbol'})
trim2 = update2[['motif_FBgn','motif_symbol','target_gene','gene_symbol','min_pval','sum_motif_hits','sum_pos_hits']].copy()
trim2.drop_duplicates(inplace=True)

trim2.groupby(['target_gene']).agg({'sum_motif_hits':['count']}).describe()

len(trim2.motif_FBgn.unique())

trim2.head()

#list of genes we have TF for/did RNAi on
TF_list = pd.read_table('../../data/list_of_tfs.txt', header=None)

TF_list.columns=['TF']

tfmerge = TF_list.merge(trim2, left_on='TF', right_on='motif_FBgn', how='inner')

len(newtrim.motif_FBgn.unique())

tfmerge.head()

newtrim = tfmerge[['motif_FBgn','motif_symbol','target_gene','gene_symbol','min_pval','sum_motif_hits','sum_pos_hits']]

newtrim.to_csv('../../output/minpval_table')

matrix = newtrim[['motif_FBgn','target_gene','min_pval']]
matrix.set_index(['target_gene','motif_FBgn'], inplace=True)

final = matrix.unstack()

final.to_csv('../../output/matrix.txt', sep='\t')

trim2[(trim2['target_gene'] == 'FBgn0000276') & (trim2['motif_FBgn'] == 'FBgn0000014')]



