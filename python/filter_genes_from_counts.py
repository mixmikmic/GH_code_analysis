import pandas as pd

# filter OUT any cell that contains at least 1 count of a gene.

# make some fake small counts matrix
df = {'cell1':[1,0,1], 'cell2':[1,2,0], 'cell3':[1,2,0],'cell4':[0,0,1]}
df = pd.DataFrame(df, index=['ensg1','ensg2','ensg3'])
df

# some list of genes to filter out
genes_to_filter_out = ['ensg2','ensg1']
genes_to_filter_out

cols = df.loc[genes_to_filter_out].sum() == 0
df[cols[cols].index]

df.T[df.loc[genes_to_filter_out].sum() == 0].T



