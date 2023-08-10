import pandas as pd
import numpy as np

X = pd.DataFrame([[1,2,3],[4,5,6],['duck','duck','goose'],[0,0,0]])
X.columns = ['col1','col2','col3']
X

def last_to_first(df):
    cols = list(df)
    cols.insert(0, cols.pop(cols.index(cols[-1])))
    return df.ix[:, cols]

# last_to_first(X)

def first_to_last(df):
    cols = list(df)
    cols.append(cols.pop(cols.index(cols[0])))
    return df.ix[:, cols]

# first_to_last(X)

def reorder_cols(df, cols):
    return df.ix[:, cols]

# reorder_cols(X,['col2','col3','col1'])

def return_every_n(df, n):
    return df.iloc[:, ::n]

# Y = pd.concat([X,X,X,X,X],axis=1)
# return_every_n(Y,2)

def delete_list(df, todelete):
    return df.drop(df.ix[:,todelete].head(0).columns, axis=1)

# delete_list(Y, 'col3')

def delete_list_ix(df, todelete):
    colnames = df.columns
    for tod in todelete:
        colnames.remove(tod)
    df.columns = range(0,df.shape[1])
    dx = df.drop(df.ix[:,todelete],axis=1)
    dx.columns = colnames
    return dx
# delete_list_ix(Y, [2,3])

todelete = [1,2]
columns = range(0,Y.shape[1])
# [columns.remove(tod) for tod in todelete]

cds = '/projects/ps-yeolab/genomes/hg19/hg19.introns.bed.txt'
bedhead = ['chrom','start','end','name','score','strand']
def get_name(row):
    """
    cdsmeans['ENST00000237247.6_cds_2_0_chr1_67091530_f']
    """
    return row['name'].split('_')[0]
cdsdf = pd.read_table(cds,names=bedhead)
cdsdf['name2'] = cdsdf.apply(get_name,axis=1)
cdsdf.set_index('name2',inplace=True)

Y = pd.read_table('/home/bay001/projects/parp13_ago2_20160606/analysis/clips/clips_20160820/input_norm_manifest.txt')

Y.ix[0]['CLIP']

# using lambda and apply
bedhead = ['chrom','start','end','name','score','strand']
df = pd.read_table(
    '/home/bay001/projects/emilie_rnae_20161003/analysis/editing/Hundley22/results/GSF860-Hundley-22_S1_R1_001.polyATrim.adapterTrim.rmRep.sorted.rg.fwd.sorted.rmdup.readfiltered.formatted.varfiltered.snpfiltered.ranked.bed',
    names=bedhead
)
df = df.head()
df['editfrac'] = df['name'].apply(lambda x: x.split('|')[2])
df

# finding all rows with a nan
df = pd.DataFrame([range(3), [0, np.NaN, 0], [0, 0, np.NaN], range(3), range(3)])
df[df.isnull().any(axis=1)]

delim = ','
X = pd.DataFrame([['ENSG1','ENSTA,ENSTB,ENSTC','some_value1'],['ENSG2','ENSTD,ENSTE',2],['ENSG3','ENSTF','some_value3']])
X.columns = ['gene','transcript','some_other']
X

Y = pd.DataFrame(X.transcript.str.split(delim).tolist(),index=[X['gene'],X['some_other']]).stack()
Y = Y.reset_index()[[0, 'gene','some_other']]
Y.columns = ['gene','transcript','some_other']
Y

# generic function for doing the above:

def explode(df, delim, col_to_split, cols_to_keep):
    """
    explodes a dataframe by splitting a column on a delimiter, and 
    producing one row for each split. 
    """
    cols_to_keep_list = [y.name for y in cols_to_keep]
    dx = pd.DataFrame(df[col_to_split].str.split(delim).tolist(),index=cols_to_keep).stack()
    dx = dx.reset_index()[[0] + cols_to_keep_list]
    dx.columns = [col_to_split] + cols_to_keep_list
    return dx

explode(X, ',', 'transcript', [X['gene'],X['some_other']])





