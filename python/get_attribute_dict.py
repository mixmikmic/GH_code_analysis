import pandas as pd
import numpy as np

def process_lookup(path):
    df = pd.read_csv(path)
    df['Q'] = df[pd.notnull(df['Total Cells in Table'])]['Table Title']
    df['Q'] = df['Q'].fillna(method='ffill')
    df['Sequence Number'] = df['Sequence Number'].apply(lambda x: str(x).zfill(4))
    df = df[df['Line Number'].str.strip().str.len() >0]
    
    df['A'] = df[df['Table Title'].str.endswith(':')]['Table Title']
    df['A'] = df.groupby(['Table ID', 'Q'])['A'].transform(lambda group: group.ffill())

    df['Column'] = df['Sequence Number'] +'_' + df['Line Number'].astype(str)
    
    df['dataCol'] = np.NAN
    df['dataCol'] = df.groupby('Sequence Number')['dataCol'].transform(lambda group:  (group.reset_index().index +1).astype(str))
    df['dataCol'] = df['Sequence Number'] +'_' + df['dataCol'].astype(int).astype(str)

    return df[['Table ID', 'Column','dataCol', 'Q', 'A', 'Table Title']]

pa = process_lookup('../../Data/raw_aggData/lookup_table20131Y.csv')
ba = process_lookup('../../Data/raw_aggData/lookup_table20135Y.csv')

pa.head(10)

ba.head(10)

len([ x for x in pa['Table ID'].unique().tolist() if x in ba['Table ID'].unique().tolist()])

df = pa.merge(ba, how='inner', on=['Table ID','Q', 'A', 'Table Title'])
df.rename(columns={'dataCol_x':'puma_attr', 'dataCol_y':'block_attr'}, inplace=1)

df.shape

df.Q = df.Q.str.lower()
df['Table Title'] = df['Table Title'].str.lower()

# dropping all imputations
df = df[~df['Q'].str.contains('imputation')]
df.shape

# dropping all samples
df = df[~df['Q'].str.contains('sample')]
df.shape

df['Q_name'] = df.apply(lambda x:'%s_%s_%s_%s' % (x['Table ID'],x['Q'],x['A'],x['Table Title']),axis=1)

df[df['puma_attr']!=df['block_attr']].head(10)

r = df[['Q_name', 'puma_attr','block_attr']]
r.reset_index(inplace=1)
r['ourID'] = r['index'].apply(lambda x: 'our%d' % x)
r.drop('index',1, inplace=1)

# r['puma_attr'] = r.puma_attr.apply(lambda x: '_'.join((x.split('_')[0].zfill(4), x.split('_')[1])))
# r['block_attr'] = r.puma_attr.apply(lambda x: '_'.join((x.split('_')[0].zfill(4), x.split('_')[1])))

r.head(10)

r.shape[0] == r['ourID'].unique().shape[0]

r.to_csv('../../Data/raw_aggData/attr_interconn.csv')

df2 = ba.merge(pa, how='left', on=['Table ID','Q', 'A', 'Table Title'])
df2 = df2[pd.isnull(df2['dataCol_y'])]

df2['Q'].unique()

df3 = pa.merge(ba, how='left', on=['Table ID','Q', 'A', 'Table Title'])
df3 = df3[pd.isnull(df3['dataCol_y'])]
pd.DataFrame(df3['Q'].unique())

df3.to_csv('../../Data/raw_aggData/what_we_can_predict.csv')



