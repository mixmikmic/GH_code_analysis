import pandas as pd

cosim = pd.read_csv('data/similarity_index.csv', index_col='UNITID')
cosim.head()

rankings = pd.DataFrame(index=cosim.index, columns=['Similar School {}'.format(i) for i in range(1,16)])
rankings.head()

for id in rankings.index:
    rankings.loc[id] = cosim[str(id)].sort_values(ascending=False)[1:16].index

rankings.head()

cosim[['INSTNM', '166027']].sort_values('166027', ascending=False).head(16)

rankings.loc[166027]

rankings.to_csv('data/similarity_rankings.csv')

