import pandas as pd

df1 = pd.DataFrame({'dog':[0,1,1], 'cat':[0,0,0], 'rat':[3,4,5]})
df1.index = df1.columns
df1

df2 = pd.DataFrame({'dog':[0, 0, 0], 'cat':[0, 1, 1], 'rat':[3,4,5]})
df2.index = df2.columns
df2

df3 = pd.DataFrame({'dog':[1, 1, 1], 'cat':[0, -1, -1], 'rat':[0,0,0]})
df3.index = df3.columns
df3

{n: df for n, df in enumerate([df1, df2, df3])}

p = pd.Panel(data={n: df for n, df in enumerate([df1, df2, df3])})

p

p.mean(axis=0)

p.std(axis=0)

df4 = df1.copy()

df4 = pd.DataFrame({'dog':[0,1,1,9], 'cat':[0,0,0,9], 'rat':[3,4,5,9], 'zebra':[9,9,9,9]})
df4.index = df4.columns
df4

p2 = pd.Panel(data={n: df for n, df in enumerate([df4, df2, df3])})

p2.mean(axis=0)

p2.std(axis=0)

import network_construction as net

net.aggregate_adjacency_matrices([df1, df2, df3, df4])

mock_data = {('High', 1): df1, ('High', 2): df2, ('High', 3):df3, ('High', 4):df4,
            ('Low', 1): df1, ('Low', 2): df2, ('Low', 3):df3, ('Low', 4):df4}

net.summarize_replicate_adjacency_matrices(mock_data)



