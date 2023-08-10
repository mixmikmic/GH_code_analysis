import pandas as pd

df = pd.DataFrame([
    {
        'a': 1,
        'b': 2,
        'd': 4
    }
])

df

columns = ['a', 'b', 'c', 'd']

df.reindex(columns=columns, fill_value=0)

columns_subset = columns[:2]
columns_subset

df.reindex(columns=columns_subset, fill_value=0)

df[columns_subset]

