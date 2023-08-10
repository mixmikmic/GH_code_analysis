# Import Preliminaries
import pandas as pd

# Create a dataset with the index besing a set of names
raw_data = {'date': ['2014-06-01T01:21:38.004053', '2014-06-02T01:21:38.004053', '2014-06-03T01:21:38.004053'],
        'score': [25, 94, 57]}
df = pd.DataFrame(raw_data, columns= ['date', 'score'])
df

# Transpose the dataset, so that the index (in this case the names) are columns
df['date'] = pd.to_datetime(df['date'])

df = df.set_index(df['date'])

df

