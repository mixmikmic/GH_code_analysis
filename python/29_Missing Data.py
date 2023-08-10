import numpy as np
import pandas as pd

# df = pd.DataFrame({'A':[1,2,np.nan],
#                   'B':[5,np.nan,np.nan],
#                   'C':[1,2,3]})

df = pd.DataFrame({'A' : [1, 2, np.nan],
                  'B' : [5, np.nan, np.nan],
                  'C' : [1, 2, 3]})

df

df.dropna() # pandas will drop ANY rows with NaN

df.dropna(axis=1) # if you want to drop ANY columns with NaNs 

df.dropna(thresh=2) # one could also feed it a NaN threhold

# when we cant to replace missing values with something
df.fillna(value='FILL VALUE')

df['A'].fillna(value=df['A'].mean())

