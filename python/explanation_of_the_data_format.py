get_ipython().system('head -n 5 ../data/eigenmode_info_data_frame.csv')

import pandas as pd

df = pd.read_csv('../data/eigenmode_info_data_frame.csv')

df.head(4)

df.iloc[103:107]

