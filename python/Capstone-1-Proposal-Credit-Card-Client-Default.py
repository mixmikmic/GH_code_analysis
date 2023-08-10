import pandas as pd

df = pd.read_csv('default of credit card clients.csv', 
                 header=1, 
                 index_col=0)
df.head(10)

