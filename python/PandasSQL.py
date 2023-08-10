import pandas as pd
from sqlalchemy import create_engine

eng = create_engine('sqlite:///parkinglot_comp.db')

df = pd.read_sql_table('training', eng)
print(df.head())

df.to_csv('example.csv')

