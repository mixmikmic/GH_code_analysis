import sqlalchemy
from sqlalchemy import (Table, Column, Integer, String, Float, Date,
                        MetaData, create_engine)
from sqlalchemy.sql import select
import pandas as pd

reddit_df = pd.read_csv('reddit_data.csv', encoding='utf-8')
reddit_df

engine = create_engine('sqlite:///reddit.db')
connection = engine.connect()

reddit_df.to_sql('reddit.db', connection, if_exists='replace')

