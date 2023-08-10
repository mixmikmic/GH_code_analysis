# HIDDEN
from pathlib import Path

# Delete the database if it already exists
dbfile = Path("sql_basics.db")
if dbfile.exists():
    dbfile.unlink()
    

import sqlalchemy

# pd.read_sql takes in a parameter for a SQLite engine, which we create below
sqlite_uri = "sqlite:///sql_basics.db"
sqlite_engine = sqlalchemy.create_engine(sqlite_uri)

# HIDDEN

# Creating a table
sql_expr = """
CREATE TABLE prices(
    retailer TEXT,
    product TEXT,
    price FLOAT);
"""
result = sqlite_engine.execute(sql_expr)

# HIDDEN

# Inserting records into the table
sql_expr = """
INSERT INTO prices VALUES 
  ('Best Buy', 'Galaxy S9', 719.00),
  ('Best Buy', 'iPod', 200.00),
  ('Amazon', 'iPad', 450.00),
  ('Amazon', 'Battery pack',  24.87),
  ('Amazon', 'Chromebook', 249.99),
  ('Target', 'iPod', 215.00),
  ('Target', 'Surface Pro', 799.00),
  ('Target', 'Google Pixel 2', 659.00),
  ('Walmart', 'Chromebook', 238.79);
"""
result = sqlite_engine.execute(sql_expr)

# HIDDEN
import pandas as pd

prices = pd.DataFrame([['Best Buy', 'Galaxy S9', 719.00],
                   ['Best Buy', 'iPod', 200.00],
                   ['Amazon', 'iPad', 450.00],
                   ['Amazon', 'Battery pack', 24.87],
                   ['Amazon', 'Chromebook', 249.99],
                   ['Target', 'iPod', 215.00],
                   ['Target', 'Surface Pro', 799.00],
                   ['Target', 'Google Pixel 2', 659.00],
                   ['Walmart', 'Chromebook', 238.79]],
                 columns=['retailer', 'product', 'price'])

import sqlalchemy

# pd.read_sql takes in a parameter for a SQLite engine, which we create below
sqlite_uri = "sqlite:///sql_basics.db"
sqlite_engine = sqlalchemy.create_engine(sqlite_uri)

sql_expr = """
SELECT * 
FROM prices
"""
pd.read_sql(sql_expr, sqlite_engine)

prices

sql_expr = """
SELECT * 
FROM prices
"""
pd.read_sql(sql_expr, sqlite_engine)

sql_expr = """
SELECT retailer
FROM prices
"""
pd.read_sql(sql_expr, sqlite_engine)

sql_expr = """
SELECT DISTINCT(retailer)
FROM prices
"""
pd.read_sql(sql_expr, sqlite_engine)

prices['retailer'].unique()

sql_expr = """
SELECT
    UPPER(retailer) AS retailer_caps,
    product,
    price / 2 AS half_price
FROM prices
"""
pd.read_sql(sql_expr, sqlite_engine)

sql_expr = """
SELECT *
FROM prices
WHERE price < 500
"""
pd.read_sql(sql_expr, sqlite_engine)

sql_expr = """
SELECT *
FROM prices
WHERE retailer = 'Amazon'
    AND NOT product = 'Battery pack'
    AND price < 300
"""
pd.read_sql(sql_expr, sqlite_engine)

prices[(prices['retailer'] == 'Amazon') 
   & ~(prices['product'] == 'Battery pack')
   & (prices['price'] <= 300)]

sql_expr = """
SELECT AVG(price) AS avg_price
FROM prices
"""
pd.read_sql(sql_expr, sqlite_engine)

prices['price'].mean()

sql_expr = """
SELECT retailer, MAX(price) as max_price
FROM prices
GROUP BY retailer
"""
pd.read_sql(sql_expr, sqlite_engine)

sql_expr = """
SELECT retailer, MAX(price) as max_price
FROM prices
GROUP BY retailer
HAVING max_price > 700
"""
pd.read_sql(sql_expr, sqlite_engine)

max_prices = prices.groupby('retailer').max()
max_prices.loc[max_prices['price'] > 700, ['price']]

sql_expr = """
SELECT *
FROM prices
ORDER BY price ASC
LIMIT 3
"""
pd.read_sql(sql_expr, sqlite_engine)

prices.sort_values('price').head(3)

