import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

from sqlalchemy import create_engine
import pandas as pd

# DSN format for database connections:  [protocol / database  name]://[username]:[password]@[hostname / ip]:[port]/[database name here]
engine = create_engine('postgresql://dsi_student:gastudents@dsi.c20gkj5cvu3l.us-east-1.rds.amazonaws.com:5432/northwind')

pd.read_sql("SELECT tablename FROM pg_catalog.pg_tables WHERE schemaname='public' LIMIT 5", con=engine)

# A:
sql_query = '''
SELECT *
FROM information_schema.tables
WHERE table_schema='public'
ORDER BY table_name ASC
'''
pd.read_sql(sql_query, con=engine)

# A:
sql_query = '''
SELECT table_name, data_type, table_schema
FROM information_schema.columns
WHERE table_schema='public'
ORDER BY table_name ASC
'''
pd.read_sql(sql_query, con=engine)

# A:
sql_query = '''
SELECT *
FROM orders
LIMIT 3
'''
pd.read_sql(sql_query, con=engine)

sql_query = '''
SELECT *
FROM products
LIMIT 3
'''
pd.read_sql(sql_query, con=engine)

sql_query = '''
SELECT *
FROM usstates
LIMIT 3
'''
pd.read_sql(sql_query, con=engine)

# A:
sql_query = '''
SELECT DISTINCT "CategoryName", "CategoryID"
FROM categories
ORDER BY "CategoryName" ASC
'''

pd.read_sql(sql_query, con=engine)

sql_query = '''
SELECT "CategoryID", "ProductName"
FROM products
'''

df = pd.read_sql(sql_query, con=engine)
df.groupby(['CategoryID']).size()

# A:
sql_query = '''
SELECT *
FROM products
WHERE "Discontinued" = 0
'''

df = pd.read_sql(sql_query, con=engine)
df.groupby(['CategoryID']).size().sort_values(ascending=False)

# A:
sql_query = '''
SELECT *
FROM products
WHERE "Discontinued" = 0
ORDER BY "UnitPrice" DESC
lIMIT 5
'''

df = pd.read_sql(sql_query, con=engine)
df

# A:
sql_query = '''
SELECT "ProductName", "UnitsInStock"
FROM products
WHERE "Discontinued" = 0
ORDER BY "UnitPrice" DESC
lIMIT 5
'''

df = pd.read_sql(sql_query, con=engine)
df

# A:
sql_query = '''
SELECT "CategoryID", "UnitPrice", "Discontinued"
FROM products
'''

df = pd.read_sql(sql_query, con=engine)
df.groupby(['Discontinued', 'CategoryID']).mean().unstack().plot(kind='bar')

sql_query = '''
SELECT *
FROM orders
'''

df = pd.read_sql(sql_query, con=engine)
df

# A:
sql_query = '''
SELECT COUNT("OrderID") as "Total no of Orders"
FROM orders
'''

df = pd.read_sql(sql_query, con=engine)
df

# A:
sql_query = '''
SELECT "OrderID", CAST(EXTRACT('year' FROM "OrderDate") AS Int) AS year
FROM orders
'''

df = pd.read_sql(sql_query, con=engine)
df.groupby(['year']).size()

# A:
sql_query = '''
SELECT "OrderID", "OrderDate"
FROM orders
'''

df = pd.read_sql(sql_query, con=engine)
df['OrderDate'] = pd.to_datetime(df['OrderDate'])
df['Quarter'] = df['OrderDate'].dt.quarter
df.groupby(['Quarter']).size()

# A:
dict(df.groupby(['Quarter']).size())

# A:

# A:

# A:

# A:

# A:

# A:

# A:

