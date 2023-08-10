import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

from sqlalchemy import create_engine
import pandas as pd

# DSN format for database connections:  [protocol / database  name]://[username]:[password]@[hostname / ip]:[port]/[database name here]
engine = create_engine('postgresql://dsi_student:gastudents@dsi.c20gkj5cvu3l.us-east-1.rds.amazonaws.com:5432/northwind')

pd.read_sql("SELECT tablename FROM pg_catalog.pg_tables WHERE schemaname='public' LIMIT 5", con=engine)

sql = """
SELECT tablename 
FROM pg_catalog.pg_tables 
WHERE schemaname='public'
"""

pd.read_sql(sql, con=engine)

sql = """
SELECT "table_name", "data_type" AS "type", "table_schema" AS "schema"
FROM INFORMATION_SCHEMA.COLUMNS
WHERE "table_schema" = 'public'
"""

pd.read_sql(sql, con=engine)

sql = """
SELECT * FROM orders 
LIMIT 3""" 
pd.read_sql(sql, con=engine)

# lets have the second one be a random table
import random
table = random.choice(['products','usstates'])

sql = "SELECT * FROM %s LIMIT 3" %table
print table
pd.read_sql(sql, con=engine)

sql = """
SELECT DISTINCT "CategoryName", "CategoryID"
FROM categories
ORDER BY "CategoryID"
"""

pd.read_sql(sql, con=engine)

sql = """
SELECT "CategoryID", COUNT(*) AS "Products Per Category"
FROM products
GROUP BY "CategoryID"
ORDER BY "CategoryID"
"""

pd.read_sql(sql, con=engine)

sql = """
SELECT "CategoryID", COUNT(*) AS "Products Per Category"
FROM products
WHERE "Discontinued" != 1
GROUP BY "CategoryID"
ORDER BY "Products Per Category" DESC
"""

pd.read_sql(sql, con=engine)

sql = """
SELECT "ProductID", "ProductName", "UnitPrice" 
FROM products
WHERE "Discontinued" != 1
ORDER BY "UnitPrice" DESC
LIMIT 5
"""

pd.read_sql(sql, con=engine)

sql = """
SELECT "ProductID", "ProductName", "UnitPrice", "UnitsInStock"
FROM products
WHERE "Discontinued" != 1
ORDER BY "UnitPrice" DESC
LIMIT 5
"""

pd.read_sql(sql, con=engine)

# Lets look at our accounts recievable.  (Money we are owed)
sql = """
SELECT "ProductID", "ProductName", "UnitPrice", "UnitsInStock","UnitsOnOrder" 
FROM products
ORDER BY "UnitsInStock" DESC
"""

df = pd.read_sql(sql, con=engine)
df['accounts_recievable'] = df['UnitPrice']*df['UnitsOnOrder']
df.sort_values(by="accounts_recievable", inplace=True, ascending =True)
df[["ProductName", "accounts_recievable"]].plot(kind="barh", x="ProductName", figsize=(10,20))

sql = """
SELECT COUNT(*) FROM orders
"""
pd.read_sql(sql, con=engine)

sql = """
SELECT EXTRACT(YEAR FROM "OrderDate") AS "Year", COUNT(*) FROM orders
GROUP BY EXTRACT(YEAR FROM "OrderDate")
"""
pd.read_sql(sql, con=engine)

sql2 = """
SELECT "OrderDate" FROM orders
"""
pd.read_sql(sql2, con=engine).head()

sql = """
SELECT EXTRACT(quarter FROM "OrderDate") AS "Quarter", COUNT(*) FROM orders
GROUP BY EXTRACT(quarter FROM "OrderDate")
"""
pd.read_sql(sql, con=engine)

df = pd.read_sql(sql, con=engine)
df.sort_values(by="Quarter").plot(x="Quarter", y="count", title="Orders by Quarter",xticks =[1,2,3,4])

sql = """
SELECT "ShipCountry", COUNT(*) as "Shipped"
FROM orders
GROUP BY "ShipCountry"
ORDER BY "Shipped" DESC
"""
pd.read_sql(sql, con=engine)

sql = """
SELECT "ShipCountry", COUNT(*) as "Shipped"
FROM orders
GROUP BY "ShipCountry"
ORDER BY "Shipped" ASC
LIMIT 1
"""
pd.read_sql(sql, con=engine)

sql = """
SELECT AVG(AGE("ShippedDate", "OrderDate")) as "Avg Ship Time"
FROM orders
"""
pd.read_sql(sql, con=engine)

sql = """
SELECT "CustomerID", COUNT(*) as "Orders"
FROM orders
GROUP BY "CustomerID"
ORDER BY "Orders" DESC
LIMIT 1
"""
pd.read_sql(sql, con=engine)

# Using Sql Joins
sql = """
SELECT 
    o."OrderID", o."CustomerID", od."UnitPrice" * od."Quantity" AS "net_order"
FROM orders o
LEFT JOIN order_details od 
ON 
    o."OrderID" = od."OrderID"

ORDER BY 3 DESC
LIMIT 1
"""
pd.read_sql(sql, con=engine)

query = """
SELECT o."CustomerID", sum(n."sum_revenue") as customer_sum

FROM orders o 

/* blah blah 
blah blah
blah blah
blah blah
*/

RIGHT JOIN 
    (SELECT od."OrderID", sum((od."UnitPrice" * od."Quantity" * (1-(od."Discount")))) as sum_revenue
    FROM order_details od
    GROUP BY od."OrderID") n

ON o."OrderID" = n."OrderID"
GROUP BY o."CustomerID"
ORDER BY customer_sum DESC
"""
pd.read_sql(query, con = engine)

# Using pd.merge
# Need to update this to match the SQL statement above!
pd.merge(left=pd.read_sql("""SELECT "OrderID", "CustomerID" FROM orders""", con=engine), 
         right =pd.read_sql("""SELECT "OrderID","UnitPrice" * "Quantity" * (1-(od."Discount")) AS "net_order"FROM order_details """, con=engine), 
         on="OrderID").sort_values(by='net_order', ascending=False).head(1)

sql = """
SELECT 
(SELECT SUM(net_order) FROM 
            (
            SELECT o."CustomerID", sum(n."sum_revenue") as net_order

            FROM orders o 

            INNER JOIN 
            (SELECT od."OrderID", sum((od."UnitPrice" * od."Quantity" * (1-(od."Discount")))) as sum_revenue
            FROM order_details od
            GROUP BY od."OrderID") n

            ON o."OrderID" = n."OrderID"
            GROUP BY o."CustomerID"
            ORDER BY net_order DESC
            
            LIMIT 5
            ) as Top_5
            
            
            ) / SUM(details."UnitPrice" * details."Quantity"* (1-(details."Discount"))) AS Top_5_pct
FROM order_details details

"""
pd.read_sql(sql, con=engine)



