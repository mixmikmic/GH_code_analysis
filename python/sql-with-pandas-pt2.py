# Necessary Libraries
import pandas as pd
import sqlite3
from pandas.io import sql

# Reading CSV to Dataframe
orders = pd.read_csv('./datasets/csv/EuroMart-ListOfOrders.csv', encoding = 'utf-8')
OBD =  pd.read_csv('./datasets/csv/EuroMart-OrderBreakdown.csv', encoding = 'utf-8')
sales_targets =  pd.read_csv('./datasets/csv/EuroMart-SalesTargets.csv', encoding = 'utf-8')

# A: 
new_o_col = {o:o.replace(' ', '_') for o in orders.columns}
new_obd_col = {obd:obd.replace(' ', '_') for obd in OBD.columns}
new_st_col = {st:st.replace(' ', '_') for st in sales_targets.columns}
orders.rename(columns=new_col, inplace=True)
OBD.rename(columns=new_obd_col, inplace=True)
sales_targets.rename(columns=new_st_col, inplace=True)

# A: 
OBD.head()

def remove_str(ele):
    ele = ele.replace('$','')
    ele = ele.replace(',','')
    return ele

OBD[['Sales', 'Profit']] = OBD[['Sales', 'Profit']].applymap(remove_str)

OBD.head()

# Establishing Local DB connection
db_connection = sqlite3.connect('datasets/sql/EuroMart.db.sqlite')

# A: 
OBD.to_sql('OBD', db_connection, if_exists='replace')
orders.to_sql('orders_new', db_connection, if_exists='replace')
sales_targets.to_sql('sales_targets_new', db_connection, if_exists='replace')

orders.head(3)

# A:
query = '''
SELECT "Customer_Name", COUNT("Order_ID") as Number_of_Orders
FROM orders_new
GROUP BY "Customer_Name"
ORDER BY "Number_of_Orders" DESC
'''
df = pd.read_sql(query, db_connection)
df

orders.head(3)

# A:
query = '''
SELECT "Order_ID", "Country", "Region", "State", "City"
FROM orders_new
'''
df = pd.read_sql(query, db_connection)
df.head()

OBD.head()

# A:
query = '''
SELECT "Order_ID", "Profit"
FROM OBD
WHERE "Profit" < 0
'''
df = pd.read_sql(query, db_connection)
df.head()

orders.head(2)

OBD.head(2)

# A:
query = '''
SELECT orders_new."CUSTOMER_NAME", OBD."Product_Name"
FROM orders_new INNER JOIN OBD
ON orders_new."Order_ID" = OBD."Order_ID"
'''
df = pd.read_sql(query, db_connection)
df.head()

# A:
query = '''
SELECT "Country", COUNT("Category") as Number_of_Orders
FROM (SELECT orders_new."Country", OBD."Category"
    FROM orders_new INNER JOIN OBD
    ON orders_new."Order_ID" = OBD."Order_ID") as sub
WHERE "Country" = 'Sweden' AND "Category" = 'Office Supplies'
'''

df = pd.read_sql(query, db_connection)
df.head()

# A:
query = '''
SELECT "Product_Name", "Discount", COUNT("Order_ID") as Number_of_Orders, SUM("Sales") as Total_Sales
FROM OBD
WHERE "Discount" > 0
GROUP BY "Product_Name"
'''
df = pd.read_sql(query, db_connection)
df['Total_Sales'].sum()

# A:
query = '''
SELECT orders_new."Country", SUM(OBD."Quantity") as Total_Quantity_Sold
FROM orders_new INNER JOIN OBD
ON orders_new."Order_ID" = OBD."Order_ID"
GROUP BY orders_new."Country"
'''
df = pd.read_sql(query, db_connection)
df.head()

# A:
query = '''
SELECT orders_new."Country", SUM(OBD."Profit") as Total_Profit
FROM orders_new INNER JOIN OBD
ON orders_new."Order_ID" = OBD."Order_ID"
GROUP BY orders_new."Country"
ORDER BY Total_Profit ASC
'''
df = pd.read_sql(query, db_connection)
df.head(10)

# A:
query = '''
SELECT orders_new."Country", SUM(OBD."Profit")/SUM(OBD."Sales") as Profit_Margin_Ratio
FROM orders_new INNER JOIN OBD
ON orders_new."Order_ID" = OBD."Order_ID"
GROUP BY orders_new."Country"
ORDER BY Profit_Margin_Ratio ASC
'''
df = pd.read_sql(query, db_connection)
df

# A:

# A:

# A:

# A:

# A:

# A:

# A:

