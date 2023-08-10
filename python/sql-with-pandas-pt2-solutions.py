# Necessary libraries:
import pandas as pd
import sqlite3
from pandas.io import sql

# Reading in the `.csv` to a DataFrame:
orders = pd.read_csv('../datasets/csv/EuroMart-ListOfOrders.csv', encoding = 'utf-8')
OBD =  pd.read_csv('../datasets/csv/EuroMart-OrderBreakdown.csv', encoding = 'utf-8')
sales_targets =  pd.read_csv('../datasets/csv/EuroMart-SalesTargets.csv', encoding = 'utf-8')

# Renaming columns to remove spaces:
orders.columns = ['order_id','order_date','customer_name','city','country','region',
                        'segment','ship_date','ship_mode','state']
OBD.columns = ['order_id','product_name','discount','sales','profit','quantity',
          'category','sub-category']
 
sales_targets.columns = ['month_of_order_date','category','target']

# Removing dollar signs from the `sales` and `profit` columns:
OBD['sales'] = OBD['sales'].map(lambda x: x.strip('$'))
OBD['sales'] = OBD['sales'].map(lambda x: float(x.replace(',','')))

OBD['profit'] = OBD['profit'].map(lambda x: x.replace('$',''))
OBD['profit'] = OBD['profit'].map(lambda x: float(x.replace(',','')))

# Establishing a local DB connection:
db_connection = sqlite3.connect('../datasets/sql/EuroMart.db.sqlite')

# # Reading out DataFrames as SQL tables:
orders.to_sql(name = 'orders', con = db_connection, if_exists = 'replace', index = False)
OBD.to_sql(name = 'order_breakdown', con = db_connection, if_exists = 'replace', index = False)
sales_targets.to_sql(name = 'sales_targets', con = db_connection, if_exists = 'replace', index = False)

# Getting the column Labels:  
orders.head(1)

OBD.head(1)

sales_targets.head(1)

# Getting all customer names and setting them to a `pandas` object:
customers = sql.read_sql('SELECT customer_name FROM orders', con = db_connection)

# Counting unique values in the list:
customers['customer_name'].value_counts().head()

# City, country, region, and state are all geographic.
sql.read_sql('SELECT city, country, region, state FROM orders', con = db_connection).head()

# Identifying any cell in the `profit` column with a '-' sign:
sql.read_sql('SELECT * from order_breakdown WHERE profit LIKE "%-%"', con = db_connection).head()
# We had not converted values ints prior to writing this.  
# It works with ints and objects!

sql.read_sql('SELECT orders."order_id", orders."customer_name", order_breakdown."product_name"'
'FROM orders '
'LEFT JOIN order_breakdown '
'ON orders."order_id"= order_breakdown."order_id"',
            con = db_connection).head()

sweedish_supplies = sql.read_sql('SELECT orders."order_id", orders."country", order_breakdown."category" '            
'FROM orders '
'LEFT JOIN order_breakdown '
'ON orders."order_id"= order_breakdown."order_id"'
'WHERE orders."country" = "Sweden" and order_breakdown."category"="Office Supplies"',
            con = db_connection)

sweedish_supplies.count()

discount_sales = sql.read_sql('SELECT discount, sales FROM order_breakdown WHERE discount > 0',
                              con = db_connection)

discount_sales['sales'].sum()

order_counts = sql.read_sql('SELECT order_breakdown."quantity", orders."country" '
                            'FROM orders '
                            'INNER JOIN order_breakdown '
                            'ON orders."order_id"= order_breakdown."order_id" ',
            con = db_connection)

order_counts.groupby('country').sum()

# Gather `country` and `profit`. 
profits = sql.read_sql('SELECT order_breakdown."profit", orders."country" '
                            'FROM orders '
                            'INNER JOIN order_breakdown '
                            'ON orders."order_id"= order_breakdown."order_id" ',
            con = db_connection)

# GROUP BY country and sum with sort on `profit`.
profits.groupby('country').sum().sort_values('profit').reset_index()[5:11]

# Total profits/Total sales
# Grabbing profits, sales, and countries:
spr = sql.read_sql('SELECT order_breakdown."profit",order_breakdown."sales", orders."country" '
                            'FROM orders '
                            'INNER JOIN order_breakdown '
                            'ON orders."order_id"= order_breakdown."order_id" ',
            con = db_connection)

# Summing profits and sales by country:
spr2 = spr.groupby('country').sum().sort_values('profit')

# Creating the ratio column:
spr2['ratio'] = spr2['profit']/spr2['sales']

# Sorting by ratio column:
spr2.sort_values('ratio', ascending = False)

sql.read_sql('SELECT orders."ship_mode",order_breakdown."sub-category"'
                            'FROM orders '
                            'INNER JOIN order_breakdown '
                            'ON orders."order_id"= order_breakdown."order_id" '
                            'WHERE "sub-category" = "Bookcases"' ,
            con = db_connection)['ship_mode'].value_counts()


sql.read_sql('SELECT orders."city",orders."country", order_breakdown."sales"'
                            'FROM orders '
                            'INNER JOIN order_breakdown '
                            'ON orders."order_id"= order_breakdown."order_id" ',
            con = db_connection).groupby(['city','country']).sum().sort_values('sales', ascending = False)

# Converting columns to datetime objects from objects:
orders['order_date'] = pd.to_datetime(orders['order_date'])
orders['ship_date'] = pd.to_datetime(orders['ship_date'])

# Engineering a feature that counts the difference in days:
orders['ship_delay'] = (orders['ship_date']-orders['order_date']).astype('timedelta64[h]')/24

# Or, just use `timedelta64[D]` to get days.

# Updating and replacing the `order` data table:
orders.to_sql(name = 'orders', con = db_connection, if_exists = 'replace', index = False)

sql.read_sql('SELECT orders."ship_delay", order_breakdown."category"'
                            'FROM orders '
                            'INNER JOIN order_breakdown '
                            'ON orders."order_id"= order_breakdown."order_id" ',
            con = db_connection).groupby('category').mean()

# First I'm going to extract the information I need using SQL:
month_sales = sql.read_sql('SELECT orders."order_date", order_breakdown."sales",order_breakdown."category" '
             'FROM orders '
             'INNER JOIN order_breakdown '
             'ON orders."order_id" = order_breakdown."order_id" ', 
             con = db_connection)

# Convert `order_date` to a datetime object.
month_sales["order_date"] = pd.to_datetime(month_sales["order_date"])

# Create a column that aggregates dates in 'mon-yy' format.
month_sales['mnth_yr'] = month_sales['order_date'].apply(lambda x: x.strftime('%b-%y'))

# Taking the new date objects and using them to GROUP BY to determine the sum of sales:
month_sales = month_sales.groupby(['mnth_yr','category']).sales.sum().reset_index()

# Pushing this new DataFrame, which was created with monthly aggregates, back to a local SQL DB:
month_sales.to_sql(name = 'sales_by_month', con = db_connection, if_exists = 'replace', index = False)

# Extracting information again, joining the newly created table and the `sales_targets` table:
targets = sql.read_sql('SELECT sales_targets."month_of_order_date", sales_targets."category", sales_targets."target",sales_by_month."sales"'
                      'FROM sales_targets '
                      'INNER JOIN sales_by_month '
                      'ON sales_targets."month_of_order_date" = sales_by_month."mnth_yr" AND '
                      'sales_targets."category" = sales_by_month."category"',
                      con = db_connection)
# This is a double JOIN in that it matches values in two columns.

# Removing string values and converting `targets` to a float dtype:
targets['target'] = targets['target'].map(lambda x: x.replace('$',''))
targets['target'] = (targets['target'].map(lambda x: x.replace(',',''))).astype(float)

# Creating a Boolean list that states whether or not sales exceeded their targets:
exceeded = []
for ind in range(len(targets['target'])):
    if targets['target'][ind] > targets['sales'][ind]:
        exceeded.append(False)
    elif targets['target'][ind] < targets['sales'][ind]:
        exceeded.append(True)

# Appending the list to the DataFrame as a column:
targets['exceeded'] = exceeded

# Getting those values that exceed targets:
targets[targets['exceeded'] == True]

# Getting those values that did not exceed expectations:

targets[targets['exceeded'] == False]



