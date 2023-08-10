# Import modules
import pyodbc
from pandas import *
import pandas as pd
import unicodedata
# Show datasources defined in ~/.odbc.ini
print(pyodbc.dataSources());

# Initialize the connection
conn = pyodbc.connect("DSN=drill64;CHARSET=UTF8", autocommit=True)
conn.setdecoding(pyodbc.SQL_WMETADATA, encoding='utf-32le')
cursor = conn.cursor()

data = pandas.read_sql("SELECT _id, name, address, phone_number, latitude, longitude, zip, first_visit, churn_risk, sentiment FROM `dfs.default`.`./tmp/crm_data` limit 100", conn)
data.address[0:10]

# setup the query and run it
s = "SELECT * FROM `dfs.tmp`.`./companylist.csv2` limit 3"
data = pandas.read_sql(s, conn)
data

# Fetch and display filtered output
cursor.execute(s)
row = cursor.fetchone() 
print row[0], row[1]

# Here's how to select from MySQL
s = "select * from ianmysql.cars.`car`"
data = pandas.read_sql(s, conn)
data

s = "SELECT id,name FROM `dfs.tmp`.`./names.json` limit 3"
data = pandas.read_sql(s, conn)
data

s = "SELECT id,address FROM `dfs.tmp`.`./addressunitedstates.json` limit 3"
data = pandas.read_sql(s, conn)
data

# Here's how to do a JOIN
s = "SELECT tbl1.name, tbl2.address FROM `dfs.tmp`.`./names.json` as tbl1 JOIN `dfs.tmp`.`./addressunitedstates.json` as tbl2 ON tbl1.id=tbl2.id"
data = pandas.read_sql(s, conn)
data

# JOIN two fields
s = "SELECT tbl1.name, tbl2.name FROM `dfs.tmp`.`./names.json` as tbl1 JOIN ianmysql.cars.`car` as tbl2 ON tbl1.id=tbl2.customerid"
data = pandas.read_sql(s, conn)
data

# JOIN three fields
s = "SELECT tbl1.name, tbl2.address, tbl3.name as car FROM `dfs.tmp`.`./names.json` as tbl1 JOIN `dfs.tmp`.`./addressunitedstates.json` as tbl2 ON tbl1.id=tbl2.id JOIN ianmysql.cars.`car` as tbl3 ON tbl1.id=tbl3.customerid"
data = pandas.read_sql(s, conn)
data

sql="SELECT * FROM `dfs.default`.`./tmp/crm_data`"
df=pd.read_sql(sql, conn)

df.head(5)

sql="SELECT * FROM `dfs.tmp`.`./crm_data` where ssn='448-15-9240'"
df=pd.read_sql(sql, conn)
df

sql="SELECT * FROM `dfs.tmp`.`./crm_data` where name='Erika Gallardo'"
df=pd.read_sql(sql, conn)
df

sql="SELECT * FROM `dfs.tmp`.`./crm_data` where name like '%bright%'"
df=pd.read_sql(sql, conn)
df.dropna().head(5)

conn = pyodbc.connect("DSN=drill64", autocommit=True)
conn.setdecoding(pyodbc.SQL_CHAR, encoding='utf-32le', to=str)
conn.setdecoding(pyodbc.SQL_WMETADATA, encoding='utf-32le', to=str)
cursor = conn.cursor()
sql = "SELECT * FROM `dfs.tmp`.`./salary_data.csv2`"
df=pd.read_sql(sql, conn)
df[(df['name'].str.contains("ea"))]

