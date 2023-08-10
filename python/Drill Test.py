import pyodbc
from pandas import *
import pandas as pd
import unicodedata

# initialize the connection
conn = pyodbc.connect("DSN=MapR Drill", autocommit=True)
cursor = conn.cursor()

# setup the query and run it
s = "SELECT * FROM `dfs.tmp`.`./companylist.csv2` limit 3"

data = pandas.read_sql(s, conn)
data

# fetch and display filtered output
cursor.execute(s)
row = cursor.fetchone() 
print row[0], row[1]

# Here's how to select from MySQL
s = "select * from ianmysql.mysql.`user`"
data = pandas.read_sql(s, conn)
data

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

# Here's how to do a JOIN
s = "SELECT tbl1.name, tbl2.name FROM `dfs.tmp`.`./names.json` as tbl1 JOIN ianmysql.cars.`car` as tbl2 ON tbl1.id=tbl2.customerid"
data = pandas.read_sql(s, conn)
data

# Here's how to do a JOIN
s = "SELECT tbl1.name, tbl2.address, tbl3.name as car FROM `dfs.tmp`.`./names.json` as tbl1 JOIN `dfs.tmp`.`./addressunitedstates.json` as tbl2 ON tbl1.id=tbl2.id JOIN ianmysql.cars.`car` as tbl3 ON tbl1.id=tbl3.customerid"
data = pandas.read_sql(s, conn)
data

