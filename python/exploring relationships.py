from tabulate import tabulate
import pandas as pd
import pandas.io.sql as pandasql
import datetime
import configparser
from psycopg2 import connect

CONFIG = configparser.ConfigParser()
CONFIG.read('db.cfg')
dbset = CONFIG['DBSETTINGS']
#Setting up postgresql connection
con = connect(database=dbset['database'],
              host=dbset['host'],
              user=dbset['user'],
              password=dbset['password'])

sql = ''' SELECT COUNT(DISTINCT c.count_info_id) AS "countinfo count",
 COUNT(DISTINCT cim.count_info_id) "countinfomics count",
 SUM(CASE WHEN c.count_info_id = cim.count_info_id THEN 1 ELSE 0 END) AS "Both"
 FROM traffic.countinfo c
 FULL OUTER JOIN traffic.countinfomics cim ON c.count_info_id = cim.count_info_id'''

data = pandasql.read_sql(sql, con)
print(tabulate(data, headers="keys", tablefmt="pipe"))

sql = '''SELECT COUNT(DISTINCT c.count_info_id) AS "countinfomics count",
COUNT(DISTINCT det.count_info_id) "det count",
SUM(CASE WHEN c.count_info_id = det.count_info_id THEN 1 ELSE 0 END) AS "Both"
FROM traffic.countinfomics c
FULL OUTER JOIN traffic.det det ON c.count_info_id = det.count_info_id'''

data = pandasql.read_sql(sql, con)
print(tabulate(data, headers="keys", tablefmt="pipe"))

sql = '''SELECT category_name, COUNT(*)
FROM traffic.countinfomics
NATURAL JOIN traffic.category
GROUP BY category_name
ORDER BY count DESC'''

data = pandasql.read_sql(sql, con)
print(tabulate(data, headers="keys", tablefmt="pipe"))

sql = '''SELECT source1, COUNT(1)
FROM traffic.countinfo 
GROUP BY source1
ORDER BY count DESC'''

data = pandasql.read_sql(sql, con)
print(tabulate(data, headers="keys", tablefmt="pipe"))

sql = '''SELECT extract('year' FROM COALESCE(c.count_date, cim.count_date)) AS "Year", COUNT(DISTINCT c.count_info_id) AS "countinfo count",
 COUNT(DISTINCT cim.count_info_id) "countinfomics count"
FROM traffic.countinfo c
FULL OUTER JOIN traffic.countinfomics cim ON c.count_date = cim.count_date::DATE
GROUP BY "Year"'''

data = pandasql.read_sql(sql, con, index_col='Year')
print(tabulate(data, headers="keys", tablefmt="pipe"))

