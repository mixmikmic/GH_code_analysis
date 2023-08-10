from psycopg2 import connect
import psycopg2.sql as pg
import configparser
import datetime
get_ipython().magic('matplotlib inline')
import pandas as pd
import pandas.io.sql as pandasql
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
sns.set(color_codes=True)
from IPython.display import HTML
def print_table(sql, con):
    return HTML(pandasql.read_sql(sql, con).to_html(index=False))

CONFIG = configparser.ConfigParser()
CONFIG.read('../../db.cfg')
dbset = CONFIG['DBSETTINGS']
con = connect(**dbset)

sql = '''SELECT date_trunc('month', datetime_bin) as yyyymm, COUNT(nullif(startpointname, 'E')) AS "Number EB obs", COUNT(nullif(startpointname, 'D')) AS "Number WB obs", COUNT(nullif(startpointname, 'E')) - COUNT(nullif(startpointname, 'D')) AS "EB - WB"
  FROM bluetooth.aggr_5min_i95
  INNER JOIN bluetooth.ref_segments USING (analysis_id)
WHERE startpointname IN ('D','E') AND endpointname IN ('D','E') AND datetime_bin >= '2016-01-01' AND datetime_bin < '2016-01-01'::DATE + INTERVAL '1 Year' 
GROUP BY yyyymm ORDER BY yyyymm;'''
print_table(sql, con)

sql = '''

SELECT extract('hour' FROM datetime_bin) AS "Hour", COUNT(nullif(startpointname, 'E')) AS "Number EB obs", COUNT(nullif(startpointname, 'D')) AS "Number WB obs", COUNT(nullif(startpointname, 'E')) - COUNT(nullif(startpointname, 'D')) AS "EB - WB"
  FROM bluetooth.aggr_5min_i95
  INNER JOIN bluetooth.ref_segments USING (analysis_id)
WHERE startpointname IN ('D','E') AND endpointname IN ('D','E') AND datetime_bin >= '2016-01-01' AND datetime_bin < '2016-01-01'::DATE + INTERVAL '1 Year' AND EXTRACT('isodow' FROM datetime_bin) < 6
GROUP BY "Hour" ORDER BY "Hour";
'''
print_table(sql, con)



