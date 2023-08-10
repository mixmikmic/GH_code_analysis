# http://stackoverflow.com/questions/19410042/how-to-make-ipython-notebook-matplotlib-plot-inline
# %Matplotlib notebook has amazing interactive output, possibly better than %Matplotlib inline 
get_ipython().magic('matplotlib notebook')
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
import config
import pymysql.cursors
import pandas as pd

connection = pymysql.connect(host='localhost',
                             user='root',
                             password=config.MYSQL_SERVER_PASSWORD,
                             db='youtubeProjectDB',
                             charset='utf8mb4', # deals with the exotic emojis
                             cursorclass=pymysql.cursors.DictCursor)

sql1 = """SELECT query_q, COUNT(query_q) FROM search_api GROUP BY query_q ORDER BY COUNT(query_q) DESC;"""


sql2 = """SELECT videoId, query_q, publishedAt FROM search_api;"""

df1 = pd.read_sql(sql1, connection)
df2 = pd.read_sql(sql2, connection).set_index(['publishedAt'])

df1

dt = df2.groupby(['query_q', pd.TimeGrouper('6M')]).count().unstack().videoId.T
dt.plot(subplots=True, kind='bar')

