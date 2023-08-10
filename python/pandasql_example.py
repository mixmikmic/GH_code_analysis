from pandas_datareader import data
import pandas as pd
import datetime
import dateutil
from pandasql import sqldf, load_meat, load_births
pysqldf = lambda q: sqldf(q, globals())

start = datetime.datetime(2015,1,1)
end   = datetime.datetime(2015,10,23)

f = data.DataReader("F", 'yahoo', start, end)

f.reset_index(level=0, inplace=True)

f.head()

f.rename(columns={'Adj Close':'Adj_Close'}, inplace=True)

f.head()

pysqldf("select * from f where High > 15 limit 10")

