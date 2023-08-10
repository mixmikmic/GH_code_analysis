from pyhive import hive
from pprint import pprint
import pandas as pd
import os
from altair import *

import IPython.display
def draw(spec):
    IPython.display.display({
        'application/vnd.vegalite.v1+json': spec.to_dict()
    }, raw=True)

pd.set_option('display.max_colwidth', -1) # dont truncate table columns
cwd = os.getcwd()
cwd="/data/shared/snap-samples/Redshift"

c = hive.Connection(host="0.0.0.0",port=10000,auth='NOSASL')
pd.read_sql('show tables',c)

def sql(q, explain=False) :
    # silly hack to handle filesystem prefix for us when creating local tables
    if "{prefix}" in q:
        q = q.replace('{prefix}',cwd)
    df=pd.read_sql(q,c)
    return df

def explain(q):
    df = sql("explain " + q)
    plan = df['plan'][0]
    pprint(plan)
    

sql('show tables')

q="""
with allusers AS ( 
select caldate adate, city,sum(qtysold) q, sum(pricepaid) p 
from salessnap group by caldate,city)
,
someusers AS (
select caldate sdate,city, sum(qtysold) a, sum(pricepaid) b 
from salessnap where likeconcerts='TRUE' AND likejazz='TRUE' group by caldate,city)

select adate,allusers.city, a, b, round(a/q,2)*100 qratio , round(b/p,2)*100 pratio
from allusers, someusers where adate=sdate order by pratio desc limit 5000
"""

df=sql(q)

df.columns

a=Chart(df).mark_bar().encode(x=X('a',
  bin=Bin(maxbins=10)),y='count(*)')
draw(a)

a=Chart(df).mark_line().encode( x=X('adate'), y='sum(a)')
draw(a)

from datetime import datetime, timedelta, date

df['yearmon']=pd.to_datetime(df['adate'],format="%Y-%m-%d" ).dt.strftime("%Y%m")

a=Chart(df).mark_line().encode( x=X('yearmon'), y='sum(a)')
draw(a)

df['a'].autocorr(lag=30)

s1="""
with firstseries AS
(
select caldate adate, sum(qtysold) q
from salessnap group by caldate
)

select * from 
( select adate , q, lead(q, 40)
     
   over ( order by adate desc) as qlag
   from firstseries  ) a
"""

df=sql(s1)

a=Chart(df).mark_circle().encode( x='q', y='qlag')
draw(a)

a=Chart(df).mark_circle().encode( x='q', y='q')
draw(a)

df['q'].corr(df['qlag'])

df

s3=""" with firstseries AS 
( select caldate adate, sum(qtysold) q 
from salessnap group by caldate ) 
select adate, q, qlag,( (q-qlag)/(case when qlag=0 then 1 else qlag end) ) as change
from ( select adate , q, lead(q, 5) over ( order by adate desc) as qlag
from firstseries  ) a order by adate desc
"""

sql(s3)



