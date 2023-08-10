get_ipython().magic('matplotlib inline')
#Support embedded charts

#The pandas library is provides support for managing and analysing tabular data
import pandas as pd

#For reporting purposes, it can be handy to associate dates with the corresponding financial year
def getFinancialYear(dt):
    year = dt.year
    if dt.month<4 or (dt.month==4 and dt.day<6): year -= 1
    return year

#Test for getFinancialYear()
#import datetime
#getFinancialYear(datetime.datetime.strptime('6Apr2013', '%d%b%Y'))

#The openprescribing API throws a 403 on requests from pandas direct
#Calling using requests works fine though....
import requests
from StringIO import StringIO

def proxy(url):
    return StringIO(requests.get(url).content)


def op_spending_by_practice(ccg, bnfcode):
    url='https://openprescribing.net/api/1.0/spending_by_practice/?code={bnfcode}&org={ccg}&format=csv'.format(ccg=ccg,bnfcode=bnfcode)
    try:
        df = pd.read_csv(proxy(url), parse_dates=['date'])
    except: return None
    df['bnf']=bnfcode
    df['fyear']=df['date'].apply(getFinancialYear)
    return df

gpspend=op_spending_by_practice('10L', '0601')
gpspend.head()

def op_spending_by_ccg(ccg, bnfcode):
    url='https://openprescribing.net/api/1.0/spending_by_ccg/?code={bnfcode}&org={ccg}&format=csv'.format(ccg=ccg,bnfcode=bnfcode)
    try:
        df = pd.read_csv(proxy(url), parse_dates=['date'])
    except: return None
    df['bnf']=bnfcode
    df['fyear']=df['date'].apply(getFinancialYear)
    return df

ccgspend=op_spending_by_ccg('10L', '0601')
ccgspend.head()

ccgspend.groupby(ccgspend['date'].apply(getFinancialYear))['actual_cost'].sum()

ccgspend.groupby(ccgspend['date'].apply(getFinancialYear))['items'].sum()

ccgav=ccgspend.groupby(ccgspend['date'].apply(getFinancialYear))[['items','actual_cost']].sum()
ccgav['av']=ccgav['actual_cost']/ccgav['items']
ccgav

#http://stackoverflow.com/a/27844045/454773
gpspend[gpspend['fyear']>2013].groupby([gpspend['fyear'],'row_id','row_name'])['actual_cost'].sum().groupby(level=0, group_keys=False).nlargest(5)

gpspend.groupby([gpspend['date'].apply(getFinancialYear),'row_id','row_name'])['items'].sum().groupby(level=0, group_keys=False).nlargest(5)

tmp=gpspend.groupby([gpspend['date'].apply(getFinancialYear),'row_id','row_name'])[['items','actual_cost']].sum()
tmp['av']=tmp['actual_cost']/tmp['items']
tmp

def op_drugs_used_in(bnf,insection=True):
    url='https://openprescribing.net/api/1.0/bnf_code/?q={bnf}&format=csv'.format(bnf=bnf)
    df =pd.read_csv(proxy(url))
    df['bnf']=bnf
    df=df[df['id'].str.startswith(bnf)]
    if insection and 'section' in df:
        df=df[pd.notnull(df['section'])]
    return df

df=op_drugs_used_in('060101')
df

df=op_drugs_used_in('6.1')
df

op_drugs_used_in('060106')

df=pd.DataFrame()
for index, row in op_drugs_used_in('060106').iterrows():
    print(row['id'])
    df2=op_spending_by_practice('10L', row['id'])
    if df2 is not None:
        df=pd.concat([df,df2])

df=pd.merge(df,op_drugs_used_in('060106'),left_on='bnf',right_on='id')

df[df['fyear']==2015].groupby(['id','name','row_name'])['actual_cost','items','quantity'].sum().sort_values(['quantity'],ascending=True).plot(kind='barh',figsize=(15,15))

ddf=df[df['fyear']==2015].groupby(['id','name','row_name'])['actual_cost','items','quantity'].sum().sort_values(['quantity'],ascending=True)
ddf

ddf['cost_per_item']=ddf['actual_cost']/ddf['items']
ddf['quantity_per_item']=ddf['quantity']/ddf['items']
ddf['cost_per_quantity']=ddf['actual_cost']/ddf['quantity']
ddf.sort_values('cost_per_quantity',ascending=False)

op_drugs_used_in('0601060D0',False)



