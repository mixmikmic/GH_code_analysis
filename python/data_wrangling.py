import pandas as pd
import numpy as np
import time
import unicodedata
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, WebDriverException

import datetime
from datetime import timedelta, datetime

import csv
import os

import sys
sys.path.append('./lib/')

from functions import *
import functions

from tqdm import tnrange, tqdm_notebook
from unidecode import unidecode

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

import lxml.html
import lxml

import glob

get_ipython().magic('matplotlib inline')

import matplotlib
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_style("darkgrid")

get_ipython().magic('load_ext autoreload')

get_ipython().magic('autoreload 2')

pattern = 'output/**/*.csv'
pattern = 'output/scraped_data_20170417.csv'

files = glob.glob(pattern, recursive=True)
files
    

col_names = ['entity', 'department', 'contract', 'year', 'month', 'group', 'lastn', 'lastn2', 'givenn', 'degree', 
             'qualif', 'position', 'region', 'specials', 'currency', 'salary', 'overt', 'start_date', 'end_date', 'obs', 'url', 'other']
len(col_names)


dfscraped = pd.DataFrame()

for f in files:
    if( ('scraped' in f) and (not 'data_clean' in f)):
        df = pd.read_csv(f, header=None, encoding='latin1', names=col_names, warn_bad_lines=True, low_memory=False)
        df['filename'] = f
        dfscraped = pd.concat([dfscraped, df], ignore_index=True)

print(dfscraped.shape)

dfscraped.filename.value_counts()

dfscraped.entity.value_counts()

dfscraped.url.nunique()

retake_entity = list(dfscraped[dfscraped.entity.str.contains("b'", na=False)].entity.value_counts().index)
retake_entity = retake_entity + list(dfscraped[dfscraped.entity.str.contains("Ã", na=False)].entity.value_counts().index)
retake_entity = retake_entity + list(dfscraped[dfscraped.entity.str.contains("Ã­", na=False)].entity.value_counts().index)
retake_entity = retake_entity + ['Ministerio del Interior y Seguridad Pblica', 'Presidencia de la Repblica ']  

print('Total Urls', dfscraped.url.nunique())
print('Urls to retake', dfscraped[dfscraped.entity.isin(retake_entity)].url.nunique())

retake_entity_urls = []
for u in list(set(dfscraped.ix[dfscraped.entity.isin(retake_entity), 'url'].unique())):
    if str(u) != 'nan':
        if "b'" in u:
            u = u[2:]
        if u[-1] == "'":
            u = u[:len(u)-1]
        retake_entity_urls.append(u)
len(retake_entity_urls)

retake_entity_urls

output_file = output_file = './output/retake_entity_0.csv'

if not os.path.isfile(output_file):
    browser = webdriver.Firefox()
    for url in retake_entity_urls[0:5]:
        functions.getTableData2(output_file, url, browser)
    

retake_entity_df = pd.read_csv(output_file, encoding='latin1', names=col_names)

retake_entity_df.entity.value_counts()

cleanLatin(retake_entity_df)

retake_entity_df['filename'] = output_file
print(retake_entity_df.shape)
print(dfscraped[dfscraped.entity.isin(retake_entity)].shape)
dfscraped = dfscraped[~dfscraped.entity.isin(retake_entity)]
dfscraped = pd.concat([dfscraped, retake_entity_df])
dfscraped.shape

write_clean_file = True
apply_cleanup_again = False


if write_clean_file:
    print('Before Processing:', dfscraped.shape)
    columns_dupe = [col for col in dfscraped.columns if not 'filename' in col]
    cleanLatin(dfscraped)
    dfscraped1 = dfscraped.drop_duplicates(columns_dupe)
    print('After Processing:', dfscraped1.shape)
    
    # Lower case values
    dfscraped1.givenn = dfscraped1.givenn.str.lower()
    dfscraped1.lastn = dfscraped1.lastn.str.lower()
    dfscraped1.lastn2 = dfscraped1.lastn2.str.lower()
    
    # Write file
    dfscraped1.to_csv('./output/scraped_data_clean_20140417.csv', encoding='utf-8', index=False)
    
    
#Before Processing: (2476309, 23)

#After Processing: (2081444, 23)

if not write_clean_file:
    dfscraped1 = pd.read_csv('./output/scraped_data_clean.csv', encoding='latin1', low_memory=False)
    columns_dupe = [col for col in dfscraped1.columns if not 'filename' in col]

if apply_cleanup_again:
    cleanLatin(dfscraped1)
    dfscraped1.drop_duplicates(columns_dupe, inplace=True)

dfscraped1.entity.replace(np.nan, -1).value_counts()

dfjusticia = dfscraped1.loc[dfscraped1.entity == 'Ministerio de Justicia']

dfjusticia.department.value_counts()

dfsename = dfjusticia.loc[dfjusticia.department == 'Servicio Nacional de Menores (SENAME)']

dfsename.groupby(['year', 'month'])['entity'].count()

print(dfsename.shape)
# Monthly data, set define months
dfmonths = pd.read_csv('./data/months.csv', dtype={'month2' : str, 'month3' : int})
dfmonths.month3 = dfmonths.month3.astype(int)

months = list(dfmonths.month2.values)

# all the rows that are months we know they are 2016
# mark them as this transformation
dfsename.ix[:,'TmonthIsKnownYear'] = 0
dfsename.ix[dfsename.year.str.lower().isin(months), 'TmonthIsKnownYear'] = 1

# Write Months
dfsename.ix[dfsename.TmonthIsKnownYear == 1, 'month2'] = dfsename.year.str.lower()
dfsename.ix[dfsename.TmonthIsKnownYear == 0, 'month2'] = 'allyear'
dfsename = pd.merge(dfsename, dfmonths, how='left', on='month2')

# Write years
dfsename.ix[:, 'year2'] = dfsename.year.str.extract('( [0-9]*)$', expand=False)
dfsename.ix[dfsename.TmonthIsKnownYear == 1, 'year2'] = 2016
dfsename.year2 = pd.to_numeric(dfsename.year2)

# Include Rows Flag
dfsename['include'] = 1
dfsename.ix[(dfsename.year2 == 2016) & (dfsename.month2 == 'allyear'), 'include'] = 0 # year 2016
dfsename.ix[(dfsename.year == 0) | (dfsename.year == 1), 'include'] = 0 # zeroes and ones

print(dfsename.shape)

print(dfsename.replace(np.nan, -1).groupby(['include' ,'TmonthIsKnownYear', 'year', 'year2', 'month2', 'month3'])['entity'].count())

print(dfsename.shape)

# Define what df we want to expand to months
cols = [col for col in dfsename.columns if ((col !='month2') and (col != 'month3') )]
df_to_monthly = dfsename.ix[dfsename.month2 == 'allyear', cols]
print(df_to_monthly.shape)

# Keys to mege
df_to_monthly['key'] = 1
dfmonths['key'] = 1

# Generate cartesian product
dfcartesian = pd.merge(df_to_monthly, dfmonths, on='key')
cols2 = [col for col in dfcartesian.columns if col !='key']
dfcartesian = dfcartesian.ix[:,cols2]
print(dfcartesian.shape)

# Remove yearly rows
dfsename = dfsename.ix[dfsename.month2 != 'allyear', :]

# Add monthly rows
dfsename = pd.concat([dfsename, dfcartesian])

print(dfsename.shape)
print(dfsename.replace(np.nan, -1).groupby(['include' ,'TmonthIsKnownYear', 'year2', 'month3'])['entity'].count().unstack('year2'))

# Set the datetimeindex index
dfsename['curDate'] = dfsename.year2.map(int).map(str) + '-' + dfsename.month3.map(int).map(str)
dfsename['curDate2'] = pd.to_datetime(dfsename.curDate, format='%Y-%m')
dfsename = dfsename.set_index(pd.DatetimeIndex(dfsename['curDate2']))

dfsename.ix[dfsename.include == 1].replace(np.nan, -1).groupby('contract').resample('M')['curDate2'].count().unstack('contract').plot()



dfsename['start_date'] = dfsename['start_date'].astype(str)
dfsename['end_date'] = dfsename['end_date'].astype(str)

# Fix some guys
dfsename.start_date = dfsename.start_date.str.replace('28/10/201$', '28/10/2010')

# Transform
dfsename['start_date2'] = pd.to_datetime(dfsename['start_date'], format='%d/%m/%Y', errors='coerce')
dfsename['end_date2'] = pd.to_datetime(dfsename['end_date'], format='%d/%m/%Y', errors='coerce')

print('Start nulls\n', dfsename.ix[pd.isnull(dfsename.start_date2), 'start_date'].value_counts())
print('End nulls\n', dfsename.ix[pd.isnull(dfsename.end_date2), 'end_date'].value_counts())


# Finished before the report date
dfsename.ix[dfsename.end_date2 < dfsename.curDate2 + timedelta(days=14) , 'include'] = 0
# Started after the report date
dfsename.ix[dfsename.start_date2 > dfsename.curDate2 + timedelta(days=14), 'include'] = 0
# Null values
dfsename.ix[dfsename.start_date2 == 'nan', 'include'] = 0
dfsename.ix[dfsename.start_date2 == '', 'include'] = 0
# Indefinite contract
dfsename.ix[(dfsename.start_date2 <= dfsename.curDate2) & (dfsename.end_date == 'Indefinido'), 'include'] = 1



dfsename.ix[dfsename.include == 1].replace(np.nan, -1).groupby('contract').resample('M')['curDate2'].count().unstack('contract').plot()

# Percentage of included
included_pct = pd.concat([dfsename.groupby(['include', 'curDate2'])['entity'].count().unstack('include'),
                         dfsename.groupby(['include', 'curDate2'])['entity'].count().unstack('include').sum(axis=1)], axis=1, join='inner')
included_pct.columns = ['no', 'yes', 'total']
included_pct['no_pct'] = included_pct.no / included_pct.total
list(included_pct.ix[included_pct.no_pct > 0.5].index)

dfsename.ix[(dfsename.curDate2 == '2007-01-01') & (dfsename.include == 0), 'start_date'].value_counts().head()

# ASSUMPTION: We will assume that the start_date 01/06/2007 is wrong and replace by 01/01/2007
dfsename['TcorrectedDate'] = 0
dfsename.ix[dfsename.start_date == '01/06/2007', 'TcorrectedDate'] = 1
dfsename.ix[dfsename.start_date == '01/06/2007', 'start_date2'] = datetime.strptime('01/01/2007', '%d/%m/%Y')
dfsename.ix[(dfsename.start_date2 <= dfsename.curDate2 + timedelta(days=14)) & 
            ((dfsename.end_date2 >= dfsename.curDate2 + timedelta(days=14)) |
            (dfsename.end_date == 'Indefinido')) , 'include'] = 1

dfsename.ix[dfsename.include == 1].replace(np.nan, -1).groupby('contract').resample('M')['curDate2'].count().unstack('contract').plot()

dfsename.ix[dfsename.include == 1].replace(np.nan, -1).groupby('contract').resample('M')['curDate2'].count().unstack('contract')

dfsename['salary2'] = dfsename['salary'].replace('\.', '', regex=True)
dfsename.salary2 = dfsename.salary2.astype(float)
dfsename.salary2.value_counts().head()



dfsename.ix[dfsename.salary2 == 0].groupby(['curDate2'])['entity'].count().sort_index()

dfsename.ix[(dfsename.salary2 == 0) & (dfsename.curDate2 == datetime(year=2014, month=7, day=1)), 'url'].value_counts()

dfsename.columns



dfsename.ix[dfsename.salary2 == 1580920.0].groupby(['curDate2', 'givenn', 'lastn', 'lastn2' ])['entity'].count()

dfsename.ix[(dfsename.givenn == 'miguel angel') & (dfsename.include == 1) & (dfsename.curDate2 == datetime(year=2016, month=6, day=1))].to_csv('./output/doublename.csv')#

dfsename.ix[dfsename.include == 1].groupby(['curDate2', 'givenn', 'lastn', 'lastn2'])['entity'].count().sort_values(ascending=False)



