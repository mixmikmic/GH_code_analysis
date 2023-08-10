import pandas as pd
import sqlite3

#If you want to build the database from scratch, delete any outstanding copy
#Uncomment and run the following command line (!) command
get_ipython().system('rm cqc.sqlite')

con = sqlite3.connect("cqc.sqlite")

url='http://www.cqc.org.uk/sites/default/files/21_September_2016_CQC_directory.zip'

fn=url.split('/')[-1]
stub=fn.split('.')[0]

#Download the data from the CQC website
get_ipython().system('wget -P downloads/ {url}')
get_ipython().system('rm -r data/CQC')
#Unzip the downloaded files into a subdirectory of the data folder, making sure the data dir exists first
get_ipython().system('mkdir -p data')
#The -o flag is overkill - if we hadn't deleted the original folder it would overwirte any similar files
get_ipython().system('unzip -o -d data/CQC downloads/{fn}')
get_ipython().system('mv data/CQC/{stub}.csv  data/CQC/locations.csv')

locations=pd.read_csv('data/CQC/locations.csv',skiprows=4)
locations.rename(columns={'CQC Location (for office use only':'CQC Location',
                          'CQC Provider ID (for office use only)':'CQC Provider ID'}, inplace=True)

locations.head(3)

tmp=locations.set_index(['CQC Location'])
#If the table exists, replace it, under the assumption we are using a more recent version of the data
tmp.to_sql(con=con, name='locations',if_exists='replace')

#We can now run a SQL query over the data
orgcode='1-1000210669'
pd.read_sql_query('SELECT * from {typ} where "CQC Location"="{orgcode}"'.format(typ='locations',orgcode=orgcode), con)

url='http://www.cqc.org.uk/sites/default/files/HSCA%20Active%20Locations.xlsx'

get_ipython().system('rm -r "data/CQC/HSCA Active Locations.xlsx"')
#Download the data from the CQC website
get_ipython().system('mkdir -p data')
get_ipython().system('wget -P data/CQC {url}')

xl=pd.ExcelFile('data/CQC/HSCA Active Locations.xlsx')
xl.sheet_names

directory=pd.read_excel('data/CQC/HSCA Active Locations.xlsx',sheetname='HSCA Active Locations',skiprows=6)

directory.head(2)

directory.columns.tolist()

#Regulated acvitity
[i.split(' - ')[1] for i in directory.columns if i.startswith('Regulated activity')]

#Service types
[i.split(' - ')[1] for i in directory.columns if i.startswith('Service type')]

#Service user bands
[i.split(' - ')[1] for i in directory.columns if i.startswith('Service user band')]

tmp=directory.set_index(['Location ID'])
#If the table exists, replace it, under the assumption we are using a more recent version of the data
tmp.to_sql(con=con, name='directory',if_exists='replace')

#We can now run a SQL query over the data
orgcode='1-1000210669'
pd.read_sql_query('SELECT * from {typ} where "Location ID"="{orgcode}"'.format(typ='directory',orgcode=orgcode), con)

#Find the most popular brands overall
q='''
SELECT "Brand Name",COUNT(*) as cnt from directory
WHERE "Brand Name" !="-"
GROUP BY "Brand Name"
HAVING cnt > 10
ORDER BY cnt DESC
'''
pd.read_sql_query(q, con).head()

url='http://www.cqc.org.uk/sites/default/files/Latest%20ratings.xlsx'

get_ipython().system('rm -r "data/CQC/Latest ratings.xlsx"')
#Download the data from the CQC website
get_ipython().system('mkdir -p data')
get_ipython().system('wget -P data/CQC {url}')

xl=pd.ExcelFile('data/CQC/HSCA Active Locations.xlsx')
xl.sheet_names

ratings=pd.read_excel('data/CQC/HSCA Active Locations.xlsx',sheetname='HSCA Active Locations',skiprows=6)
ratings.head(2)

tmp=ratings.set_index(['Location ID'])
#If the table exists, replace it, under the assumption we are using a more recent version of the data
tmp.to_sql(con=con, name='ratings',if_exists='replace')

#We can now run a SQL query over the data
orgcode='1-1000210669'
pd.read_sql_query('SELECT * from {typ} where "Location ID"="{orgcode}"'.format(typ='ratings',orgcode=orgcode), con)



