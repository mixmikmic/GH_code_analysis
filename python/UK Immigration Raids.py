url='https://spreadsheetjournalism.files.wordpress.com/2018/01/foi-42354-annex-a.xlsx'

#https://stackoverflow.com/a/34503421/454773
import requests

def downloadAndSave(url, outfile=None, chunk_size=2000):
    r = requests.get(url, stream=True)
    outfile = outfile if outfile else url.split('/')[-1]
    with open(outfile, 'wb') as fd:
        for chunk in r.iter_content(chunk_size):
            fd.write(chunk)
    #print('File written as: {}'.format(outfile)
    return outfile

filename = downloadAndSave(url, '../data/{}'.format(url.split('/')[-1]))

#Show the filename
filename

#pandas is a powerful library for working with tabular datasets
import pandas as pd

sheets=pd.read_excel(filename, sheet_name=None)
sheets.keys()

sheets['ENCOUNTERED NOT ARRESTED'].head()

sheets['ENCOUNTERED NOT ARRESTED'].tail(12)

sheets = None

#pd.read_excel(local_file,sheet_name='Locations',skiprows=0)
enc_not_arr =pd.read_excel(filename, 
                           sheet_name='ENCOUNTERED NOT ARRESTED', 
                           skiprows=3)
enc_not_arr.head()

enc_not_arr =pd.read_excel(filename, 
                           sheet_name='ENCOUNTERED NOT ARRESTED', 
                           skiprows=4)
enc_not_arr.head()

# View 'Grand Total' row
enc_not_arr[enc_not_arr['Nationality']=='Grand Total']

# Get index of 'Grand Total' row
index = enc_not_arr[enc_not_arr['Nationality']=='Grand Total'].index[0]
index

#Include rows above the Grand Total row 
enc_not_arr = enc_not_arr[enc_not_arr.index < index]

enc_not_arr.tail()

enc_not_arr.drop(['Grand Total'], axis=1, inplace = True)

enc_arr = pd.read_excel(filename, 
                           sheet_name='ENCOUNTERED ARRESTED', 
                           skiprows=4)

enc_arr.head()

index = enc_arr[enc_arr['Nationality']=='Grand Total'].index[0]
index

enc_arr.columns

enc_arr.drop(['Grand Total'], axis=1, inplace = True)
enc_arr.columns = [col.strip() for col in enc_arr.columns]

#Include rows above the Grand Total row
index = enc_arr[enc_arr['Nationality']=='Grand Total'].index[0]
enc_arr = enc_arr[enc_arr.index < index]

enc_arr.tail()

#Check sum against original blog post - claimed as: 1143
print( enc_arr['BRISTOL'].sum(), enc_not_arr['BRISTOL'].sum() )
print( enc_arr['BRISTOL'].sum() + enc_not_arr['BRISTOL'].sum() )

enc_not_arr.columns = [col.strip() for col in enc_not_arr.columns]

pd.melt(enc_not_arr, id_vars='Nationality').head()

enc_not_arr_long = pd.melt(enc_not_arr,
                           id_vars='Nationality',
                           var_name ='City',
                           value_name = 'Not Arrested')
enc_not_arr_long.head()

enc_arr_long = pd.melt(enc_arr,
                       id_vars='Nationality',
                       var_name = 'City',
                       value_name = 'Arrested',)
enc_arr_long.head()

enc_merge_long = enc_arr_long.merge( enc_not_arr_long, 
                                    on = ['Nationality', 'City'], 
                                    how='outer')
enc_merge_long.head()

set(['a','b','c']).symmetric_difference(set(['b','c','d']))

set(enc_arr_long['City']).symmetric_difference(set(enc_not_arr_long['City']))

items_in_arr_alone = set(enc_arr_long['City']) - (set(enc_not_arr_long['City']) )
print( 'Items in enc_arr_long not in enc_not_arr_long: {}'.format(items_in_arr_alone) )

items_in_not_arr_alone = set(enc_not_arr_long['City']) - (set(enc_arr_long['City']) )
print( 'Items in enc_not_arr_long not in enc_arr_long: {}'.format(items_in_not_arr_alone) )

enc_arr_long['City'] = enc_arr_long['City'].str.replace('CARDIF', 'CARDIFF')
enc_arr_long['City'] = enc_arr_long['City'].str.replace('SHEFIELD', 'SHEFFIELD')

set(enc_arr_long['City']).symmetric_difference(set(enc_not_arr_long['City']))

set(enc_arr_long['Nationality']).symmetric_difference(set(enc_not_arr_long['Nationality']))

#Let's start to abstract things out
typ= 'Nationality'

items_in_arr_alone = set(enc_arr_long[typ]) - (set(enc_not_arr_long[typ]) )
print( 'Items in enc_arr_long not in enc_not_arr_long: {}'.format(items_in_arr_alone) )

items_in_not_arr_alone = set(enc_not_arr_long[typ]) - (set(enc_arr_long[typ]) )
print( 'Items in enc_not_arr_long not in enc_arr_long: {}'.format(items_in_not_arr_alone) )

enc_not_arr_long['City'] = enc_not_arr_long['City'].str.replace('(951 Convention definition)',
                                                                '(1951 Convention definition)')

#Incorrect version
enc_merge_long = enc_arr_long.merge( enc_not_arr_long, on = ['Nationality', 'City'] )

#Correct version - we need an 'outer' join to ensure all rows from both sides are included
#enc_merge_long = enc_arr_long.merge( enc_not_arr_long, on = ['Nationality', 'City'],
#                                    how='outer' )

enc_merge_long.head()

#Let's use a shorter variable name
df = enc_merge_long

df['Per Cent Arrested'] =  100 * df['Arrested'] / (df['Arrested'] + df['Not Arrested'])
df.sort_values('Per Cent Arrested', ascending = False).head()

df_bycity = df.groupby('City')['Arrested','Not Arrested'].apply( sum)
df_bycity.head()

df_bycity.index

def pc(df):
    return 100 * df['Arrested'] / (df['Arrested'] + df['Not Arrested'])

df_bycity['Per Cent Arrested'] =  pc(df_bycity)
df_bycity.sort_values('Per Cent Arrested', ascending = False).head()

df_bynationality = df.groupby('Nationality')['Arrested','Not Arrested'].apply( sum)
df_bynationality['Per Cent Arrested'] =  pc(df_bynationality)
df_bynationality.sort_values('Per Cent Arrested', ascending = False).head()

df_bycity['Not Arrested']

df_bycity['Arrested']

df_bycity['Arrested'] + df_bycity['Not Arrested']

enc_not_arr[enc_not_arr['BRISTOL'].notnull()]['BRISTOL'].sum()

tmp1 = enc_not_arr[enc_not_arr['BRISTOL'].notnull()][['Nationality','BRISTOL']]
tmp1.head(3)

tmp1['BRISTOL'].sum()

tmp2 = enc_merge_long[enc_merge_long['City']=='BRISTOL'][['Nationality','Not Arrested']]
tmp2.head(3)

tmp3 = tmp1.merge(tmp2, on='Nationality', how='outer')
tmp3.head(3)

tmp3.sum()

tmp3[tmp3['BRISTOL']!=tmp3['Not Arrested']].head()

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

df_bycity.notnull()

x=df_bycity['Not Arrested']
y=df_bycity['Arrested']


fig, ax = plt.subplots()
ax.scatter(x, y)

ax.set_yscale('log')
ax.set_xscale('log')

ax.set_xlabel('Not Arrested')
ax.set_ylabel('Arrested')

for i, txt in enumerate(df_bycity.index):
    ax.annotate(txt, (x[i],y[i]))

x=df_bynationality['Not Arrested']
y=df_bynationality['Arrested']

fig, ax = plt.subplots()
ax.scatter(x, y)

ax.set_yscale('log')
ax.set_xscale('log')

ax.set_xlabel('Not Arrested')
ax.set_ylabel('Arrested')

fig.set_size_inches(18.5, 10.5)

for i, txt in enumerate(df_bynationality.index):
    ax.annotate(txt, (x[i],y[i]))

mycsvfile = '../data/immigrationdata.csv'

#When saving the CSV data, don't save the dataframe index values
enc_merge_long.to_csv( mycsvfile, index=False)

#Preview the saved CSV data
#!head {mycsvfile}

import sqlite3

dbname = '../db/immigration.db'

#Make sure we're working with a clean db
get_ipython().system('rm {dbname}')

conn = sqlite3.connect(dbname)

table = 'immigrationData'

enc_merge_long.to_sql(table, conn, index=False)

q = 'SELECT * FROM {} LIMIT 3'.format(table)

#Explicitly using suggested value of "immigrationData" for the table name
#q = 'SELECT * FROM immigrationData LIMIT 3'

pd.read_sql(q, conn)

q = '''
SELECT City, SUM(Arrested) AS "Total Arrested",
        SUM("Not Arrested") AS "Total Not Arrested", 
        SUM(Arrested) + SUM("Not Arrested") AS "Total"
        FROM {} GROUP BY City
'''.format(table)
pd.read_sql(q, conn)



