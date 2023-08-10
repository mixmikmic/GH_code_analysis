path = 'd:/users/jlevy/documents/python bb/'

import pandas as pd

file_name  = 'WB_military.csv'
xwalk_name = 'UN_Regions.csv'

df    = pd.read_csv(path+file_name,  encoding='ISO-8859-1', sep='\t', skipfooter=5, na_values='..')
xwalk = pd.read_csv(path+xwalk_name, encoding='ISO-8859-1')

df[:5]

xwalk[:5]

df = df.merge(xwalk, left_on='Country Name', right_on='Country', how='outer', indicator=True)
df[:5]

left_only  = df[df['_merge'] == 'left_only']
right_only = df[df['_merge'] == 'right_only']
df         = df[df['_merge'] == 'both']

print(len(left_only))
print(len(right_only))
print(len(df))

left_only['Country Name'].unique()

# Find any country name in the crosswalk that contains the characters "Iran"
xwalk[
      xwalk['Country'].str.contains('Iran')
      ]

df.columns

total_mil = df.columns.values[6]
perc_gdp = df.columns.values[4]

print(total_mil)
print(perc_gdp)

by_region = df.groupby('Region').agg({total_mil:'sum', perc_gdp:'mean'})
by_region[:5]

by_region[perc_gdp] = by_region[perc_gdp] * .01
by_region[:5]

import datetime

start = datetime.datetime(2010, 1, 1)
end   = datetime.datetime(2015, 12, 31)

print(end)
print('day:', end.day)
print('month:', end.month)
print('year', end.year)

elapsed = end - start
print('elapsed days:', elapsed.days)

import numpy as np
from pandas_datareader import data

macro = data.DataReader(['GDP', 'GNP', 'CPILFESL'], "fred", start, end)
macro[:10]

quarterly = macro.resample('QS').mean()
quarterly[:10]

quarterly[['GDP_ld', 'GNP_ld', 'CPILFESL_ld']] = np.log( quarterly ) - np.log( quarterly.shift() )
quarterly[:10]

get_ipython().magic('matplotlib inline')
quarterly[['GDP_ld', 'GNP_ld', 'CPILFESL_ld']].plot()

import requests

url = 'http://www2.census.gov/ces/bds/estab/bds_e_st_release.csv'

name = 'state.csv'

r = requests.get(url)
    
with open(path+name, 'wb') as ofile:
    ofile.write(r.content)

urls = [['state.csv',  'http://www2.census.gov/ces/bds/estab/bds_e_st_release.csv'],
        ['age.csv',    'http://www2.census.gov/ces/bds/estab/bds_e_age_release.csv'],
        ['sector.csv', 'http://www2.census.gov/ces/bds/estab/bds_e_sic_release.csv']]

for name, url in urls:
    r = requests.get(url)
    with open(path+name, 'wb') as ofile:
        ofile.write(r.content)

from bs4 import BeautifulSoup

headers = {'user-agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.116 Safari/537.36'}

url = 'http://maktaba.tetea.org/exam-results/FTNA2015/S0206.htm'

r = requests.get(url, headers=headers)
soup = BeautifulSoup(r.text, 'lxml')

soup.text[1000:2000]

table = soup.find('table')

rows = []

for row in table.find_all('tr'):
    rows.append([val.text for val in row.find_all('td')])

print(rows[24])
print(rows[25])
print(rows[26])

from IPython.display import Image
Image(filename='grades.PNG')

name = 'grades.csv'

with open(path+name, 'w') as ofile:
    
    for row in rows[24:152]:
        
        if row[4] != 'Absent':
            line = ','.join(row) + '\n'
        
        elif row[4] == 'Absent':
            values = row[:4] + ['Absent']*17 + ['\n']
            line = ','.join(values)
            
        ofile.write(line)

header = ['CNO','Repeat','Name','Sex','CIV','HIST','GEO','EDK','KIS','ENG','FRN','PHY',
          'CHEM','BIO','COMP','MATH','FOOD','COMM','KEEP','GPA','CLASS']

df = pd.read_csv(path+name, names=header)

df[:5]

from tika import parser

name = 'interstellar.pdf'
text = parser.from_file(path+name)

lines = text['content'].split('\n')

lines[:20]

lines_noblanks = [line for line in lines if line != '']

lines_noblanks[:16]

theories = []
get_next = False

for line in lines_noblanks:
    
    if 'theory' in line:
        current = line + ' '
        get_next = True
    
    elif get_next is True:
        current += line
        theories.append(current)
        get_next = False

print(len(theories))

print(theories[2])

bibliogrophy = []

is_bib = False

for line in lines_noblanks:
    
    if 'References' in line:
        is_bib = True
        
    if is_bib is True:
        bibliogrophy.append(line)

bibliogrophy[:9]

