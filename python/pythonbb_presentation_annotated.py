#Change this to the path on your computer that you've downloaded all the contents to
path = 'd:/users/jlevy/documents/python bb/'

import pandas as pd

file_name  = 'WB_military.csv'
xwalk_name = 'UN_Regions.csv'

#This file is tab-separated, with five lines at the end that aren't data, uses .. to denoate a missing value,
#and has non-American characters in it that require us to specify the encoding
df    = pd.read_csv(path+file_name,  encoding='ISO-8859-1', sep='\t', skipfooter=5, na_values='..')

#This file is a standard comman-separated document
xwalk = pd.read_csv(path+xwalk_name, encoding='ISO-8859-1')

#This is Python's slicing notation; here we're telling it to give us the first five entries.

df[:5]

xwalk[:5]

#Pandas uses SQL-style merging, where 'inner' is the intersection, 'outer' is everything, 'left' uses
#keys only from the left dataframe (df), and 'right' from the right (xwalk)

df = df.merge(xwalk, left_on='Country Name', right_on='Country', how='outer', indicator=True)
df[:5]

#The _merge column is created by the "indactor=True" command in our merge statement, which is what we use
#to filter on

left_only  = df[df['_merge'] == 'left_only']
right_only = df[df['_merge'] == 'right_only']
df         = df[df['_merge'] == 'both']

print(len(left_only))
print(len(right_only))
print(len(df))

#Extract the unique entries in the 'Country Name' column

left_only['Country Name'].unique()

# Find any country name in the crosswalk that contains the characters "Iran"
xwalk[ xwalk['Country'].str.contains('Iran') ]

#View all the columns in our dataframe

df.columns

#Extract the string names for the 6th and 4th columns above; these could be typed
#in, but since they're long we'll do it this way instead.  Note that Python starts
#its indexing at 0, so [6] is the 7th item from the list above.

total_mil = df.columns.values[6]
perc_gdp = df.columns.values[4]

print(total_mil)
print(perc_gdp)

#Now we group the dataframe by the unique names in the Region column, and tell it how to aggregate the data
#differently within each

by_region = df.groupby('Region').agg({total_mil:'sum', perc_gdp:'mean'})
by_region[:5]

#Just to demonstrate a simple operation, we'll multiply the % of GDP column by .01 to get a 
#more typical way percentages are displayed in data

by_region[perc_gdp] = by_region[perc_gdp] * .01
by_region[:5]

import datetime

start = datetime.datetime(2010, 1, 1)
end   = datetime.datetime(2015, 12, 31)

#The datetime object knows its properties, so we can ask it for each one individual if we want
print(end)
print('day:', end.day)
print('month:', end.month)
print('year', end.year)

#Math operations can be performed on date objects, so that things like the length of
#months or presence of leap years can be handled automatically

elapsed = end - start
print('elapsed days:', elapsed.days)

import numpy as np
from pandas_datareader import data

#The data is put directly into a Pandas dataframe
macro = data.DataReader(['GDP', 'GNP', 'CPILFESL'], "fred", start, end)
macro[:10]

#Since only one of our series is monthly, we resample (aggregate) it all to quarterly

quarterly = macro.resample('QS').mean()
quarterly[:10]

#We create three new columns in order to calculate the log-difference (similar to percentage change)
#of each original column

quarterly[['GDP_ld', 'GNP_ld', 'CPILFESL_ld']] = np.log( quarterly ) - np.log( quarterly.shift() )
quarterly[:10]

#Then we use the simple, automatic plotting method to take a look at it.  This can be highly customized

get_ipython().magic('matplotlib inline')
quarterly[['GDP_ld', 'GNP_ld', 'CPILFESL_ld']].plot()

import requests

url = 'http://www2.census.gov/ces/bds/estab/bds_e_st_release.csv'

name = 'state.csv'

#This one line is the entire code necessary to get the contents of that url
r = requests.get(url)
    
#And we write what we just retrieved to file (you'll find state.csv in your path from the top of the document)
with open(path+name, 'wb') as ofile:
    ofile.write(r.content)

#Or we can do the same thing as above, but loop through a list to do it for more than one series

urls = [['state.csv',  'http://www2.census.gov/ces/bds/estab/bds_e_st_release.csv'],
        ['age.csv',    'http://www2.census.gov/ces/bds/estab/bds_e_age_release.csv'],
        ['sector.csv', 'http://www2.census.gov/ces/bds/estab/bds_e_sic_release.csv']]

for name, url in urls:
    r = requests.get(url)
    with open(path+name, 'wb') as ofile:
        ofile.write(r.content)

from bs4 import BeautifulSoup

#It is not always necessary to specify headers, but some websites require it.  This 'user-agent' string is
#how a website knows, for example, if you're accessing it from a mobile device
headers = {'user-agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.116 Safari/537.36'}

url = 'http://maktaba.tetea.org/exam-results/FTNA2015/S0206.htm'

r = requests.get(url, headers=headers)
soup = BeautifulSoup(r.text, 'lxml')

#We can take a quick look at the raw html from the website - we can see data in there, but it's all jumbled up
soup.text[1000:2000]

#However, the BeautifulSoup object knows how to parse all that html, so we can ask it for things
#by specifying html tags
table = soup.find('table')

rows = []

for row in table.find_all('tr'):
    rows.append([val.text for val in row.find_all('td')])

#We can see the data here now, in the form of lists

print(rows[24])
print(rows[25])
print(rows[26])

#A snapshot of the actual website.  Compare it to the lists above.

from IPython.display import Image
Image(filename='grades.PNG')

#Here we write the data out to a csv file (again, you'll find it in your path).  Note that the
#'Absent' entry for Agness has many columns merged into one.  We use an 'if' statement
#below to parse that differently than entries without an 'Absent'

name = 'grades.csv'

with open(path+name, 'w') as ofile:
    
    for row in rows[24:152]:
        
        #Parse regular students
        if row[4] != 'Absent':
            line = ','.join(row) + '\n'
        
        #Parse absent students
        elif row[4] == 'Absent':
            values = row[:4] + ['Absent']*17 + ['\n']
            line = ','.join(values)
            
        ofile.write(line)

#Due to the odd vertical way they formatted the headers on the website (see the
#screenshot above), we excluded them and then entered our own manually here

header = ['CNO','Repeat','Name','Sex','CIV','HIST','GEO','EDK','KIS','ENG','FRN','PHY',
          'CHEM','BIO','COMP','MATH','FOOD','COMM','KEEP','GPA','CLASS']

df = pd.read_csv(path+name, names=header)

df[:5]

#Tika is a text parser that can handle many things, including PDF documents

from tika import parser

name = 'interstellar.pdf'
text = parser.from_file(path+name)

#It looks like the PDF document has a lot of blank lines, from all its formatting
lines = text['content'].split('\n')

lines[:20]

#Drop out all the blank lines

lines_noblanks = [line for line in lines if line != '']

lines_noblanks[:16]

#Let's find every mention of "theory" in the paper, then record the line it is on plus
#the following line

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

#Take a look at the third (remember the 0-based indexing!) hit on "theory"

print(len(theories))

print(theories[2])

#What a funny thing to say!  Who might be cited in such an odd paper?

bibliogrophy = []

is_bib = False

for line in lines_noblanks:
    
    if 'References' in line:
        is_bib = True
        
    if is_bib is True:
        bibliogrophy.append(line)

#Notice the second entry, and recall that this paper was written in... 1978

bibliogrophy[:9]



