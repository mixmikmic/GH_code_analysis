import pandas as pd
from pdftables import get_tables
import re

pdfFile='./Input/health.pdf'
pdfObj = open(pdfFile, 'rb')
tables = get_tables(pdfObj)

# exclude the headers to just include the country-records
for table in  tables:
    table[:] = table[5:]

for table in tables:
    for row in table[:5]:
        print(row)
    print('>>>>>>>')

# function to generate the dict for DataFrame formation
def add_to_dict(table,data):
    for row in table:
        if row[0] == 'SUMMARY':
            break
        if row[0] != '':
            data[row[0]] = row[1:]

data1 = dict()
data2 = dict()
for table in tables[:3]:
    add_to_dict(table,data1)

add_to_dict(tables[3],data2)

#create DataFrame for the first 3 pages and the fourth page
df2 = pd.DataFrame(data2)
df1 = pd.DataFrame(data1)
df2 = df2.T
df1 = df1.T
df2 = df2.reset_index()
df1 = df1.reset_index() 
df1.rename(columns={'index':"Country Name"}, inplace=True)
df2.rename(columns={'index':"Country Name"}, inplace=True);

df1.shape

df2.shape

df1.iloc[:,4].apply(lambda x: True if len(x)>4 else False).value_counts()

s = '100'
p1 = '(\d{2,3}?)(\d+$)'
p2 = '(\d{3})(\d+$)'
p3 = '(\d+)–'
p4 = '\d+'
re.match(p4,s).group(0)



