import requests
from bs4 import BeautifulSoup as bs
import pandas as pd
import re

#download HTML and create Beautiful Soup object
#start 
url = "http://www.draftexpress.com/nba-pre-draft-measurements/all/NBA+Draft+Combine"

response = requests.get(url)
html = response.content

soup = bs(html, 'lxml')

#find all of the links to the various pages
page_nums = soup.findAll('a', string=re.compile('\d'))

#create list of page labels that aren't the starting page
start_nums = []
for p in page_nums:
    start_nums.append(int(p.string))

print(start_nums)
                      

#the part of the url that doesn't change from page to page
url_base = 'http://www.draftexpress.com/nba-pre-draft-measurements/all/NBA+Draft+Combine/all/all'

#create list of page urls
pages = [url_base]
for num in start_nums:
    pages.append(url_base + '/' + str(num))

print(pages)

#retrieve data table from starting pages
table = soup.find('table', attrs={'class': 'sorttable'})

#print(table.prettify())

#create list of column headers
df_headers = []
for item in table.findAll('th'):
    #only include headers that don't contain sub categories
    if item.has_attr("rowspan") and item.get("rowspan") == "2":
        df_headers.append(item.string.strip())

#I had to hard code the columns that contain sub categories i.e. height -> no shoes, shoes; vertical -> max, max reach, no, step, no step reach; hand -> length, width

#set desired start indices for sub categories in list of column headers
height_IDX = 3   
vertical_IDX = 7
hand_IDX = 13

for i in range(len(table.findAll('tr')[1].findAll('th'))):
    item = table.findAll('tr')[1].findAll('th')[i].string.strip()
    
    if i in range(0,2):
        df_headers.insert(height_IDX, item)
        height_IDX += 1
    elif i in range(2,6):
        df_headers.insert(vertical_IDX, item)
        vertical_IDX += 1
    else:
        df_headers.insert(hand_IDX, item)
        hand_IDX += 1
        
print(df_headers)

#create dictionary from headers list: one for containing data; the other for containing index reference
df_dict = {}
df_idx_ref = {}
idx = 0
for name in df_headers:
    df_dict[name] = []
    df_idx_ref[idx] = name
    idx += 1
    
print("df_dict: {}\n".format(df_dict))
print("df_idx_ref: {}".format(df_idx_ref))

#go through each page to populate df_dict
for page in pages:
    
    url = page
    response = requests.get(url)
    html = response.content
    soup = bs(html, 'lxml')
    table = soup.find('table', attrs={'class': 'sorttable'})

    #populate df_dict with corresponding data from each row
    rows = table.findAll('tr')[2:]

    for row in rows:
        data = row.findAll('td')
        idx = 0
        for d in data:
            if d.has_attr('data-order'):
                if d.get('data-order').strip() in ['-1.0', '-1.00', '-']:
                    df_dict[df_idx_ref[idx]].append(None)
                else:
                    df_dict[df_idx_ref[idx]].append(round(float(d.get('data-order').strip()), 2))

            else:
                #many columns don't contain data-order attribute
                #need to convert data type for columns: max, max reach, no step, no step reach, body fat, bench, agility, sprint
                if idx in [7, 8, 9, 10, 12, 15, 16, 17] and d.text.strip() != '-':
                    df_dict[df_idx_ref[idx]].append(round(float(d.text.strip()),2))
                else:
                    if d.text.strip() == '-':
                        df_dict[df_idx_ref[idx]].append(None)
                    else:
                        df_dict[df_idx_ref[idx]].append(d.text.strip())
            idx += 1

#print out contents of df_dict
for key in df_dict:
    print('{}: {}\n'.format(key, df_dict[key]))

#check to see that each column contains same number of entries
for key in df_dict:
    print("{}: {}".format(key,len(df_dict[key])))

#convert dictionary to dataframe
df = pd.DataFrame(df_dict, columns=df_dict.keys())

#if draft hasn't occurred, sort dataframe in alphabetical order based on player name; change by argument in sort_values from 'Player' to 'Draft Pick'
#reset index for dataframe
df = df.sort_values(by=['Year', 'Draft pick', 'Player']).reset_index(drop=True)

#rename columns that are subcategories of Height, Vertical, or Hand on Draft Express
df = df.rename(columns= {
    'No Shoes': 'Height (No Shoes)',
    'With Shoes': 'Height (With Shoes)',
    'Max': 'Vertical (Max)',
    'Max Reach': 'Vertical (Max Reach)',
    'No Step': 'Vertical (No Step)',
    'No Step Reach': 'Vertical (No Step Reach)',
    'Length': 'Hand (Length)',
    'Width': 'Hand (Width)'
})

df

#get summary statistics of dataframe
round(df.describe(), 2)

#create csv file from dataframe
df.to_csv('nba_draft_combine_all_years.csv')

#Can now perform visualization and analysis on dataframe using matplotlib and pandas
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

#simple scatter plot
df.plot.scatter('Weight', 'Height (No Shoes)')

