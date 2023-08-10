import csv
import requests
from bs4 import BeautifulSoup

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

df = pd.DataFrame()
# Get a list of dictionaries for the injuries 
injuries_data = []
for i in range(0,723): # number of page increments plus one
    url_string = "http://www.prosportstransactions.com/baseball/Search/SearchResults.php?Player=&Team=&BeginDate=1999-03-01&EndDate=2017-11-01&DLChkBx=yes&submit=Search&start="+str(25* i )
    req = requests.get(url_string)
    soup = BeautifulSoup(req.content, 'lxml')
    for item in soup.find_all("tr", {"align":"left"}):# Code for each individual page to capture data
        raw_text = item.text.strip().split("\n")
        injuries_data.append(raw_text)

# Create a dataframe from the injuries data for 723 pages, with 25 per page = 18075 ish        
df = pd.DataFrame(injuries_data)
df.head()

df.columns = ['Date','Team','Acquired','Relinquished','Notes']
df.head()

df.shape[0]

# Create a dummy column that is 1 if the row represents an injury 
# or a 0 if the row represents a player reactivated.
df['Injury'] = [1 if 'placed' in text else 0 for text in df.Notes]

# Start to extract the number out of the Notes column.
# Replace the hyphen in '15-day' with a space to help splitting and extracting digits.
df.Notes = df.Notes.apply(lambda x: x.replace('-',' '))

def filter_notes_for_DL(notes):
    if '15' in notes:
        return 15
    elif '60' in notes:
        return 60
    elif '10' in notes:
        return 10
    elif '7' in notes:
        return 7
    elif 'restricted' in notes:
        return 0
    elif 'temporary' in notes:
        return 0
    else:
        return 0

df['DL_length'] = df.Notes.map(filter_notes_for_DL)

def extract_injury(notes):
    """Function parses notes column
    to obtain the injury type and returns a string"""
    if len(notes.split('with')) > 1:
        return notes.split('with')[1]
    else:
        return 'unknown'
  

df.Notes.head()

# Create a column that describes the type of injury based on the notes column using
# the function I created: extract_injury, df['Injury_Type']
df['Injury_Type'] = df.Notes.map(extract_injury)

# What kind of injuries are we looking at?
df['Injury_Type'].value_counts()

# Remove rows where df['Injury']==0
print('Before removing reactivations:',df.shape)
df = df[df.Injury != 0]
print('With only placements onto the Disabled List:',df.shape)

df.to_csv('injuries.csv',index=False)

