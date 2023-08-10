import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

get_ipython().magic('matplotlib inline')

state_regex = re.compile('^Estimated crime in (\w*)\s*$')
data_header_regex = re.compile('^Year,')

crime_by_state = {}
parser_state = 'wait_for_state'
with open("crime-by-state-source.csv") as f:
    for line in f:
        if parser_state == 'wait_for_state':
            m = state_regex.match(line)
            if m:
                state_name = m.group(1)
                parser_state = 'wait_for_data'
        elif parser_state == 'wait_for_data':
            if data_header_regex.match(line):
                parser_state = 'read_data'
                header = line.split(',')[:-1]
                data = []
        elif parser_state == 'read_data':
            if ',' in line:
                data.append(line.split(','))
            else:
                parser_state = 'wait_for_state'
                df = pd.DataFrame(data, columns=header)
                for col in df.columns:
                    df[col] = pd.to_numeric(df[col])
                df.index = df['Year']
                crime_by_state[state_name] = df
crime_by_state = pd.Panel(crime_by_state)

(crime_by_state.loc[:,:,'Violent crime total']/crime_by_state.loc[:,:,'Population']).plot(figsize=(15,10))

crime_total_per_capita = crime_by_state.loc[:,:,'Violent crime total']/crime_by_state.loc[:,:,'Population']

crime_total_per_capita.corr()

crime_by_state = crime_by_state.transpose(2,1,0)

crime_by_state

crime_by_state['Violent crime per capita'] = crime_by_state['Violent crime total']/crime_by_state['Population']
crime_by_state['Total Expenditures for Education'] = np.nan # Prepare column to accept data
crime_by_state['Total Expenditures for Education per capita'] = np.nan # Prepare column to accept data

from os import listdir
from os.path import isfile, join

# Adding a new column 'Violent crime per capita' to each state
# for state in crime_by_state:
#     crime_by_state[state]['Violent crime per capita'] = crime_by_state.loc[state,:,'Violent crime total']/crime_by_state.loc[state,:,'Population']
#     crime_by_state[state]['Total Expenditures for Education'] = np.nan # Prepare column to accept data
#     crime_by_state[state]['Total Expenditures for Education per capita'] = np.nan # Prepare column to accept data

files = [f for f in listdir('budget') if isfile(join('budget', f))]

def cutoff(dummydf):
    if len(dummydf.index) > 60:
        return len(dummydf.index) - 56 - accountfornanrows(dummydf)
    return 0

def accountfornanrows(dummydf):
    if checkfirstrow == None:
        return 0
    else:
        return 1

def checkfirstrow(dummydf):
    if dummydf.iloc[0, 0] == int and np.isnan(dummydf.iloc[0, 0]): # For some weird reason, stfis971b.xls returns str type at the first cell. Seem to be because the dtype is int64
        return 1
    else:
        return None

    
def determine_year(dummydf):
    # Get the year from the index.
    # I chose a random number inbetween because NaN frequently occurs at the beginning
    # and end of some dataframes
    year = int(dummydf.index[5])
    if year > 100:
        return year
    return 1900 + year
    
def te11col(year):
    '''1987-1988 -->
    1989-91 --> DU
    1992, 3, 4, 5, 96,97,1998, --> EJ
    1998, 1999, 2000, 2001, 2, 3, --> EF
    2004-2014 --> EK'''
    if year < 1992:
        return 'DU'
    if year < 1998:
        return 'EJ'
    if year < 2004:
        return 'EF'
    return 'EK'



funding_by_year = {}

for file in files:
    filepath = join('budget', file)
    probe = pd.read_excel(filepath, parse_cols='A, D', index_col=0)#, squeeze=True) # Apparently, two columns are needed to accurately determine column depth
    year = determine_year(probe)
    #Note:
    # A: Year
    # D: State
    # te11col(year): location ot TE11 column
    data = pd.read_excel(filepath, parse_cols='A, D, '+ te11col(year), skiprows=[checkfirstrow(probe)], skip_footer=cutoff(probe),
                         na_values=[-1, -1.0, 'M'], header=0,index_col=1)#, index_col=0) # For some weird reason, the -1 in stfis051b.xls is not being replaced
    data["SURVYEAR"] = year
    for column in data.columns:
        data[column] = pd.to_numeric(data[column])
        data[column]
    funding_by_year[year] = data
    

funding_by_year = pd.Panel(funding_by_year)  #Produced the error: InvalidIndexError: Reindexing only valid with uniquely valued Index objects

crime_by_state['Total Expenditures for Education'] = funding_by_year.loc[:,:,'TE11'].transpose()

# for year in funding_by_year:
#     for state in crime_by_state:
#         crime_by_state[state]['Total Expenditures for Education'][year] = funding_by_year[year]['TE11'][state]
    
# for state in crime_by_state:
#     print(crime_by_state[state]['Total Expenditures for Education'])
#     print(state)
# Observation: Some states in the second data source are not present in the first data source, e.g. Guam
crime_by_state['Total Expenditures for Education']

# for state in crime_by_state:
#     crime_by_state[state]['Total Expenditures for Education per capita'] = crime_by_state.loc[state,:,'Total Expenditures for Education']/crime_by_state.loc[state,:,'Population']
#     print(crime_by_state[state]['Total Expenditures for Education per capita'])

crime_by_state['Total Expenditures for Education per capita'] = crime_by_state.loc['Total Expenditures for Education']/crime_by_state.loc['Population']

crime_by_state['Total Expenditures for Education per capita'].plot(figsize=(15,10))
(crime_by_state['Violent crime total']/crime_by_state['Population']).plot(figsize=(15,10))

education = crime_by_state['Total Expenditures for Education per capita']
crime = (crime_by_state['Violent crime total']/crime_by_state['Population'])

crime.pct_change().corrwith(education.pct_change().shift(20)).plot(kind='barh', figsize=(15,10))

df = pd.DataFrame({
    "education": crime_by_state['Total Expenditures for Education per capita'].mean(),
    "crime": 1/(crime_by_state['Violent crime total']/crime_by_state['Population']).mean(),
})

df /= df.max()

df.plot(kind='bar', figsize=(15,10))



