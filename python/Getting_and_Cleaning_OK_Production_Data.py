import pandas as pd
import numpy as np

pd.options.display.max_rows = 10

data_dir = 'data/historical/'

import urllib
urllib.urlretrieve ('ftp://ftp.occ.state.ok.us/OG_DATA/historical.ZIP', 'data/historical.zip')

# For Python 3, try:
# import urllib.request 
# urllib.request.urlretrieve(url, filename)

import zipfile
zip_ref = zipfile.ZipFile('data/historical.zip', 'r')
zip_ref.extractall(data_dir)
zip_ref.close()

import os
os.listdir('data/historical/')

def show_file(filename, nrows):
    from itertools import islice
    with open(filename) as f:
        for line in islice(f, nrows):
            print line

show_file("data/historical/1987prodn.txt", 5)

show_file("data/historical/2015prodn.txt", 5)

#generate a list of years that will be used to form filenames
years = np.arange(1988, 2016, 1)

#list to track the dataframes from each year that will be 
#concatenated after processing
annual_data = {}

for year in years:
    
    #data for 1994 is missing, so ignore this year
    if (year == 1994):
        continue
    
    filename = data_dir + str(year)+'prodn.txt'
    print('Reading %s'%filename)
    
    #the file format is different for years 2008 and before
    #so these files need to be handled differently
    #
    #there are 6 lines in years between 1990 and 1995 that have
    #an extra field.  This could be due to an extra delimiter
    #'|' in the line.  These are ignored.
    if year <= 2008:
        data = pd.read_csv(filename, sep="|", engine='c', 
                        doublequote=False,error_bad_lines=False,
                        low_memory=False)
        
        #remove the first row with ------
        data = data.ix[1:]
        
        #remove the whitespace from the column names
        data = data.rename(columns=lambda x: x.strip())
        
        #these columns don't appear in the later years, so drop
        #them now for consistency
        data.drop(['PURCH_SUFFIX', 'OPER_SUFFIX', 'OFB.1'], axis=1, 
                inplace=True)
                
        #the last Gas column for December will be incorrectly
        #named because of whitespace.  Rename it manually now
        data.columns.values[-1] = 'GAS.11'
    else:
        #the files for 2009 and later are much cleaner and 
        #require less processing
        
        #Sometimes there is an extra line at the top at the file
        # that contains the string 'ExtractRecord'.  If this 
        # is the case, ignore the first line
        with open(filename, 'r') as f:
            first_line = f.readline()
        if 'API_NUMBER' not in first_line:
            data = pd.read_csv(filename, sep="|", engine='c', 
            doublequote=False,error_bad_lines=False, skiprows=1)
        else:
            data = pd.read_csv(filename, sep="|", engine='c', 
                        doublequote=False,error_bad_lines=False)

    # remove any entries that don't have a valid well number
    # some files (like 2004 & 2005) have a row with 
    # '39949 rows selected.' at the bottom.  Most of this row
    # will have NaN values, and will cause problems.  Remove
    # them.
    data = data[pd.notnull(data['API_NUMBER'])]
    
    annual_data[year] = data

data = pd.concat(annual_data)

#Form the full unique API number by concatenating the 
# Okalhoma state code '35' with the API_COUNTY and 
# API_NUMBER fields
data['API_NUMBER'] = data['API_NUMBER'].astype(int, raise_on_error=True).apply(lambda x: '{0:0>5}'.format(x))
data['API_COUNTY'] = data['API_COUNTY'].astype(int, raise_on_error=True).apply(lambda x: '{0:0>3}'.format(x))
data['ENTITY_ID'] = '35'+data['API_COUNTY']+data['API_NUMBER']+'0000'

prod_data = data[['GAS', 'OIL', 'GAS.1', 'OIL.1', 'GAS.2', 
             'OIL.2', 'GAS.3', 'OIL.3', 'GAS.4', 'OIL.4',
             'GAS.5', 'OIL.5', 'GAS.6', 'OIL.6', 'GAS.7', 
             'OIL.7', 'GAS.8', 'OIL.8', 'GAS.9', 'OIL.9',
             'GAS.10', 'OIL.10', 'GAS.11', 'OIL.11',
             'ENTITY_ID', 'YEAR']]

#data.apply(pd.to_numeric, errors='ignore')
#convert the data type of all OIL and GAS columns to numeric
for col_name in list(prod_data.columns.values):
    if ('GAS' in col_name) or ('OIL' in col_name):
        prod_data[col_name] = pd.to_numeric(prod_data[col_name], errors='coerce')

#rename the oil and gas montly totals to something more
#descriptive
prod_data.rename(columns={'GAS': 'GAS - January',
                     'OIL': 'OIL - January',
                     'GAS.1': 'GAS - February',
                     'OIL.1': 'OIL - February',
                     'GAS.2': 'GAS - March',
                     'OIL.2': 'OIL - March',
                     'GAS.3': 'GAS - April ',
                     'OIL.3': 'OIL - April',
                     'GAS.4': 'GAS - May',
                     'OIL.4': 'OIL - May',
                     'GAS.5': 'GAS - June',
                     'OIL.5': 'OIL - June',
                     'GAS.6': 'GAS - July',
                     'OIL.6': 'OIL - July',
                     'GAS.7': 'GAS - August',
                     'OIL.7': 'OIL - August',
                     'GAS.8': 'GAS - September',
                     'OIL.8': 'OIL - September',
                     'GAS.9': 'GAS - October',
                     'OIL.9': 'OIL - October',
                     'GAS.10': 'GAS - November',
                     'OIL.10': 'OIL - November',
                     'GAS.11': 'GAS - December',
                     'OIL.11': 'OIL - December',
                     }, inplace=True)

#some entity id's have multiple entries.  I do not know 
#why this is.  it appears like they might be the production
#attributed to different owners. try summing the production
#for now...
prod_data = prod_data.groupby(by=['ENTITY_ID','YEAR']).sum()
prod_data.reset_index(inplace=True)

prod_data

#now clean the formation string.  These tend to have 
#extraneous whitespace
def clean_text_data(in_string):

    out_string = str(in_string).strip()
    return " ".join(out_string.split())

#data["FORMATION"] = data["FORMATION"].astype(str) 
data["FORMATION"] = data["FORMATION"].apply(clean_text_data)
data["WELL_NAME"] = data["WELL_NAME"].apply(clean_text_data)
data["OPERATOR"] = data["OPERATOR"].apply(clean_text_data)
data["PURCHASER"] = data["PURCHASER"].apply(clean_text_data)
data

data.columns

columns_to_keep = ['ENTITY_ID', 'FORMATION', 'LATITUDE', 'LONGITUDE', 'OPERATOR', 'WELL_NAME', 'PURCHASER']

annual_production_data = {}
annual_meta_data = {}

for year in annual_data:
    print('Cleaning %d data...'%year)
    data = annual_data[year]

    
    #Form the full unique API number by concatenating the 
    # Okalhoma state code '35' with the API_COUNTY and 
    # API_NUMBER fields
    data['API_NUMBER'] = data['API_NUMBER'].astype(int, raise_on_error=True).apply(lambda x: '{0:0>5}'.format(x))
    data['API_COUNTY'] = data['API_COUNTY'].astype(int, raise_on_error=True).apply(lambda x: '{0:0>3}'.format(x))
    data['ENTITY_ID'] = '35'+data['API_COUNTY']+data['API_NUMBER']+'0000'
    
    prod_data = data[['GAS', 'OIL', 'GAS.1', 'OIL.1', 'GAS.2', 
                 'OIL.2', 'GAS.3', 'OIL.3', 'GAS.4', 'OIL.4',
                 'GAS.5', 'OIL.5', 'GAS.6', 'OIL.6', 'GAS.7', 
                 'OIL.7', 'GAS.8', 'OIL.8', 'GAS.9', 'OIL.9',
                 'GAS.10', 'OIL.10', 'GAS.11', 'OIL.11',
                 'ENTITY_ID', 'YEAR']]

    #data.apply(pd.to_numeric, errors='ignore')
    #convert the data type of all OIL and GAS columns to numeric
    for col_name in list(prod_data.columns.values):
        if ('GAS' in col_name) or ('OIL' in col_name):
            prod_data[col_name] = pd.to_numeric(prod_data[col_name], errors='coerce')
    
    #rename the oil and gas montly totals to something more
    #descriptive
    prod_data.rename(columns={'GAS': 'GAS - January',
                         'OIL': 'OIL - January',
                         'GAS.1': 'GAS - February',
                         'OIL.1': 'OIL - February',
                         'GAS.2': 'GAS - March',
                         'OIL.2': 'OIL - March',
                         'GAS.3': 'GAS - April ',
                         'OIL.3': 'OIL - April',
                         'GAS.4': 'GAS - May',
                         'OIL.4': 'OIL - May',
                         'GAS.5': 'GAS - June',
                         'OIL.5': 'OIL - June',
                         'GAS.6': 'GAS - July',
                         'OIL.6': 'OIL - July',
                         'GAS.7': 'GAS - August',
                         'OIL.7': 'OIL - August',
                         'GAS.8': 'GAS - September',
                         'OIL.8': 'OIL - September',
                         'GAS.9': 'GAS - October',
                         'OIL.9': 'OIL - October',
                         'GAS.10': 'GAS - November',
                         'OIL.10': 'OIL - November',
                         'GAS.11': 'GAS - December',
                         'OIL.11': 'OIL - December',
                         }, inplace=True)
                         
    #some entity id's have multiple entries.  I do not know 
    #why this is.  it appears like they might be the production
    #attributed to different owners. try summing the production
    #for now...
    prod_data = prod_data.groupby(by=['ENTITY_ID','YEAR']).sum()
    prod_data.reset_index(inplace=True)
    
    
    #add a year column (may remove this when we try summing over the entire frame)
    #prod_data.loc[:,'Year'] = year
    annual_production_data[year] = prod_data
    
    
    #now clean the formation string.  These tend to have 
    #extraneous whitespace
    def clean_formation_string(in_string):
        
        out_string = str(in_string).strip()
        return " ".join(out_string.split())
        
    #data["FORMATION"] = data["FORMATION"].astype(str) 
    data["FORMATION"] = data["FORMATION"].apply(clean_formation_string)
    data["WELL_NAME"] = data["WELL_NAME"].apply(clean_formation_string)
    data["OPERATOR"] = data["OPERATOR"].apply(clean_formation_string)
    data["PURCHASER"] = data["PURCHASER"].apply(clean_formation_string)
    annual_data[year] = data

annual_production_data[2005]

all_data = pd.concat(annual_data)
all_data

temp = annual_data[1996]
temp[temp['ENTITY_ID'] == '35149201050000']
                                            





temp[temp['ENTITY_ID'] == '35149201050000']['PURCHASER']



