import csv
import os
import pprint
import pandas as pd
from os import listdir
from os.path import isfile, join

# an example census table
filename = 'ACS_15_5YR_B01001_with_ann.csv'

# script to open one csv file and remove the margin of error columns
#def open_clean(file):
#    data = []
#    with open(file, 'rb') as f:
#        r = csv.DictReader(f)
#        for line in r:
#            data.append(line)
#    return data

f = pd.read_csv(filename, skiprows=[0]) #skips first row since these csvs have two header rows, 
#skips first row since these csvs have two header rows, 2nd row has the header we want

f.head(1)

# open csv file and remove columns that have margin of error

f = pd.read_csv(filename, skiprows=[0])
keep_col = []
header_list = list(f)
print(header_list)
for header in header_list:
    if 'Margin' not in header:
        keep_col.append(header)
print(keep_col)
new_f = f[keep_col]
new_f.to_csv('clean_'+filename, index=False)
            

# make a list of all files in current notebook directory
onlyfiles = [f for f in listdir() if isfile(join(f))]
onlyfiles

# subset this list by actual census data tables (_with_ann.csv suffix)
census_tables = [file for file in onlyfiles if '_with_ann.csv' in file]
census_tables

# make a function to clean all files in the census_tables list and write new clean cvs file

def open_clean(file_list):
    for file in census_tables:
        f = pd.read_csv(file, skiprows=[0])
        keep_col = [] # list of column names I want to keep
        header_list = list(f)
        #print(header_list)
        for header in header_list:
            if 'Margin' not in header:
                keep_col.append(header)
        #print(keep_col)
        new_f = f[keep_col]
        new_f.to_csv('clean_'+file, index=False)
        print('cleaned and saved: '+'clean_'+file)

open_clean(census_tables)



