# Check out an example data file
get_ipython().system('cat files/dat.txt')

# First, explicitly open the file object for reading
f_obj = open('files/dat.txt', 'r')

# You can then loop through the file object, grabbing each line of data
for line in f_obj:
    # Note that I'm removing the new line marker at the end of each line (the '\n')
    print(line.strip('\n'))

# File objects then have to closed when you are finished with them
f_obj.close()

# Since opening and closing files basically always goes together, there is a shortcut to do both of them
#  Use 'with' keyword to open files, and the file object will automatically be closed at the end of the code block
with open('files/dat.txt', 'r') as f_obj:
    for line in f_obj:
        print(line.strip('\n'))

import pandas as pd

# Tab complete to check out all the read functions available
pd.read_

# Let's have a look at a csv file (printed out in plain text)
get_ipython().system('cat files/dat.csv')

# Python has a module devoted to working with csv's
import csv

# We can read through our file with the csv module
with open('files/dat.csv') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=',')
    for row in csv_reader:
        print(', '.join(row))

# Pandas also has functions to directly load csv data
get_ipython().magic('pinfo pd.read_csv')

# Let's read in our csv file
pd.read_csv(open('files/dat.csv'), header=None)

# Let's have a look at a json file (printed out in plain text)
get_ipython().system('cat files/dat.json')

# Think of json's as similar to dictionaries
d = {'firstName': 'John', 'age': '53'}
print(d)

# Python also has a module for dealing with json
import json

# Load a json file
with open('files/dat.json') as dat_file:    
    dat = json.load(dat_file)

# Check what data type this gets loaded as
print(type(dat))

# Pandas also has support for reading in json files
get_ipython().magic('pinfo pd.read_json')

# You can read in json formatted strings with pandas
#  Note that here I am specifying to read it in as a pd.Series, as there is a single line of data
pd.read_json('{ "first": "Alan", "place": "Manchester"}', typ='series')

# Read in our json file with pandas
pd.read_json(open('files/dat.json'), typ='series')

