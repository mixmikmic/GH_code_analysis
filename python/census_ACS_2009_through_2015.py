import pandas as pd
import numpy as np
import os
import sys

version = ".".join(map(str, sys.version_info[:3]))
print('python version ', version)
print('numpy version ', np.__version__)
print('pandas version ',pd.__version__)

from census import Census
from census import __version__ as census__version__
from us import states

print('census library version ', census__version__)

# Retrieving my Census API key from a file outside of the
# local git repository 
api_key_filepath = os.environ.get('CENSUS_KEY_PATH')
fh = open(api_key_filepath,'r')
api_key = fh.read()
api_key = api_key.rstrip('\n')
fh.close()

c = Census(api_key)
c.acs5.get(('NAME', 'B25034_010E'),
            {'for': 'state:{}'.format(states.CA.fips)}, year=2014)

import requests
print('requests version ', requests.__version__)

year = '2015' ## 5 year 2011 through 2015
census_api_url = "http://api.census.gov/data/" + year + "/acs5"
#payload = {'get':['NAME', 'B05003I_003E'], 'for':{'state':'*'},'key':api_key}
payload = {'get':['B05003I_003E'], 'for':{'county':'*'},'key':api_key}
r = requests.get(census_api_url, params=payload)

# Response is list of lists as a UTF-8 encoded string
# The first row contains the column headers
rows = r.text.split(',\n')
print('r.text type is ', type(r.text))
print('num_rows', len(rows))

rows[0:3]

# Convert each row from a string to an actual list

# Strip characters from string and split
# on commas
def str_list2elements(s):
    s = s.replace('[','')
    s = s.replace(']','')
    s = s.replace('"','')
    elements = s.split(',')
    return elements

rows2 = [str_list2elements(s) for s in rows]

rows2[0:3]

# Pop the zeroth element of rows2
columns = rows2.pop(0)
# Construct a DataFrame from rows2
acs5_09to15_df = pd.DataFrame(rows2)
acs5_09to15_df.columns = columns
print('acs5_09to15_df (num_rows,num_cols) ', acs5_09to15_df.shape)
acs5_09to15_df.head(5)

acs5_09to15_df['GEOID'] = [s1+s2 for s1,s2 in zip(acs5_09to15_df.state,
                                               acs5_09to15_df.county)]
acs5_09to15_df.head(3)

filename_out = '../output/census_acs5_09to15_population_by_county.csv'
acs5_09to15_df.columns = ['B05003I_003E','STATE_FIPS','COUNTY_FIPS','GEOID']
acs5_09to15_df.to_csv(filename_out, cols=columns, index=False)

# Test loading of file
test_df = pd.read_csv(filename_out,
                      dtype={'B05003I_003E':int,
                             'STATE_FIPS':str,
                             'COUNTY_FIPS':str,
                             'GEOID':str})
print('(num_rows,num_cols) ', test_df.shape)
test_df.head(3)



