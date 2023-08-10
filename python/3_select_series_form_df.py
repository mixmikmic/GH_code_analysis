import pandas as pd

# The csv file is separated by commas
url = 'http://bit.ly/uforeports'

# method 1: read_table
ufo = pd.read_table(url, sep=',')

# method 2: read_csv
# this is a short-cut here using read_csv because it uses comma as the default separator
ufo = pd.read_csv(url)
ufo.head()

# Method 1: Selecting City series (this will always work)
ufo['City']

# Method 2: Selecting City series
ufo.City

# 'City' is case-sensitive, you cannot use 'city'

# confirm type
type(ufo['City'])
type(ufo.City)

ufo['Colors Reported']

# example of concatenating strings
'ab' + 'cd'

# created a new column called "Location" with a concatenation of "City" and "State"
ufo['Location'] = ufo.City + ', ' + ufo.State
ufo.head()

