import pandas as pd

url = 'http://bit.ly/uforeports'
ufo = pd.read_csv(url)

ufo.head()

# To check out only the columns
# It will output a list of columns
ufo.columns

# inplace=True to affect DataFrame
ufo.rename(columns = {'Colors Reported': 'Colors_Reported', 'Shape Reported': 'Shape_Reported'}, inplace=True)

ufo.columns

ufo_cols = ['city', 'colors reported', 'shape reported', 'state', 'time']

ufo.columns = ufo_cols

ufo.head()

url = 'http://bit.ly/uforeports'
ufo = pd.read_csv(url, names=ufo_cols, header=0)

ufo.head()

ufo.columns = ufo.columns.str.replace(' ', '_')

ufo.head()

