import pandas as pd

# Creating pandas DataFrame
url = 'http://bit.ly/uforeports'
ufo = pd.read_csv(url)

ufo.head()

ufo.shape

# Removing column
# axis=0 row axis
# axis=1 column axis
# inplace=True to effect change
ufo.drop('Colors Reported', axis=1, inplace=True)

ufo.head()

# Removing column
list_drop = ['City', 'State']
ufo.drop(list_drop, axis=1, inplace=True)

ufo.head()

# Removing rows 0 and 1
# axis=0 is the default, so technically, you can leave this out
rows = [0, 1]
ufo.drop(rows, axis=0, inplace=True)

ufo.head()

