import pandas as pd

link = 'http://bit.ly/uforeports'
ufo = pd.read_csv(link)

ufo.head()

# to get 3 random rows
# each time you run this, you would have 3 different rows
ufo.sample(n=3)

# you can use random_state for reproducibility
ufo.sample(n=3, random_state=2)

# fraction of rows
# here you get 75% of the rows
ufo.sample(frac=0.75, random_state=99)

train = ufo.sample(frac=0.75, random_state=99)

# you can't simply split 0.75 and 0.25 without overlapping
# this code tries to find that train = 75% and test = 25%
test = ufo.loc[~ufo.index.isin(train.index), :]

