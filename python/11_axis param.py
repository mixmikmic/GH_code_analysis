import pandas as pd

url = 'http://bit.ly/drinksbycountry'
drinks = pd.read_csv(url)

drinks.head()

# let's remove "continent" column
# axis=1 drops the column
drinks.drop('continent', axis=1).head()

# drops second row
# axis=0 drops the row
drinks.drop(2, axis=0).head()

# drops multiple rows
drop_rows = [0, 1]
drinks.drop(drop_rows, axis=0).head()

# mean of each numeric column
drinks.mean()

# it is the same as the following command as axis=0 is the default
# drinks.mean(axis=0)
# it instructs pandas to move vertically

# mean of each row
# drinks.mean(axis='columns)
drinks.mean(axis=1).head()

drinks.mean(axis='index').head()
# this is the same as 
# drinks.mean(axis=0).head()

