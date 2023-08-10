#import pandas and numpy libraries
import pandas as pd
import numpy as np
import sys

#read the .txt file into a pandas dataframe
water_in_1990 = pd.read_table(r"water_ca1990co.txt")

#now to glean some info about the data frame (rows, cols, head, etc.)
print(water_in_1990.head())
dimensions = water_in_1990.shape
rows = dimensions[0]
cols = dimensions[1]
print("There are %d rows and %d columns" % (rows, cols))
print(water_in_1990.info())

#pandas .loc[] method allows you to select a row from your dataframe, using
#the row index number. (0-len(rows))
row19 = water_in_1990.loc[18]
print(row19)

print(row19)
""" with 163 columns in this dataset, it's a bit unweildy, I'll need to
    do some work to tease out the data I'm looking for. Water usage by
    county, that is. Let's see what all the column names are"""
#Putting all the column names into a list
col_names = water_in_1990.columns.tolist()
print(col_names)



