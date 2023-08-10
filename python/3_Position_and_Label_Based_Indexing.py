# loading libraries and reading the data
import numpy as np
import pandas as pd

market_df = pd.read_csv("../Data/global_sales_data/market_fact.csv")
market_df.head()

help(pd.DataFrame.iloc)

# Selecting a single element
# Note that 2, 4 corresponds to the third row and fifth column (Sales)
market_df.iloc[2, 4]

# Selecting a single row, and all columns
# Select the 6th row, with label (and index) = 5
market_df.iloc[5]

# The above is equivalent to this
# The ":" indicates "all rows/columns"
market_df.iloc[5, :]

# equivalent to market_df.iloc[5, ]

# Select multiple rows using a list of indices
market_df.iloc[[3, 7, 8]]

# Equivalently, you can use:
market_df.iloc[[3, 7, 8], :]

# same as market_df.iloc[[3, 7, 8], ]

# Selecting rows using a range of integer indices
# Notice that 4 is included, 8 is not
market_df.iloc[4:8]

# or equivalently
market_df.iloc[4:8, :]

# or market_df.iloc[4:8, ]

# Selecting a single column
# Notice that the column index starts at 0, and 2 represents the third column (Cust_id)
market_df.iloc[:, 2]

# Selecting multiple columns
market_df.iloc[:, 3:8]

# Selecting multiple rows and columns
market_df.iloc[3:6, 2:5]

# Using booleans
# This selects the rows corresponding to True
market_df.iloc[[True, True, False, True, True, False, True]]

help(pd.DataFrame.loc)

# Selecting a single element
# Select row label = 2 and column label = 'Sales
market_df.loc[2, 'Sales']

# Selecting a single row using a single label
# df.loc reads 5 as a label, not index
market_df.loc[5]

# or equivalently
market_df.loc[5, :]

# or market_df.loc[5, ]

# Select multiple rows using a list of row labels
market_df.loc[[3, 7, 8]]

# Or equivalently
market_df.loc[[3, 7, 8], :]

# Selecting rows using a range of labels
# Notice that with df.loc, both 4 and 8 are included, unlike with df.iloc
# This is an important difference between iloc and loc
market_df.loc[4:8]

# Or equivalently
market_df.loc[4:8, ]

# Or equivalently
market_df.loc[4:8, :]

# The use of label based indexing will be more clear when we have custom row indices
# Let's change the indices to Ord_id
market_df.set_index('Ord_id', inplace = True)
market_df.head()

# Select Ord_id = Ord_5406 and some columns
market_df.loc['Ord_5406', ['Sales', 'Profit', 'Cust_id']]

# Select multiple orders using labels, and some columns
market_df.loc[['Ord_5406', 'Ord_5446', 'Ord_5485'], 'Sales':'Profit']

# Using booleans
# This selects the rows corresponding to True
market_df.loc[[True, True, False, True, True, False, True]]

