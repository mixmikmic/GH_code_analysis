import pandas as pd
from pandas_datareader import data

companies = ["MSFT", "GOOG", "AAPL", "YHOO", "AMZN"]
p = data.DataReader(companies, data_source = 'google', start = '2010-01-01', end='2016-12-31')
p 
# This returns a panel object

p.items

p.major_axis

p.minor_axis

p.axes

# ndim - Number of dimensions (panel is 3)
p.ndim

# dtypes - Series of index labels of each data frame (the .items attribute)
p.dtypes

# shape - Tuple of the measurements of the panel
# (number of dataframe, number of rows, number of columns)
p.shape

# size - Total number of values in the panel
p.size

# Can prove this value by multiplying values in .shape tuple
p.shape[0] * p.shape[1] * p.shape[2]

# values - Array of values
#  Nested structure
p.values

# Get available values on the items axis
p.items

# Pull the Open dataframe
p['Open']

# Pull the Volume dataframe
p['Volume']

p.Volume

# Extract using index labels (loc)
# Extracting close dataframe
p.loc["Close"]

# Get Close values for a specific date
p.loc["Close", '2010-01-04']

# Get close value for MSFT on 2010-01-04
p.loc["Close", '2010-01-04', "MSFT"]

# Extract using index position (iloc)
p.items

# Get close (3rd index)
p.iloc[3]

# Get 2010-01-04 (0) in close dataframe 
p.iloc[3, 0]

# Get MSFT (3rd index) close value on date above 
p.iloc[3, 0, 3]

# Mix/match position and labels using .ix
p.ix["Close", 0, "GOOG"]

# Convert panel to multi-index dataframe
df = p.to_frame()
df.head()

# Convert multi-index dataframe to a panel
p2 = df.to_panel()
p2

# Have the major_axis *attribute* - Returns values that are the row labels in the panel
p.major_axis

# major_xs() is a *method* - Returns a dataframe where the minor access is the Index and the Items axis are the columns

#  Get data for 2010-01-04
p.major_xs("2010-01-04")

p.minor_axis

# Get all of AAPL's stock values
p.minor_xs('AAPL')

p.axes

# Swap Stock information (Open/Close/High/Low/Volume) with the Company information
#   This is the items axis and the minor axis
# Feed it a list of indexes the new panel will have in the order we want 

# Items - Index position of 0
# Major - Index position of 1
# Minor - Index position of 2
p2 = p.transpose(2, 1 ,0)
p2

# Now can get company information like this, instead of using the .minor_xs() method
p2['AAPL']

# Using major_xs() and minor_xs() still works
# Get information for all on a specific day
p2.major_xs('2010-01-04')

# Get all volumes
p2.minor_xs("Volume")

# Swap Stock information (Open/Close/High/Low/Volume) with the Company information
p3 = p.swapaxes("items", "minor")
p3

p3['MSFT']



