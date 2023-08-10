import pandas as pd

# Generate fake datasets with numpy
import numpy as np

# Generate large fake dataset
#   Create random integers between 0 and 100 and put it in a 1000x50 array
data = np.random.randint(0, 100, [1000, 50])
data

# Create a dataframe from the numpy array
df = pd.DataFrame(data)
df.head()

# Options
#  - max_rows -> Number of rows shown
#  - max_columns -> Number of columns shown
print pd.options.display.max_rows
print pd.options.display.max_columns

# Set new values
pd.options.display.max_rows = 4
df

# Reset to 60
pd.options.display.max_rows = 60

df

pd.get_option("max_rows")

pd.set_option("max_rows", 4)
df

pd.reset_option('max_rows')
df

pd.describe_option('max_rows')

# randn -> Dataset of statistical diviations
df = pd.DataFrame(np.random.randn(5,5))
df

# Only show 2 decimal points, without changing underlying data
pd.get_option('precision')

pd.set_option('precision', 2)
df

# If reset, the original numbers remain
pd.reset_option('precision')
df

