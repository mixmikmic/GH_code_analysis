# Run this if you need to install the requirements (Prefixing with ! runs it as a shell command).
# You do not need to run this in Azure, so it is commented out. But you should run it if you have
# Jupyter running on your own computer.
#!pip install -r requirements.txt

# Import the libraries we're using
import pandas as pd

# Load the waterfowl data into a dataframe
url = "https://data.delaware.gov/api/views/bxyv-7mgn/rows.csv?accessType=DOWNLOAD"
waterfowl_df = pd.read_csv(url)

# Time to explore the data.

# First, check the number of rows and columns against the data on the portal!
waterfowl_df.shape

# As of February 10, 2018, we have 1942 rows and 36 columns. If you do this later, you may have more than 1942 rows.
# Check your output against the open data portal!

# Now look at the column types
waterfowl_df.dtypes

# Weird that Mergansers is a float. That's something you would explore if you wanted to chart that bird.

# Look at the first few rows of data. Compare to the data on the data portal!
waterfowl_df.head()

# Look at the last few rows:



waterfowl_df.tail()

# Pandas has a handy describe() function
# count tells the number of values that column has (some columns can be NaN (Not a Number))
# Look at the mean, median (50%) and max
waterfowl_df.describe()

# Let's sum all the columns to select what birds we want
waterfowl_df.sum()

# Let's look at the number of rows for each year
waterfowl_df.groupby('Year').count()

# ***** This cell requires you to fill something in! *****

# Copy the previous command, and paste it below.
# Before running, edit it to get the sum by year.
# This is an example of "method chaining," and it's part of the power of Pandas!

# Going back to the .count() example, most years have 44 rows, but there are discrepencies!

# Let's look at the counts of January in each year
waterfowl_df_january = waterfowl_df[waterfowl_df['Month']=='January']
waterfowl_df_january.groupby('Year').count()

# In 2010 and before the number of observations in January was 11.
# Since 2011 it has 14. Let's look at 2010 and 2011 (ignore the warning if you see one)
waterfowl_df_january[waterfowl_df['Year'].isin([2010, 2011])]

# ***** This cell requires you to fill something in! *****

# 2011 has three observations with the timeperiod set to 'Late'
# Remove observations where the timeperiod = 'Late' 
# (in otherwords, keep the observations where the time period does not equal (!=) 'Late')
waterfowl_df_january_sub = waterfowl_df_january[waterfowl_df_january['Time Period']!='REPLACE_ME']

# Finish the previous line, then we'll check the counts again
waterfowl_df_january_sub.groupby('Year').count()

# We have 11 observations for each year! So far so good.
# Note: You'll see that 'Time Period' is 0 through 2010. That is because it is not set until 2010, and the
# .count() method only counts values that are set!

# In 2011 Delaware started counting some areas multiple times, but we only want to look at a single observation
# for each area ('Unit' in the data) for January of each year.

# Let's check unit counts
waterfowl_df_january_sub.groupby('Unit').count()



