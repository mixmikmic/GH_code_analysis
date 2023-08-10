# Make matplotlib static. Use notebook instead of inline to make interactive
get_ipython().run_line_magic('matplotlib', 'inline')

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load part of the dataset to have a look
df = pd.read_csv('kickstarter-projects/ks-projects-201801.csv', nrows=10)

# Let's have a look
df.head()

# We see that there are two date/time columns, we'll tell pandas to parse them when loading the full file.
df = pd.read_csv('kickstarter-projects/ks-projects-201801.csv',
                 parse_dates=['deadline', 'launched'],
                 encoding = "ISO-8859-1")

# Get some general info on the dataset.
# Int is for integer, float for floating point (number with decimal), object for string (text) or mixed types.
df.info()

# Describe is the basic summary stats function for numerical values.
df.describe()

stats_cols = ['backers', 'usd_pledged_real', 'usd_goal_real']
df[stats_cols].describe()

df.columns

df.main_category.unique()

df.state.unique()

df.groupby('main_category')['ID'].count()

df.groupby('state')['ID'].count()

# It would be nicer if those states were capitalized. We can easily do that
df['state'] = df['state'].str.capitalize()
df.groupby('state')['ID'].count()

# We'll also rename some of the columns so that the output is cleaner.
df.columns = ['ID', 'name', 'category', 'Main category', 'currency', 'deadline',
              'goal', 'launched', 'pledged', 'State', 'Backers', 'country',
              'usd pledged', 'Pledged (USD)', 'Goal (USD)']

# Let's add a column that compute the funding percentage
df['Funding %'] = df['pledged'] / df['goal']

stats_cols = ['Backers', 'Pledged (USD)', 'Goal (USD)', 'Funding %']
desc_stats = df[df.State.isin(['Successful', 'Failed'])].groupby('State')[stats_cols].describe()
desc_stats

desc_stats.transpose()

desc_stats.transpose().unstack(level=0)

# This here is what we want!
desc_stats = desc_stats.transpose().unstack(level=0).transpose()
desc_stats

# We still need to rename the columns
desc_stats.columns = ['Count', 'Mean', 'Std. Dev.', 'Min.', '25th Pct.', 'Median', '75th Pct.', 'Max']
desc_stats

# We can export that to Excel
desc_stats.to_excel('DescStats_v1.xlsx')

# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter('DescStats_v2.xlsx', engine='xlsxwriter')
desc_stats.to_excel(writer, sheet_name='Sheet1')

# Get the xlsxwriter objects from the dataframe writer object.
workbook  = writer.book
worksheet = writer.sheets['Sheet1']

# Add some cell formats.
format1 = workbook.add_format({'num_format': '#,##0.00'})
format2 = workbook.add_format({'num_format': '#,##0'})

# Set the column width.
worksheet.set_column('B:B', 18, None)

# Set the column format.
worksheet.set_column('D:D', None, format1)
worksheet.set_column('F:I', None, format1)
worksheet.set_column('C:C', None, format2)

# Set the column width and format.
worksheet.set_column('E:E', 12, format1)
worksheet.set_column('J:J', 14, format1)

# Close the Pandas Excel writer and output the Excel file.
writer.save()

# First, create new columns with the year of each date columns
df['launched_year'] = df['launched'].dt.year
df['deadline_year'] = df['deadline'].dt.year

# Next, compute the duration of the campain, in weeks
df['duration'] = np.round((df['deadline'] - df['launched']).dt.days / 7)

df['duration'].describe()

pd.crosstab(df.launched_year, df.State)

# We can safely drop all events with a 1970 start date.
df = df[df.launched_year != 1970]

pd.crosstab(df.launched_year, df.State)

# Say we only want to compare fails and success, we can create a separate view of the dataset that
# contains only those states
df_result = df[df.State.isin(['Failed', 'Successful'])]
pd.crosstab(df_result.launched_year, df_result.State)

# Let's plot this instead
pd.crosstab(df_result.launched_year, df_result.State).plot()

# Ok, but maybe a bar plot would be better.
pd.crosstab(df_result.launched_year, df_result.State).plot.bar()

# Some additional imports - usually you would put these at the top of the notebook.
from matplotlib.ticker import FuncFormatter


# First change, let's capture the axis and set colors and hatches
# For the list of xkcd colors see https://xkcd.com/color/rgb/
ax = pd.crosstab(df_result.launched_year, df_result.State).plot.bar(
    color=['xkcd:dark gray', 'xkcd:light blue'])

# Set the label formatter of the y-axis to have thousand commas.
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))

# Rotate the year labels
ax.xaxis.set_tick_params(rotation=0)

# Add some labels
ax.set_xlabel('Year', fontsize=13)
ax.set_ylabel('Number of projects', fontsize=13)
# Add a title
ax.set_title('Number of failed and successful projects per year', fontsize=14)

# Set the location of the legend
ax.legend(loc='upper left')

# Tell matplotlib to make it look sharp
plt.tight_layout()

# Export to pdf
plt.savefig('BarPlot_States.pdf')



