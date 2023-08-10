# Tutorial for basic filtering, sorting, dealing w data in Pandas.  jrigby, Oct 2016

get_ipython().magic('pylab inline')
import pandas
pandas.set_option("display.max_rows",999)
from astropy.io import ascii
#from astropy.table import Table
pandas.options.display.max_rows = 999   
pandas.options.display.max_columns = 999 # Show me everything!  Scrollbars below will now cover the whole dataframe

# Read a machine-readable table from an ApJ paper into Pandas, via astropy.Table
infile = "http://iopscience.iop.org/0067-0049/226/1/5/suppdata/apjsaa2f41t6_mrt.txt"
temp_table = ascii.read(infile) # this automagically gets the format right.  Good going, Astropy Tables!
df = temp_table.to_pandas()  # Convert from astropy Table to Pandas Data Frame.  Needs astropy 1.2
df.head(2) # show the first few lines of data frame.  The "2" is optional

df.tail(2) # Show the last few lines.  The "2" is optional

df.mean()  # Mean of every column

df.describe()
# Quick statistics for each column

df['O2-3727'].max(), df['O2-3727'].min()  #  What's the brightest and weakest [O II] 3727?

#By default, the index is the row number.  Let's set the index to the ID, which is more meaningful.
df.set_index('ID', inplace=True)
df.head(2)

#df['O2-3727']  # get one column.  commented out b/c the output is ugly
#df.loc['MK02']  # get one row
df.ix['MK02']  # Get one row

# Now that we've re-indexed to the ID numbers, it's easier to find data.
df.drop('MK01', inplace=True) # Delete a row.  I hate you, MK01, you're gone!

# Take the ratio of two lines, and shove it to a new column
df['newrat']= df['Ne3-3869']/df['O2-3727']
df['newrat'].median() # Median value of the ratio

# Quick plotting directly in pandas
df.plot(x='O3-5007', y='O3-4363', kind='scatter')

# More complicated plots, from matplotlib
plt.scatter(df['O3-5007'], df['O3-4363']/df['O3-5007'])
plt.xlabel("[O III] 5007 flux")
plt.ylabel("[O III] 4363/5007 flux ratio")

# Save this pandas to a pickle
df.to_pickle('dummy_example_saved_to_pickle.p') # save as a pickle
df.to_csv('dummy_example_saved.csv')            # save as a comma separated variable
# One thing that bugs me about Pandas is that pandas.read_table() and pandas.read_csv() strip 
# the metadata from the header.  I wonder if there's a way to store it a block, and put it 
# back on saving?

# Make a new dataframe, with the strongest O3-4363 emitters
mask = df['O3-4363'].gt(2.* df['O3-4363'].median())  # mask is a boolean area where the flux is >2x median
subset = df[mask].copy()  # It's safer to copy, see http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy)
subset.describe()



