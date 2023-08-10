get_ipython().magic('pylab')
get_ipython().magic('matplotlib inline')
import pandas
from astropy.io import ascii
from astropy.table import Table

# Read a simple ascii text file into Pandas
df1 = pandas.read_table("zody_and_ism.txt", delim_whitespace=True, comment="#")
df1.head()

ax = df1.plot(x='wave', y='zody', kind='line', color='blue')
df1.plot(x='wave', y='ISM', ax=ax, color='red', label="Galaxy") #put on same axis, w ax=ax
plt.yscale('log')



