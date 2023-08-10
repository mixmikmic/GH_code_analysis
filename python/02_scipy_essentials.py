import numpy as np

# Creates a numpy array of integers 0 to 10,000
a = np.arange(0,10000)

# Sum this array using pythons default method or numpys matrix operations

print 'Built-In Array Sum - uses itteration'
get_ipython().magic('timeit sum(a)')

print '\nNumpy Array Sum - uses numpy matrix operations'
get_ipython().magic('timeit np.sum(a)')

import numpy as np

# Create 1-d arrays from lists or tuples
x = np.array([0,1,2,3])
y = np.array((4,5,6,7.0))

# Create 2-d arrays from nested structures
z = np.array( [[1,1,1],[2,2,2],[3,3,3]] )

print 'Shape: x={} y={} z={}'.format(x.shape, y.shape, z.shape)
print 'N-Dim: x={} y={} z={}'.format(x.ndim, y.ndim, z.ndim)
print 'D-Type: x={} y={} z={}'.format(x.dtype, y.dtype, z.dtype)

import numpy as np

# Default Arrays

# Zeros - creates an array of the given shape and type filled with zeros
a = np.zeros((2,2), dtype=np.int)

# Ones - creates an array of the given shape and type filled with ones
b = np.ones((2,2), dtype=np.int)

# Full - creates an array of the given shape and type filled with a specified value
c = np.full((2,2), 42.0, dtype=np.float)

# Numerical Range Arrays

# ARange - creates a 1d array of values on a specified interval with specified step-length
d = np.arange(10)    # default start is 0, default step is 1
e = np.arange(0,10,2)

# Linspace - creates a 1d array of evenly spaced values in a specified interval with a specified number of values
# very useful for graphing
f = np.linspace(0,10, num=5)

# Randomly Generated Arrays
#
# Numpy also has functions to create arrays with randomly filled values of different sorts
# this can be extremely useful for statistics and baseline comparisons

# Rand - returns an array of shape (n,n) filled with values randomly sampled from a uniform distribution between 0-1
g = np.random.rand(2,2)

# Randn - returns an array of shape (n,n) filled with values randomly sampled from the standard normal distribution
h = np.random.randn(2,2)

# RandInt - returns an array of shape (n,n) filled with random integers from a specified range
i = np.random.randint(4,8, size=(2,2))

# Specific Distributions
#
# Full List - http://docs.scipy.org/doc/numpy/reference/routines.random.html#distributions
#
# For more advanced use numpy can sample from a large array of different distribution types, this can be very 
# useful in statistics, distribution fitting and other purposes. 

# ChiSquare - samples from a chisquare distribution of n degrees of freedom
j = np.random.chisquare(2, size=4)

# Zipf - samples from a zipf distribution with distribution parameter a
k = np.random.zipf(2, size=4)

import numpy as np

# Creates a 4x3 ndarray of strings
a = np.array([['a','b','c'],['d','e','f'],['g','h','i'],['j','k','l']])
print 'Original NDArray'
print a

# Standard syntax for indexing nested/multidimensional sequences
# Row first followed by column
print '\nRow 0, Col 2 (standard syntax)'
print a[0][2]

# Numpy syntax for indexing multidimensional arrays
print '\nRow 0, Col 2 (numpy syntax)'
print a[0,2]

# Examples using slices

# All rows in first column
print '\nColumn 0'
print a[:,0]

# All columns in first row
print '\nRow 0'
print a[0,:]

# Bonus
# Every even row
print '\nEven Rows'
print a[::2, :]

import numpy as np

a = np.array([['a','b','c'],['d','e','f'],['g','h','i'],['j','k','l']])

# Index the first row using a list of indices
b = a[0, [0,2] ]
print b

# Index the first column using a list of indices
c = a[ [0,1], 0]
print c

# Index the 2d ndarray using two lists of indices
# The first list passed acts as the row index and the second is the column index
d = a[[0,1,2],[0,1,2]]
print d

import numpy as np

# Create a 20x20 array of values drawn from a normal distribution
normal = np.random.randn(20,20)

# Defining outliers as any value 2 or more standard deviations from the mean
# Creates a mask array of booleans by applying a conditional to the original ndarray
# uses boolean or to merge both conditions then inverts with ~ (logical not)
mask = ~( (normal < -2) | (normal > 2) )

# Apply the mask array to our original
# removes outliers
no_out = normal[mask]

import numpy as np

# Create a 3x3 array of ints from 0-8
a1 = np.arange(9).reshape((3,3))

# Double all values
print a1 * 2

# Standardize all values
print (a1 - np.mean(a1))/np.std(a1)

# Bonus - It's simple to perform scalar operations on subsets of ndarrays using slices
# remember that slices return views so any operations done to them can be assigned to the original

# Multiply the first row by 10 
a1[0,:] = a1[0,:] * 10
print a1

import numpy as np

a1 = np.ones(10)
a2 = np.arange(10)

print a1 + a2
print a1 * a2

# Bonus: we can even mix elementwise and scalar operations
print (a1 + a2) + 10

import numpy as np

a1 = np.ones(100).reshape(10,10)
a2 = np.arange(10).reshape(1,10)

print a1*a2

import numpy as np

# Lets use numpy to create a simulated distribution of iq scores
# we assume that the mean is 100, the standard deviation is 10 and the distribution is normal

# Create two sample arrays from the normal distribution
# One is very large 10,000 and the other is small 10
big = np.random.randn(10000)
little = np.random.randn(10)

# Using scalar operations scale both arrays to match an iq distribution
big = (big*10) + 100
little = (little*10) + 100

# Calculate some simple statistics for each distribution
print 'Small Sample: mean={} median={} std={}'.format(np.mean(little), np.median(little), np.std(little))
print 'Big Sample: mean={} median={} std={}'.format(np.mean(big), np.median(big), np.std(big))

import pandas as pd

# Series declared from lists will create a 'default' index which is a numeric range from 0-N
default = pd.Series([4,5,6,7,8,9])
print 'Series with Default Index'
print default

# Series declared from lists will create a 'default' index which is a numeric range from 0-N
indexed = pd.Series([4,5,6,7,8,9], index=['a','b','c','d','e','f'])
print '\nSeries with Index '
print indexed

import pandas as pd

# Series can also be created directly from dicts
ex = pd.Series({'a':10, 'b':9, 'c':8, 'd':7})
print ex

print '\nNumeric Indexing'
print ex[1]

print '\nUsing the Index'
print ex['a']

print '\nIndex Arrays with Numeric or Index'
print ex[[0,1]]
print ex[['a','b']]

print '\nSlices Using Numeric or Index'
print ex[0:3]
print ex['a':'c']

import pandas as pd

data = {'age':[12,34,23,52], 'income':[50,78,122,200],'pet':[0,1,1,0]}

# DataFrames can be created from dictionaries of equal length lists/numpy arrays
frame = pd.DataFrame(data)
print 'DF from dict, keys become column headers'
print frame

# Passing a list of columns allows you to specify a subset of columns to include
frame = pd.DataFrame(data, columns=['age','income'])
print '\nDF from dict, columns passed explicitly specify subset to use'
print frame

import pandas as pd

data = {'age':[12,34,23,52], 'income':[50,78,122,200],'pet':[0,1,1,0]}
frame = pd.DataFrame(data)

# Indexing with column names
# note: we cannot use slices to index columns with this syntax
print 'Indexing with ColumnNames'
print frame['pet']
print frame[['age','income']]

# Access columns as attributes
print '\nColumns as Attributes'
print frame.age

# Bonus - standard multidimensional indexing is often used to select rows and columns
print frame['income'][3]

import pandas as pd

data = {'age':[12,34,23,52], 'income':[50,78,122,200],'pet':[0,1,1,0]}
frame = pd.DataFrame(data)

# Select rows 0-3 for column 0
print frame.ix[0:3,0]

# Select all rows for column 'income'
print frame.ix[:,'income']

# Use slices to select a subset of columns
print frame.ix[:,'age':'income']

import pandas as pd

data = {'age':[12,34,23,52], 'income':[50,78,122,200],'pet':[0,1,1,0]}
frame = pd.DataFrame(data)

# Create a mask vector for all people with a pet
mask = frame['pet'] == 1

# Apply the mask vector to the pandas index to get all subjects with pets (preserves index values)
frame[mask]

# An abreviated version
frame[frame['pet']==1]

# Mean income for all people with a pet in one line
print np.mean(frame[frame['pet']==1]['income'])

import pandas as pd

# Import a csv with data for a machiavellianism test
mach = pd.read_csv('./data/mach2/data.csv')
# List the automatically read columns
mach.columns

# Get the mean machiavellianism score and then the sub-means for men and women
mean = np.mean(mach['score'])
mean_m = np.mean(mach[mach['gender']==1]['score'])
mean_f = np.mean(mach[mach['gender']==2]['score'])

print 'Mean Machiavellianism Score = {}, Male Mean = {}, Female Mean={}'.format(mean, mean_m, mean_f)

get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

x = np.arange(0, 10, 0.1)
y = np.sin(x)
plt.plot(x,y)  # pyplot function call

# List of available styles
print plt.style.available

# Set the style to nate silver clone
plt.style.use('fivethirtyeight')
plt.plot(x,y)  # pyplot function call
# Set the style back to default seaborn
plt.style.use('seaborn-darkgrid')

# Bonus - secret desu
# plt.xkcd()
# plt.rcdefaults()  # for after

get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Default figure
x = np.arange(0, 10, 0.1)
plt.plot(x, np.sin(x))

# New figure created in global namespace
plt.figure()
x = np.arange(0, 10, 0.1)
plt.plot(x, np.cos(x))

# New figure and subplot assigned to variables
# a more manageable method
fig1 = plt.figure()
ax1 = fig1.add_subplot(1,1,1)
x = np.arange(0, 10, 0.1)
# Draw multiple things to our plot
ax1.plot(x, np.sin(x),'b')
ax1.plot(x, np.cos(x),'r')

get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Create a figure
fig1 = plt.figure()
# Create subplots for that figure
ax1 = fig1.add_subplot(2,1,1)
ax2 = fig1.add_subplot(2,1,2)

# Draw to our subplots
x = np.arange(0, 10, 0.1)
ax1.plot(x, np.sin(x))
ax2.plot(x, np.cos(x))

# bonus - the same code using subplots()
fig, axes = plt.subplots(2,1)
axes[0].plot(x, np.sin(x), color='r')
axes[1].plot(x, np.cos(x), color='r')

get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

fig1 = plt.figure()
ax1 = fig1.add_subplot(1,1,1)

# Set a title for the subplot
ax1.set_title('Empty Plot', fontsize=18)

# Set an x and y axis lable
ax1.set_xlabel('X Axis', fontsize=14)
ax1.set_ylabel('Y Axis', fontsize=14)

# Set ticks for one of our axes
# ticks must take the form of a list of numeric values
ax1.set_xticks([0,1,2,3,10,20,30])

# Set the labels for our axis ticks
ax1.set_xticklabels([0,1,2,3,'ten','twenty','thirty'])

# Bonus - arange() is an excellent way to explicitly set the range and intervals for your graphs
ax1.set_yticks(np.arange(0,11,1))

get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Create two subplots with line formatting using explicit arguments or abreviated form
fig, axes = plt.subplots(1,2)
x = np.arange(0, 100, 1)
# Draw a different type of noise to each subplot

# Explicit 
axes[0].plot(x, np.random.standard_normal(size=100), color='r', marker='o',linestyle='--')
# Abreviated
axes[1].plot(x, np.random.zipf(2.0,size=100), 'g-x')

get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

fig1 = plt.figure()
ax1 = fig1.add_subplot(1,1,1)

x = np.arange(0, 5, 0.1)
ax1.plot(x, np.sin(x),'--b', label='sine')
ax1.plot(x, np.cos(x),'.r', label='cosine')
ax1.legend()

get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import a csv with data for the npas
# uh-oh looks like some participants included commas in their free response section
# that messed up our columns for the csv so lets exclude them
bad_participants = [27,43,53,68,177,258,666,687,718,725,771,813,                    842,1006,1094,1095,1132,1159,1182,1189,1225,                    1238,1288,1333,1339,1350,1409]
npas = pd.read_csv('./data/NPAS-data/NPAS-data.csv',skiprows=bad_participants)

# Create a single plot
fig, ax = plt.subplots(1,1)

# Get counts of men, women and other genders
men = len(npas['gender'][npas['gender']==1])
women = len(npas['gender'][npas['gender']==2])
other = len(npas['gender'][npas['gender']==3])
# Store in a list
genders = [men,women,other]

# Create a bar graph of the gender numbers for the NPAS
ax.bar(left=[1,2,3],height=genders, width=0.8)

# Set the xlim, xticks, and ticklabels to make it prettier
ax.set_xlim([0.5,4.3])
ax.set_xticks([1.4,2.4,3.4])
ax.set_xticklabels(['Men', 'Women', 'Other'], fontsize=12)

# Add appropriate labels to our graph
ax.set_title('Participant Genders', fontsize=16)
ax.set_ylabel('Frequency', fontsize=14)

# Create a single plot
fig, ax = plt.subplots(1,1)

# Display a histogram of the 'nerdy' values in our data column
# these values represent the total nerdy score from this personality inventory
# the range is 1-7 so we use 7 bins
ax.hist(npas['nerdy'],bins=7)

# Add appropriate labels to our graph
ax.set_title('Nerdy Personality Traits', fontsize=16)
ax.set_xlabel('NPAS Score', fontsize=14)
ax.set_ylabel('Frequency', fontsize=14)

# Create a single plot
fig, ax = plt.subplots(1,1)

# Scatter of x vs y
ax.scatter(npas['testelapse'], npas['surveyelapse'])

# We have some extreme outliers lets set our axes range to something reasonable (10 minutes)
ax.set_xlim([0,10*60])
ax.set_ylim([0,10*60])

# Add appropriate labels to our graph
ax.set_title('Test vs Survey Time', fontsize=16)
ax.set_xlabel('Test Duration (sec)', fontsize=14)
ax.set_ylabel('Survey Duration (sec)', fontsize=14)

# Create a figure with one subplot
fig, ax = plt.subplots(1,1)

# Graph the first five personality items using boxplots
personalityData1 = [npas['TIPI1'], npas['TIPI2'], npas['TIPI3'], npas['TIPI4'], npas['TIPI5']]
ax.boxplot(personalityData1, vert=True)

# Add appropriate labels to our graph
ax.set_title('Ten Item Personality Inventory', fontsize=16)
ax.set_ylabel('Score', fontsize=12)
personalityLabels1 = ['Extraverted','Critical','Dependable','Anxious','Openness']
ax.set_xticklabels(personalityLabels1, fontsize=12, rotation=30)

# Create a figure with one subplot
fig, ax = plt.subplots(1,1)

# Graph the first five personality items using boxplots
personalityData2 = [npas['TIPI6'], npas['TIPI7'], npas['TIPI8'], npas['TIPI9'], npas['TIPI10']]
ax.violinplot(personalityData2, vert=True, showmedians=True)

# Add appropriate labels to our graph
ax.set_title('Ten Item Personality Inventory', fontsize=16)
ax.set_ylabel('Score', fontsize=12)
personalityLabels1 = ['','Reserved','Warm','Disorganized','Calm','Conventional']
ax.set_xticklabels(personalityLabels1, fontsize=12, rotation=30)

import numpy as np
import pandas as pd
import scipy.stats as stats

# Import our csv with data for the npas again
bad_participants = [27,43,53,68,177,258,666,687,718,725,771,813,                    842,1006,1094,1095,1132,1159,1182,1189,1225,                    1238,1288,1333,1339,1350,1409]
npas = pd.read_csv('./data/NPAS-data/NPAS-data.csv',skiprows=bad_participants)

# Ages of all participants as a numpy nd-array
ages = npas['age']

# Measures of Centrality
# Mean
mean = np.mean(ages)
# Median
med = np.mean(ages)
# Mode - returns an array of modes and their counts for each axis in a matrix
mods, counts = stats.mode(ages)
print 'Mean = {} Median = {} Mode = {}'.format(mean, med, mods[0])

# Measures of Variance
# Variance
var = np.var(ages)
# Standard Deviation
std = np.std(ages)
print 'Variance = {} STD = {}'.format(var, std)

# Bonus
# Scipy stats also has a convenience function describe() which returns a number of common descriptive stats for an array
n, minmax, mean, var, skew, kurt = stats.describe(ages)
print 'n={} min/max={} mean={} var={} skew={} kurt={}'.format(n, minmax, mean, var, skew, kurt)

# Test the hypothesis that average familysize is 2 in our sample
t, p = stats.ttest_1samp(npas['familysize'],2)
print 't={:.2f} p={:.8f}'.format(t,p)

# Get The Nerdy scores for men and women
m_nerd = npas['nerdy'][npas['gender']==1]
f_nerd = npas['nerdy'][npas['gender']==2]

# Test the hypothesis that women and men have an equal mean nerdy attributes score
t, p = stats.ttest_ind(m_nerd, f_nerd)
print 't={:.2f} p={:.8f}'.format(t,p)

# Get The 'Reserved' and 'Anxious' scores for all subjects
anx = npas['TIPI4']
res = npas['TIPI6']

# Test the hypothesis that mean reservation and anxiety have the same average values within subjects
t, p = stats.ttest_rel(anx, res)
print 't={:.2f} p={:.8f}'.format(t,p)

# Get the frequencies of male and female participants
men_f = np.sum(npas['gender']==1) # remember booleans are integers
fem_f = np.sum(npas['gender']==2)

# Test the hypothesis that men and women are equally frequent in the sample
chi, p = stats.chisquare([men_f,fem_f])
print 'chi={:.2f} p={:.8f}'.format(chi,p)

# Get the nerdiness scores for each coded response to the sexual orientation question
# 1 = heterosexual, 2=bisexual, 3=homosexual, 4=asexual, 5=other
samples = [ npas['nerdy'][npas['orientation']==x] for x in range(1,5,1)]

# Test the hypothesis that all orientation groups have an equal nerdy attributes score
f, p = stats.f_oneway(*samples)
print 'f={:.2f} p={:.8f}'.format(f,p)

# Get the age and nerdiness scores for all subjects
ages = npas['age']
nerd = npas['nerdy']

# Test the hypothesis that there exists a linear relationship between nerdy personality
# attributes + age and calculate the slope and intercept for that possible relationship
slope, inter, r, p, stderr = stats.linregress(ages, nerd)
print 'nerd = {:.8f} x age + {:.2f}'.format(slope, inter)
print 'r={:.8f}, p={:.8f}, stderr={:.8f}'.format(r,p,stderr)

get_ipython().magic('matplotlib inline')
# Someone claimed to be 30,000 years old so lets manually remove outliers age > 100, time > 20 minutes
npas_clean = npas[npas['age']< 100]
npas_clean = npas_clean[npas_clean['testelapse']< 60*20]

# Make our scatter plot
fig, ax = plt.subplots(1,1)
ax.scatter(npas_clean['age'], npas_clean['testelapse'])

# Add appropriate labels to our graph
ax.set_title('Age vs Test Time', fontsize=16)
ax.set_xlabel('Age (years)', fontsize=14)
ax.set_ylabel('TestTime (sec)', fontsize=14)
# Set limits
ax.set_xlim([0,100])
ax.set_ylim([0,60*20])

# Run our regression model
slope, inter, r, p, stderr = stats.linregress(npas_clean['age'], npas_clean['testelapse'])

# How to graph a regression line
# 
# Grab the x-ticks values as an array
x = ax.get_xticks()
# Create a new array of y values using our slope and intercept
y = x*slope + inter 
# Add to our plot as a solid red line
ax.plot(x,y, color='r', linestyle='-')

