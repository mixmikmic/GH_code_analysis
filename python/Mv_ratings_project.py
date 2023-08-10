# Importing .pyplot
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

# Change the default style
plt.style.use('seaborn-white')

# Generate a figure with a single ax
fig, ax = plt.subplots(figsize = (18,9))

# Generate a hist on the ax
from numpy.random import normal
ax.hist(normal(size = 1000000, scale = 6), bins = 29, range = (-20, 20))

# Hide spines and remove grid
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(False)

# Axes labels
ax.set_ylabel('Number Of Values (Frequency)', fontsize = 18, weight = 'bold')
ax.set_xlabel('Values', fontsize = 18, weight = 'bold')

# Tweak tick parameters
ax.set_yticks([0, 170000])
ax.tick_params(axis = 'both', which = 'both', labelleft = False)

ax.set_xticks([-20,20])
ax.set_xticklabels(['Min', 'Max'], fontsize = 14, weight = 'bold')

# Title
fig.suptitle('  Meet The Normal Distribution', fontsize = 26, weight = 'bold')

# Delimiting areas by generating vertical lines
ax.axvline(-7.5, color = 'black', alpha = 0.4)
ax.axvline(7.5, color = 'black', alpha = 0.4)

# Explanatory text
ax.text(0,125000, 'The tallest bars are here.\n It means most of the values\n are average.', fontsize = 20,
        weight = 'bold', ha ='center')
ax.text(-15,75000, 'Few values are low,\n and fewer extremely low.', fontsize = 20, weight = 'bold', ha ='center')
ax.text(15,75000, 'Few values are high,\n and fewer extremely high.', fontsize = 20, weight = 'bold', ha ='center')

# Increase pad btw the title and graph
plt.tight_layout(pad = 10)

plt.show()

# Change the default style
plt.style.use('seaborn')

# Generate a figure with 4 axes (2 rows by 2 columns)
fig = plt.figure(figsize = (18,10))
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
ax4 = fig.add_subplot(2,2,4)

# Removie tick marks, labels and grids for all axes
for ax in fig.axes:
    ax.tick_params(axis = 'both', which = 'both', labelleft = False, labelbottom = False)
    ax.grid(False)

# Values for hists
bad = [0,0,0,1,1,1,1,1,1,1,1,1,2,2,2,2,2,3,3,3,4,4,5,5,6]
average = [0,1,1,2,2,2,2,2,3,3,3,3,3,3,3,3,3,4,4,4,4,4,5,5,6]
good = [0,1,1,2,2,3,3,3,4,4,4,4,4,5,5,5,5,5,5,5,5,5,5,6]
uniform = [0,0,0,0,1,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,4,5,5,5,5,6,6,6,6]

# Good cluster ax 
ax1.hist(good, bins = 7)

# Average cluster ax
ax2.hist(average, bins = 7)

# Bad cluster ax
ax3.hist(bad, bins = 7)

# Uniform cluster ax
ax4.hist(uniform, bins = 7)
ax4.set_ylim(0,10) # makes the bin's height appear shorter

# Text and arrows
fig.suptitle('Four Possible Distributions Of\n The Ratings For A Single Movie', fontsize = 26, 
             weight = 'bold')

ax1.text(1.5,7,'Cluster In The High\n Ratings Area (Likely)', fontsize = 20, weight = 'bold', ha = 'center')
ax1.arrow(3,8,0.9,0, width = 0.075, color = 'b', alpha = 0.5)

ax2.text(4.9,6, 'Cluster In The\n Average Ratings\n Area (Very Likely)', fontsize = 20, weight = 'bold',
         ha = 'center')
ax2.arrow(3.90,7.85,-0.15,0, width = 0.060, color = 'b', alpha = 0.5)

ax3.text(4.5,6.5, 'Cluster In The Low\n Ratings Area (Likely)', fontsize = 20, weight ='bold', ha = 'center')
ax3.arrow(3.05,7.55,-0.9,0, width = 0.075, color = 'b', alpha = 0.5)

ax4.text(3,6.5, 'No Prominent Clusters\n (Unlikely)', fontsize = 20, weight = 'bold', ha = 'center')

ax3.text(7,-4,'''A Large Enough Sample Of Averaged Ratings For Different Movies Should 
Render A Normal Distribution, Given These Likelihoods''', fontsize = 20, weight = 'bold', ha ='center')

plt.show()

# Import pandas
import pandas as pd

# Read in the dataset
new_ds = pd.read_csv('movie_ratings_16_17.csv')

# Print some info to help the reader understand the structure of the dataset
print(new_ds.shape)
new_ds.head(5) # Check the github link given to understand what the values of each column describe

# Use the FTE style
plt.style.use('fivethirtyeight')

# Generate a figure with 4 axes (2 rows by 2 columns)
fig = plt.figure(figsize = (15,10))
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
ax4 = fig.add_subplot(2,2,4)

# Remove grids for all axes
for ax in fig.axes:
    ax.grid(False)

    
# IMDB
ax1.hist(new_ds.imdb, bins = 20, range = (0,10), align = 'left') # bin range = 0.5
ax1.axvline(3, color = 'black', alpha = 0.4)
ax1.axvline(7, color = 'black', alpha = 0.4)
ax1.set_ylim(0, 60)
ax1.text(5,50, 'The Average\n Area', fontsize = 17, weight = 'bold', ha = 'center')
ax1.set_yticks([0,15,30,45,60])
ax1.set_xticks([0,3,7,10])
ax1.text(5,-7.5, 'IMDB\n (0-10)', fontsize = 14, weight = 'bold', ha = 'center')
ax1.text(-0.86,59.2, 'movies', fontsize = 11)


# Fandango
ax2.hist(new_ds.fandango, bins = 10, range = (0,5), align = 'left') # bin range = 0.5
ax2.axvline(1.5, color = 'black', alpha = 0.4)
ax2.axvline(3.5, color = 'black', alpha = 0.4)
ax2.set_ylim(0,90)
ax2.set_yticks([0,23,45,68,90])
ax2.set_xticks([0,1.5,3.5,5])
ax2.text(2.5,-11, 'Fandango\n (0-5 stars)', fontsize = 14, weight = 'bold', ha = 'center')
ax2.text(-0.56,88.7, 'movies', fontsize = 11)


# Metacritic
ax3.hist(new_ds.metascore, bins = 20, range = (0,100), align = 'left') 
# bin range = 5 (equivalent to 0.5 if normalized to 0-10)
ax3.axvline(30, color = 'black', alpha = 0.4)
ax3.axvline(70, color = 'black', alpha = 0.4)
ax3.set_ylim(0,30)
ax3.set_yticks([0,8,15,23,30])
ax3.set_xticks([0,30,70,100])
ax3.text(50,-3.65, 'Metascore\n (0-100)', fontsize = 14, weight = 'bold', ha = 'center')

# RT
ax4.hist(new_ds.tmeter, bins = 20, range = (0,100), align = 'left') # bin range = 5 
ax4.axvline(30, color = 'black', alpha = 0.4)
ax4.axvline(70, color = 'black', alpha = 0.4)
ax4.set_ylim(0,30)
ax4.set_yticks([0,8,15,23,30])
ax4.set_xticks([0,30,70,100])
ax4.text(50,-3.65, 'Tomatometer\n (0-100%)', fontsize = 14, weight = 'bold', ha = 'center')

# Text
fig.suptitle('Looking For Something Normal', fontsize = 24, weight = 'bold')
ax3.text(-12,-10, 'Author: Alex Olteanu', fontsize = 10, weight = 'bold')
ax4.text(30,-10, 'Source: IMDB, Fandango, Metacritic, Rotten Tomatoes (Websites)', fontsize = 10, weight = 'bold')

plt.show()

# Read in the dataset
ds = pd.read_csv('movie_metadata.csv') # I removed a few duplicates before importing

# Make the dataset easier to be processed
titles = ds[['movie_title', 'imdb_score']]

# Drop rows with missing scores
titles = titles.dropna(subset = ['imdb_score'])
titles.shape

# Keep the FTE style

# Add one ax
fig, ax = plt.subplots()
fig.set_size_inches(10,6.5)
ax.grid(False) # removes the grid

# Generate the hist
ax.hist(titles.imdb_score, bins = 20, range = (0,10), align = 'left') # bin range = 0.5
ax.set_yticks([0,400,800,1200])
ax.set_xticks([0,3,7,10])
ax.axvline(3, color = 'black', alpha = 0.4)
ax.axvline(7, color = 'black', alpha = 0.4)
fig.suptitle('The Distribution For 4917 IMDB Movie Ratings\n Mirrors The One Above',fontsize = 20,
             weight = 'bold')

# Increase the pad btw title and graph; this function works well when dealing with one ax
plt.tight_layout(pad = 6)

# Text
ax.text(-0.85,1180, 'movies', fontsize = 11)
ax.text(5, -65, 'Rating', fontsize = 13.5, weight = 'bold', ha = 'center')
ax.text(-1.4, -350, 'Author: Alex Olteanu', fontsize = 10, weight = 'bold')
ax.text(5.5, -350, 'Source: Dataset Compiled By Kaggle User chuansun76', fontsize = 10, weight = 'bold')

plt.show()

# Importing Hickey's dataset to use for generating a comparative graph
fte_ds = pd.read_csv('fandango_score_comparison.csv')
print(fte_ds.shape)
fte_ds.head(3)

###### Generate a figure with two axes containing comparative line plots ######

### Getting the values ###
'''First, get the values and their frequencies. Then, normalize the frequencies to percent, so you can compare
the two datasets which have different number of datapoints.'''


# Fandango
fdg_vals = new_ds.fandango.value_counts(normalize = True).sort_index() * 100 
# 'normalize' gives the quotient of (frequency of a value/total nr of values); multiply by 100 to get percentages
# Sort all indexes, otherwise the line plots will look chaotic;
fte_fdg = fte_ds.Fandango_Stars.value_counts(normalize = True).sort_index() * 100

# Metascore
ms_vals = new_ds.nr_metascore.value_counts(normalize = True).sort_index() * 100
fte_ms = fte_ds.Metacritic_norm_round.value_counts(normalize = True).sort_index() * 100

# IMDB
imdb_vals = new_ds.nr_imdb.value_counts(normalize = True).sort_index() * 100
fte_imdb = fte_ds.IMDB_norm_round.value_counts(normalize = True).sort_index() * 100

# Tomatometer
tmeter_vals = new_ds.nr_tmeter.value_counts(normalize = True).sort_index() * 100
fte_tmeter = fte_ds.RT_norm_round.value_counts(normalize = True).sort_index() * 100

### The graph ###

# Keep the FTE style

# Generate a figure with 2 axes (1 row by 2 columns)
fig = plt.figure(figsize = (17.5, 7))
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)

# Plot the lines (consider colorblindness when choosing colors)
# FTE dataset
ax1.plot(fte_fdg.index, fte_fdg.values, c = (230/255, 159/255, 0))
ax1.plot(fte_ms.index, fte_ms.values, c = (0, 114/255, 178/255))
ax1.plot(fte_imdb.index, fte_imdb.values, c = (240/255, 228/255, 66/255))
ax1.plot(fte_tmeter.index, fte_tmeter.values, c = (0, 158/255, 115/255))

# New dataset
ax2.plot(fdg_vals.index, fdg_vals.values, c = (230/255, 159/255, 0))
ax2.plot(ms_vals.index, ms_vals.values, c = (0, 114/255, 178/255))
ax2.plot(imdb_vals.index, imdb_vals.values, c = (240/255, 228/255, 66/255))
ax2.plot(tmeter_vals.index, tmeter_vals.values, c = (0, 158/255, 115/255))

# Tweak the axes
# Ax1
# Ticks
ax1.set_yticks([0,10,20,30,40,50])
ax1.set_xticks([0,1,2,3,4,5])
ax1.set_yticklabels(['', '10%', '', '', '40%', '']) # Had to do it this way to keep the gridlines for 0,20, and 30
ax1.set_xticklabels(['0 stars', '1', '2', '3', '4', '5 stars', ])
ax1.tick_params(labelsize = 14) # font size for tick labels
# Legend
ax1.text(3.25,44, 'IMDB', color = (220/255, 208/255, 46/255), fontsize = 16, weight = 'bold')
ax1.text(4.1,39.5, 'Fandango', color = (230/255, 159/255, 0), fontsize = 16, weight = 'bold')
ax1.text(1.05,15.1, 'Metacritic', color = (0, 114/255, 178/255), fontsize = 16, weight = 'bold')
ax1.text(2.7, 4, 'RottenTomatoes', color = (0, 158/255, 115/255), fontsize = 16, weight = 'bold')
ax1.text(-0.3, 33, 'From 146 movies', fontsize = 12, weight = 'bold', rotation = 'vertical')

# Ax2
# Ticksb
ax2.yaxis.tick_right() # moves the y-axis to the right
ax2.set_yticks([0,10,20,30,40,50])
ax2.set_xticks([0,1,2,3,4,5])
ax2.set_yticklabels(['', '10%', '', '', '40%', ''])
ax2.set_xticklabels(['0 stars', '1', '2', '3', '4', '5 stars'])
ax2.tick_params(labelsize = 14)
# Legend
ax2.text(2.8,37, 'IMDB', color = (220/255, 208/255, 46/255), fontsize = 16, weight = 'bold')
ax2.text(3.6,40, 'Fandango', color = (230/255, 159/255, 0), fontsize = 16, weight = 'bold')
ax2.text(1.5,20.2, 'Metacritic', color = (0, 114/255, 178/255), fontsize = 16, weight = 'bold')
ax2.text(-0.1,13.5, 'RottenTomatoes', color = (0, 158/255, 115/255), fontsize = 16, weight = 'bold')
ax2.text(5.4, 33, 'From 214 movies', fontsize = 12, weight = 'bold', rotation = 270)

# Titles & Subtitles
fig.suptitle('Different Movie, Same Story', fontsize = 34, weight = 'bold')
ax1.set_title('October 2015', loc = 'left', weight = 'bold')
ax2.set_title('March 2017', loc = 'right', weight = 'bold')
ax1.text(2.5, -17, 'Walt Hickey From FTE Was \n Puzzled By Fandango\'s Skewed Distribution', fontsize = 21, 
         weight = 'bold', ha = 'center')
ax2.text(2.5, -17, 'We Could Name This Kind Of Distribution \n "The Fandango Distribution"', fontsize = 21, weight = 'bold', ha = 'center')
ax1.text(-0.5, -38, 'Author: Alex Olteanu', fontsize = 11)
ax2.text(0, -38, 'Source: Walt Hickey\'s Dataset; Fandango, Metacritic, IMDB, and Rotten Tomatoes (Websites)', 
         fontsize = 11)
plt.tight_layout(pad = 7) # increases the padding btw the fig title and the axes objects


plt.show()
fig.savefig('small_multiple_fdg.jpg')

# Keep the FTE style

# One ax
fig, ax = plt.subplots()
fig.set_size_inches(12,7.5)
ax.grid(False) # removes all gridlines

# Plot the bar graphs
ax.bar(fte_fdg.index, fte_fdg.values, width = 0.2, color = (213/255, 94/255, 0)) # red; 2015
ax.bar(fdg_vals.index, fdg_vals.values, width = 0.2, align = 'edge',color = (0/255, 114/255, 178/255)) # blue;2017

# Tweak the graph
from numpy import arange
ax.set_xticks(arange(0,5.1,0.5))
ax.set_xticklabels(('0', '', '', '', '', '2.5', '3.0', '3.5', '4.0', '4.5', '5.0 stars'))
ax.set_yticks(range(0, 70, 10)) 
ax.set_yticklabels(('', '10%', '', '', '40%', '', '', ''), weight = 'bold', fontsize = 12)
ax.yaxis.tick_right()

# Text
ax.text(0, 55, '2015', fontsize = 24, weight = 1000, color = (213/255, 94/255, 0)) # year
ax.text(2, 55, '2017', fontsize = 24, weight = 1000, color = (0/255, 114/255, 178/255)) # year
# Means and modes computed before
ax.text(0.12,50, '4.1', fontsize = 18, weight = 1000, color = (213/255, 94/255, 0)) # mean 2015
ax.text(2.135,50, '3.9', fontsize = 18, weight = 1000, color = (0/255, 114/255, 178/255)) # mean 2017
ax.text(1,50, 'Mean', fontsize = 22, weight = 'bold', color = 'black') # mean 
ax.text(1,45, 'Mode', fontsize = 22, weight = 'bold', color = 'black') # mode
ax.text(0.12,45, '4.5', fontsize = 18, weight = 1000, color = (213/255, 94/255, 0)) # mode 2015
ax.text(2.135,45, '4.0', fontsize = 18, weight = 1000, color = (0/255, 114/255, 178/255)) # mode 2017
ax.text(0, -9, 'Author: Alex Olteanu', fontsize = 10)
ax.text(3.8, -9, 'Source: FiveThirtyEight\'s dataset, fandango.com', fontsize = 10)
ax.annotate("The barren area of \n movies not worth seeing", xy=(0.2, 0.2), xycoords='data', xytext=(1.2, 18), 
            textcoords='data', size=20, va="center", ha="center", bbox=dict(boxstyle="circle", fc="w", alpha = 0.5))
                  
# Title
fig.suptitle('Magnifying On Fandango', fontsize = 28, weight = 'bold')
plt.tight_layout(pad = 6.5) 

plt.show()

# Correlation Values
print(new_ds.corr().loc['fandango'][['metascore', 'imdb']])

### Generating the top-image ###

# Keep the fivethirtyeight style (remains from above)

# Generate a figure with a single ax
fig, ax = plt.subplots(figsize = (18,9))

# Generate a hist on the ax; normal was imported earlier from numpy.random
ax.hist(normal(size = 1000000, scale = 6), bins = 29, range = (-20, 20))

# Hide grid and tweak tick parameters
ax.grid(False)

ax.set_yticks([0, 170000])
ax.tick_params(axis = 'both', which = 'both', labelleft = False)

ax.set_xticks([-20,20])
ax.set_xticklabels(['Min', 'Max'])

# Explanatory text
ax.text(0, -7000, 'Rating Value', weight = 'bold', fontsize = 14, ha = 'center')
ax.text(0,115000, 'Most Of The Movies\n Are Average', fontsize = 18, weight = 'bold', ha ='center')
ax.text(-15,115000, 'Few Movies Are\n Terrible', fontsize = 18, weight = 'bold', ha ='center')
ax.text(15,115000, 'Few Movies Are\n Outstanding', fontsize = 18, weight = 'bold', ha ='center')
fig.suptitle('Movie Ratings Should Reflect\n Movie Quality', fontsize = 22, weight = 'bold')

# Delimitating areas
ax.axvline(-7.5, color = 'black', alpha = 0.4)
ax.axvline(7.5, color = 'black', alpha = 0.4)

# Increase the pad btw title and graph
plt.tight_layout(pad = 8)

plt.show()

