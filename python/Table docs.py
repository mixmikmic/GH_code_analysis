import matplotlib
from datascience import Table
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('fivethirtyeight')

Table.read_table('testdata.manual.2009.06.14.csv') #must be stored in same jupyter folder as active notebook

twitter_data = Table.read_table('testdata.manual.2009.06.14.csv') #to save the new table, name it

NOAA_data = Table.read_table('ftp://aftp.cmdl.noaa.gov/products/trends/co2/co2_mm_mlo.txt', comment = "#", 
                                 na_values = [-99.99, -1], delim_whitespace = True,
                   names = ["year", "month", "decimal_date", "average", "interpolated", "trend", "days"])
NOAA_data

NOAA_data['average']                             #both the 'query' column
#NOAA_data_url[3]
print(NOAA_data.column_index('average'))         #returns 3
NOAA_data['average'] is NOAA_data[3]         #each method refers to the same column so they are the same

NOAA_data.column('average')                      #same exact results as above, just different ways to access column values
#NOAA_data_url.column(3)
print(NOAA_data.column_index('average'))               
NOAA_data.column('average')  is NOAA_data.column(3)

twitter_data.rows 
twitter_data.row(5)     #shows what data is contained in row #5, the tweet by GeorgeVHulme
#twitter_data.row(600)  #index must be in range of table row nums

twitter_data.columns    #shows what data are contained in column 5, labeled 'text'
twitter_data.column(5)

twitter_data = twitter_data.relabeled(['0','1','2','3','4','5'], ['polarity','id','date','query','user','text']) 
#labels are always strings

twitter_data = twitter_data.with_column('source', 'GitHub')
twitter_data
#Since only one value was input, every row will assume that value for the new column

twitter_data = twitter_data.with_columns([['source', 'twitter'], ['row_num', np.arange(498)]])
twitter_data
#Note how every value in 'source' was changed and how you can also add unique values in the form of an array.

twitter_data.with_rows([
    [2,0,'Tue Jun 7 12:02:35 UTC 2016', 'data8', 'deneros_dog', 'Cant wait to start my first project', 'Twitter', 499],
    [0,0,'Tue Jun 7 12:03:45 UTC 2016', 'data100', 'deneros_dog', 'Will we learn Apache Spark?', 'GitHub', 500]])

#Note how each array was formatted to match the previous data types identically

twitter_data.drop(['source', 'id']) #without 'source' or 'id'
#twitter_data.drop('row_num') without row_num

odd_rows = twitter_data.take(np.arange(1,499,2)) #only only odd-numbered rows
even_rows = twitter_data.exclude(np.arange(1,499,2)) #everything but odd-numbered rows 

odd_rows.append(even_rows)  #all the even numbers are now on the bottom of the table

odd_rows.append_column('source', 'odd_twitter') 
#notice, the table is not returned, but only mutated with the above call that replaced the column

twitter_data.select(['query', 'polarity', 'text'])

twitter_data.move_to_start('row_num')
#Uncomment and delete this message to see that the twitter_data table object has changed
#twitter_data

twitter_data.move_to_end('date')

twitter_data.sort('polarity', descending = True, distinct = False) 
#twitter_data.sort('polarity', descending = True, distinct = True)  #uncomment this to see when distinct = True

twitter_data.group('polarity')   #when no collect function is specified, the counts are returned

twitter_data.group('polarity', collect = np.mean)  
#most functions operate best on numerical values, and return nothing if operating on strings

twitter_data.group('user').sort('count', descending = True) 
# Since most methods return a mutated instance of the original table, you can often use multiple methods in 
# a single statement to return more complex or specific results.
# This shows that some users posted 2-3 tweets because it grouped the table by username counting the number of tweets 
# associated with that user, then, the table is sorted so that the highest counts are at the top.

## Simple Pivot Table

twitter_data.pivot('polarity', 'user')
#each polarity value represents a column_label and each user that ever had a tweet with a polarity is the row and 
#the values in the table show that each user had one tweet per polarity since most users only posted one tweet

## Slightly more complex pivot table

NOAA_data.pivot('month','year', values = 'average', collect = np.mean, zero = 0)

#each unique month is the column_labels, and each year marks the rows so its easier to see trends among months or years
#the values show what was in the original table where that year and month's measurement coincided with an average value

twitter_data.stack('polarity')

twitter_data.stack('polarity', labels = 'text')

twitter_data.stats(ops = [min, max, np.mean, np.std])

twitter_data.percentile(0)    # same as min
twitter_data.percentile(100)  # same as max
twitter_data.percentile(50)   # same as median

twitter_data.sample()         # If called with no arguments, it randomly samples rows a num_rows number of times

twitter_data.sample(10, with_replacement = False) # otherwise, it just samples k-times

sample, unsampled_rows = twitter_data.split(10)     # you can create two values by separating them by a comma
sample
#unsampled_rows                                     # Uncomment this to view what rest looks like

## Binning
# Binning values helps you understand multiple ranges, so this shows that most twitter ID's are 0-1000, or 1999-2000 
# interestingly not in between.
twitter_data.bin(select = 'id', bins = 10, range = (0,9999), density = False)

twitter_data.bin(select = 'polarity', bins = 4)
#As you can see bins operate by finding values that are less than or equal to the specified bin, so be careful.

twitter_data.bin(select = 'polarity', bins = 4, density = True)
#Now by setting density = True, each count is compared relative to a whole

twitter_data.bin(select = 'polarity', bins = [0,2,4], density = True)
#But sometimes you can get wrong data by the terms you specify so be careful

## Your First Plot!

NOAA_data.plot('year','average')

## This plot is slightly more complex, but as you can see it just plots average and trend relative to year by decimal date.

NOAA_data.plot('decimal_date', ['average', 'trend'], overlay = True )



