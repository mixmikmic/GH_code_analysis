get_ipython().magic('matplotlib inline')

#pd, np, and plt are typical aliases used for pandas, numpy, and matplotlib respectively
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#the from statement here indicates that we're going to import specific classes from the module, namely datetime and timedelta 
from datetime import datetime, timedelta

#read in csv to dataframe , 'names' refers to file names, and dtype or datatype is the type of data we're expecting from the file
tlog = pd.read_csv('updated_tlog.csv', names=['basket_id','timestamp','upc','quantity','total_cost'], dtype={'upc':np.str})
#returns the first n-rows of the dataframe
tlog.head()

tlog.head()
#returns the first few elements of the unsorted data frame, not sorted in-place by default

tlog.head(10)
#returns the first ten elements of the sliced dataframe

tlog[['timestamp','total_cost']]
#returns slices of the existing dataframe, returning only the given attributes 

tlog[5:20]

tlog.sort('total_cost', ascending=False).head()
#return the first few elements sorted in descending order by total cost, note dataframe is sorted and then head is selected

tlog.sort('total_cost', ascending=False, inplace=True)
#when in place is set to true, a new instance of the dataframe object is not created and the original is mutated
tlog.head()

tlog[tlog.upc=='2927400622']

tlog[tlog.total_cost > 5.0].head()
#select all of those attributes whose total cost is greater than 5.0 and return head

tlog[(tlog.total_cost>5.0) & (tlog.total_cost<80.0)].head()
#select all of those elements whose total cost is greater than 5.0 and less than 80.0 and return the head

tlog['total_cost'] # same as tlog.total_cost
tlog['pennies'] = tlog.total_cost*100  # not equal to tlog.pennies = tlog.total_cost * 100  because tlog.X is a copy and tlog['X'] is a reference
tlog.head()

tlog['timestamp'] = pd.to_datetime(tlog.timestamp)
#take this column (timestamp) and transform it from a string to a datetime object

tlog[tlog.timestamp > datetime(2015,1,9,11)].sort('timestamp').head()
#return all elements whose timestamp is later than 2019-01-09, with those dates furthest in the future coming first

tlog.total_cost.sum()
#sum over dataframe['total_cost'] contents, returns scalar

#create new attribute hour, derived from applying an anonymous function that extracts an element's hour to each element in the dataframe
tlog['hour'] = tlog.timestamp.apply(lambda x: x.hour)
tlog.head()

#group by here is essentially identical to its use in SQL, abstract over dataframes element and return counts by different basket ids
basket_counts = tlog.groupby('basket_id').count().upc
print(basket_counts)

#plot a histogram 
plt.hist(basket_counts, bins=range(15))

sales_by_product = tlog.groupby('upc').sum()
#group by upc code and sum all other attributes (since no attribute or group of attribute was specified)

sales_by_product.sort('total_cost', ascending=False, inplace=True)
#given the grouped and summed data, sort by total cost in descending order; sort in place
sales_by_product.head()

products = pd.read_csv('updated_products.csv', names=['id','upc','description'], dtype={'upc':np.str})
products.head()

tlog['upc'] = tlog.upc.apply(lambda s: s[4:])
#apply anonymous function that extracts first four characters of upc string to each element of dataframe

merged = pd.merge(sales_by_product,products, left_index=True, right_on='upc', how='left')
merged.head(20)



