import pandas as pd
import os
import numpy as np

import matplotlib 

import matplotlib.pyplot as plt

# For statistical tests
import scipy.stats as st

get_ipython().magic('matplotlib inline')

from sklearn.cluster import KMeans

#from sklearn.cross_validation import train_test_split

data = pd.read_csv('data.csv')
data = data.dropna()
num_customers = data.shape[0]
print num_customers

data.head()

data.columns.values

list_purchase = []
for k in range(1,6):
    text_ = 'avg_purchase_shop_' + str(k)
    aux_ = data[text_].sum()/1000.
    list_purchase.append(aux_)
dfAux = pd.DataFrame()
dfAux['Store Number'] = [1,2,3,4,5]
dfAux['Avg. Purchase in Shop'] = list_purchase

dfAux.plot(x='Store Number',y='Avg. Purchase in Shop',kind='bar',
           title='Average Purchase per Store',legend=False,grid=True)
plt.xlabel('Shop Number')
plt.ylabel('Avg. Purch. in Store (Thousands of Euros)')
plt.show()

list_purchase = []
for k in range(1,6):
    aux_ = sum(data['shops_used']== k)
    perc_ = float(aux_/1000.)
    list_purchase.append(perc_)
dfAux = pd.DataFrame()
dfAux['Number of Shops Used'] = [1,2,3,4,5]
dfAux['Number of Customers'] = list_purchase

dfAux.plot(x='Number of Shops Used',y='Number of Customers',kind='bar',
           title='Distribution of Customers According to Number of Stores Used',
           legend=False,grid=True)
plt.xlabel('Number of shops used ')
plt.ylabel('Number of customers (thousands)')
plt.show()

list_purchase = []
labels_ = ["One store","Two stores","Three stores","Four stores","Five stores"]
colors = ['lightblue','red','yellow','magenta','green']
explode = (0.05,0.05,0.05,0.05,0.05)
for k in range(1,6):
    aux_ = sum(data['shops_used']== k)
    perc_ = float(aux_)
    list_purchase.append(perc_)
list_purchase = np.array(list_purchase) 
list_purchase = list_purchase*100./sum(list_purchase)

    
matplotlib.rcParams.update({'font.size': 12})
plt.pie(list_purchase,explode=explode,colors=colors,labels=labels_,
        autopct='%1.1f%%',shadow=True,startangle=10)

plt.axis('equal')

plt.title('Customers according to number of stores used')

plt.show()

list_purchase = []
for k in range(1,6):
    text_ = 'avg_price_shop_' + str(k)
    aux_ = sum(data[text_] <> 0)
    perc_ = float(sum(data[text_] <> 0)/1000.)
    list_purchase.append(perc_)
dfAux = pd.DataFrame()
dfAux['Shop Number'] = [1,2,3,4,5]
dfAux['Number of Customers'] = list_purchase

dfAux.plot(x='Shop Number',y='Number of Customers',kind='bar',
           title='Distribution of Customers According to Store Number',
           legend=False,grid=True)
plt.xlabel('Shop Number')
plt.ylabel('Number of Customers (Thousands)')
plt.show()

# Pie chart

labels_ = ["Shop 1","Shop 2","Shop 3","Shop 4","Shop 5"]
colors = ['lightblue','red','yellow','magenta','green']
list_purchase = np.array(list_purchase) 
list_purchase = list_purchase*100./sum(list_purchase)
 
matplotlib.rcParams.update({'font.size': 12})
plt.pie(list_purchase,explode=explode,colors=colors,labels=labels_,
        autopct='%1.1f%%',shadow=True,startangle=-30)

plt.axis('equal')

plt.title('Customers according to shop number')

plt.show()



data[['distance_shop_1', 'distance_shop_2',
       'distance_shop_3', 'distance_shop_4', 'distance_shop_5']].head()

dist_ = data[['distance_shop_1', 'distance_shop_2',
       'distance_shop_3', 'distance_shop_4', 'distance_shop_5']].values
print dist_

# Empty lists
closest = []
farthest = []
# For every customer (columns in previous matrix )...
for k in range(num_customers):
    # ... find the index that corresponds to the maximum and minimum 
    # distances and add one to it (remember indexing in Python begins
    # at zero while the indexing at customer number begins at one) ...
    Min = 1 + dist_[k,:].argmin()
    Max = 1 + dist_[k,:].argmax()
    #  ... and add each value to corresponding list
    closest.append(Min)
    farthest.append(Max)
# Finally add columns to the data frame
data['closest_shop'] = closest
data['farthest_shop'] = farthest

data[['distance_shop_1', 'distance_shop_2',
       'distance_shop_3', 'distance_shop_4', 'distance_shop_5','closest_shop',
      'farthest_shop']].head()

# Reference for pie chart: https://www.getdatajoy.com/examples/python-plots/pie-chart
list_purchase = []
labels_ = ["Store 1","Store 2","Store 3","Store 4","Store 5"]
colors = ['lightblue','red','yellow','magenta','green']
explode = (0.2,0.05,0.05,0.05,0.05)
for k in range(1,6):
    aux_ = sum(data['closest_shop'] == k)
    perc_ = float(aux_)*100./float(num_customers)
    list_purchase.append(perc_)

    
matplotlib.rcParams.update({'font.size': 12})
plt.pie(list_purchase,explode=explode,colors=colors,labels=labels_,
        autopct='%1.1f%%',shadow=True,startangle=-15)

plt.axis('equal')

plt.title('Customers according to closeness to stores')

plt.show()


# Reference for pie chart: https://www.getdatajoy.com/examples/python-plots/pie-chart
list_purchase = []
labels_ = ["Store 1","Store 2","Store 3","Store 4","Store 5"]
colors = ['lightblue','red','yellow','magenta','green']
explode = (0.05,0.05,0.05,0.05,0.05)
text_ = 'avg_purchase_shop_'
num_ = sum((data['shops_used'] == 1))
for k in range(1,6):
    aux_ = sum((data[text_ + str(k)] != 0) & (data['shops_used'] == 1))
    perc_ = float(aux_)*100./float(num_)
    list_purchase.append(perc_)

    
matplotlib.rcParams.update({'font.size': 12})
plt.pie(list_purchase,explode=explode,colors=colors,labels=labels_,
        autopct='%1.1f%%',shadow=True,startangle=45)

plt.axis('equal')

plt.title('Customers who buy exclusively in one store')

plt.show()



# Fraction of people who purchse in a given store that live closest to it

# Set empty lists and empty data frame
listA = []; listB = []
df_stack = pd.DataFrame()
# Auxiliary test to extract desired values
text_ = 'amount_purchased_shop_'
# For each of the five stores...
for k in range(1,6):
    # ... the total number of customers who purchased at 
    # the store in turn ...
    a = sum((data[text_ + str(k)] != 0.)) 
    # ... and from that total number, get the ones that ALSO live 
    # closest to the store in turn.
    b = sum((data[text_ + str(k)] != 0.) & (data.closest_shop == k))
    
    # From the customers who purchase at a given store we determine the 
    # fraction of them that live the closest to the store in question
    fract_ = float(b)/float(a)
    # Store the complement (the ones who DO NOT live closest to store)...
    listA.append((1. - fract_)*100.)
    # ... and the fraction itself
    listB.append(fract_*100.)

# Create dataframe to make the stacked bar plots
df_stack['Closest to shop'] = listB
df_stack['Not closest to shop'] = listA
df_stack.index = range(1,6)

# And plot
matplotlib.rcParams.update({'font.size': 12})
df_stack.plot(kind='bar',
           title='Fraction of purchases in a given store',stacked=True,
              grid=True)
plt.xlabel('Shop chosen')
plt.ylabel('Percentage of customers')
plt.show()


df_stack

# Empty lists and dataframe
listA = [];listB = []
df_stack = pd.DataFrame()

# Auxiliary text to filter data of interest
text_ = 'amount_purchased_shop_'

# For each store...
for k in range(1,6):
    
    # Get number of customers that purchased in the store in turn and ONLY in that store...
    a =  sum((data[text_ + str(k)] != 0.) & (data.shops_used == 1))
    # ... and extract from them those who are closest to the store in turn
    b = sum((data[text_ + str(k)] != 0.) & (data.shops_used == 1) &
            (data.closest_shop == k))
    
    # From the previous to numbers get the fraction desired...
    fract_ = float(b)/float(a)

    # ... and store it, as well as its complement, in the corresponding
    # lists created before
    listA.append((1. - fract_)*100.)
    listB.append(fract_*100.)
    

# Create the corrsponding dataframe ...
df_stack['Closest to shop'] = listB
df_stack['Not closest to shop'] = listA
df_stack.index = range(1,6)

# ... and plot
df_stack.plot(kind='bar',
           title='Distribution of Customers Purchased in exactly one given store',stacked=True,
              grid=True)
plt.xlabel('Shop chosen')
plt.ylabel('Percentage of customers')
plt.show()

    

df_stack

# PERCENTAGE OF ALL PEOPLE WHO PURCHASE AT ONLY ONE GIVEN SHOP AND ARE CLOSEST TO SUCH STORE

# Empty lists and dataframe
listA = [];listB = []
df_stack = pd.DataFrame()

# Auxiliary text to filter data of interest
text_ = 'amount_purchased_shop_'

# For each store...
for k in range(1,6):
    
    # Get number of customers that purchased ONLY in one store and live the closest
    # to the store in question
    a =  sum((data.closest_shop == k) & (data.shops_used == 1))
    # ... and extract from them those who purchased in that store
    b = sum((data[text_ + str(k)] != 0.) & (data.shops_used == 1) &
            (data.closest_shop == k))
    
    # From the previous to numbers get the fraction desired...
    fract_ = float(b)/float(a)

    # ... and store it, as well as its complement, in the corresponding
    # lists created before
    listA.append((1. - fract_)*100.)
    listB.append(fract_*100.)
    

# Create the corrsponding dataframe ...
df_stack['Closest to shop'] = listB
df_stack['Not closest to shop'] = listA
df_stack.index = range(1,6)

# ... and plot
df_stack.plot(kind='bar',
           title='Distribution of Customers Purchased in exactly one given store',stacked=True,
              grid=True)
plt.xlabel('Shop chosen')
plt.ylabel('Percentage of customers')
plt.show()

    

df_stack

# auxiliar text
text_ = 'amount_purchased_shop_'
# Set empty list that will keep track of each component per segment
one_store = []
# List that will store lists containing contributions from each store
closest_to_X = []
# Set to zero cumulative sum. This will be only used
# as a check
cumul_num = 0
# For evert store...
for k in range(1,6):
    # Subset the dataset to determine the group of one-store-buyers that
    # purchased in the store in turn
    dfX = data[ (data[text_+str(k)] != 0.) & (data.shops_used == 1) ]
    # Get number of customers from the dataframe in turn
    numX = dfX.shape[0]
    # Save such value in the created list ...
    one_store.append(numX)
    # ... and update cumulative sum
    cumul_num += numX
    
    # Store the recently created list into the remaining empty list
    closest_to_X.append( [sum(dfX.closest_shop == n) for n in range(1,6)] )
    
# Create an empty dataframe
df_close = pd.DataFrame()
# populate first column with total number of one-store-buyers according to each store
df_close['Total customers'] = one_store
# Create auxiliary text to populate rest of columns
text_ = 'Closest to '
# For each store ...
for m in range(5):
    # ... populate the corresponding column ...
    df_close[text_+str(m+1)] = [closest_to_X[k][m] for k in range(5)]
# ... and set indices in dataframe
df_close.index = range(1,6)  


# We check cumulative sum agains total number of one-shop-buyers
print cumul_num 
print sum(data.shops_used == 1)

df_close

df_stack = df_close[['Closest to 1','Closest to 2','Closest to 3',
                     'Closest to 4','Closest to 5']]

df_stack


df_stack.plot(kind='bar',
           title='Distribution of only-one-store buyers',stacked=True)
plt.xlabel('Shop chosen')
plt.ylabel('Number of customers')
plt.show()

# Auxiliary texts for the processing
text_1 = 'unique_products_purchased_shop_'
text_2 = 'amount_purchased_shop_'
# Empty list to creat the needed dataframe
list_purchase = []
# For each store ...
for k in range(1,6):
    # ... we subset the dataset to find one-shop buyers and the individuals that purchased 
    # in the corresponding store
    aux_ = max(data[(data[text_2 + str(k)] != 0.) & (data.shops_used == 1) ][text_1 + str(k)].values)
    # Add such a number of the created list
    list_purchase.append(aux_)

# Create empty dataframe
dfAux = pd.DataFrame()
# Add columns
dfAux['Shop Number'] = [1,2,3,4,5]
dfAux['Number of unique products'] = list_purchase
# ... and index it
#dfAux.index = range(1,6)

# Generate the plot
dfAux.plot(x='Shop Number',y='Number of unique products',kind='bar',
           title='Max. number of unique products purchased (only one-store-buyers)',
           legend=False,grid=True)
plt.xlabel('Shop Number')
plt.ylabel('Number of unique products')
plt.show()


# Reference for pie chart: https://www.getdatajoy.com/examples/python-plots/pie-chart
list_purchase = []
labels_ = ["Store 2","Store 3","Store 4","Store 5"]
colors = ['red','yellow','magenta','green']
explode = (0.05,0.05,0.05,0.05)
text_ = 'avg_purchase_shop_'
num_ = sum((data['shops_used'] == 2) & (data['avg_purchase_shop_1'] != 0))

cumul_ = 0
for k in range(2,6):
    aux_ = sum((data[text_ + str(k)] != 0) & (data['shops_used'] == 2) & (data['avg_purchase_shop_1'] != 0))   
    cumul_ += aux_
    perc_ = float(aux_)*100./float(num_)
    list_purchase.append(perc_)

matplotlib.rcParams.update({'font.size': 12})
plt.pie(list_purchase,explode=explode,colors=colors,labels=labels_,
        autopct='%1.1f%%',shadow=True,startangle=45)

plt.axis('equal')

plt.title('Customers who buy exclusively in shop 1 and exactly one other shop')

plt.show()

def ABTest_Proportions(data,field_1,field_2,alpha_):
    acol = np.array(data[data.shops_used == 1][field_1].apply(lambda x: 1. if x !=0 else 0 ))
    bcol = np.array(data[data.shops_used == 1][field_2].apply(lambda x: 1. if x !=0 else 0 ))

    
    if ((len(acol) < 30) | (len(bcol) < 30)):
        print "Your samples do not seem to be large enough"
        return len(acol),len(bcol)
    else:
        # Compute proportions
        a_proportion = sum(acol)/len(acol)
        b_proportion = sum(bcol)/len(bcol)
        # Test statistic
        test_statistic = (a_proportion - b_proportion )
        # Pooled proportion
        pooled_proportion = (sum(acol) + sum(bcol))/(len(acol) + len(bcol)) 
        # Standard error
        SError = np.sqrt(pooled_proportion*(1-pooled_proportion)* (1./len(acol) + 1./len(bcol))  ) 
        # Z-score
        z_score = test_statistic / SError
        p_value = st.norm.sf(abs(z_score)) #one-sided
        print 'z score = ' , z_score
        print 'p value = ' , p_value
        print 'Null hypothesis is' , p_value > alpha_
        print "Difference in proportions between" , field_1 , "and" , field_2 
        if p_value > alpha_:
            print "may have been due to chance"
        else: 
            print "may NOT have been due to chance"
        print "######"*10

text_ = 'amount_purchased_shop_'
for i in range(1,6):
    for j in range(i+1,6):
        print "Comparing proportions for shops "  + str(i) + " and " + str(j)
        ABTest_Proportions(data,'amount_purchased_shop_'+str(i),'amount_purchased_shop_'+str(j),0.05)

# Auxiliary texts for the processing
text_1 = 'unique_products_purchased_shop_'
text_2 = 'amount_purchased_shop_'
# Empty list to creat the needed dataframe
list_purchase = []
# For each store ...
for k in range(1,6):
    # ... we subset the dataset to find one-shop buyers and the individuals that purchased 
    # in the corresponding store
    aux_ = max(data[(data[text_2 + str(k)] != 0.) & (data.shops_used == 1) ][text_1 + str(k)].values)
    # Add such a number of the created list
    list_purchase.append(aux_)

# Create empty dataframe
dfAux = pd.DataFrame()
# Add columns
dfAux['Shop Number'] = [1,2,3,4,5]
dfAux['Number of unique products'] = list_purchase
# ... and index it
#dfAux.index = range(1,6)

# Generate the plot
dfAux.plot(x='Shop Number',y='Number of unique products',kind='bar',
           title='Max. number of unique products purchased (only one-store-buyers)',
           legend=False,grid=True)
plt.xlabel('Shop Number')
plt.ylabel('Number of unique products')
plt.show()

print list_purchase

# Important references: 
# https://www.quora.com/Is-it-possible-to-use-cross-validation-to-select-the-number-of-clusters-for-k-means-or-the-EM-algorithm-for-mixture-models-of-Gaussians
# http://www.statsoft.com/Textbook/Cluster-Analysis#vfold

# We import tools to split dataset into train and test sets
from sklearn.cross_validation import train_test_split

# We focus on customers who purchased exactly in only two shops and grab only unique products purchased in every store 
# as well as distance to each of them.
unique_2 = data[(data.shops_used == 2)][['unique_products_purchased_shop_1',
                                                                           'unique_products_purchased_shop_2',
                                         'unique_products_purchased_shop_3','unique_products_purchased_shop_4',
                                         'unique_products_purchased_shop_5']].values


#unique_2 = data[(data.shops_used == 2)][['unique_products_purchased_shop_1',
#                                                                           'unique_products_purchased_shop_2',
#                                         'unique_products_purchased_shop_3','unique_products_purchased_shop_4',
#                                         'unique_products_purchased_shop_5','distance_shop_1','distance_shop_2',
#                                                                           'distance_shop_3','distance_shop_4',
#                                                                           'distance_shop_5']].values


# We split dataset into training set (70%) and test set (30%)
train_2, test_2 = train_test_split(unique_2, train_size=0.7)
# We get an idea of the number of records in the training set
print train_2.shape

from sklearn import preprocessing

print train_2.mean(axis=0)
print train_2.std(axis=0)

train_2 = preprocessing.scale(train_2)
print train_2.mean(axis=0)
print train_2.std(axis=0)

from scipy.spatial.distance import cdist, pdist
from matplotlib import pyplot as plt

# Determine your k range
k_range = range(1,10)

# Fit the kmeans model for each n_clusters = k
k_means_var = [KMeans(n_clusters=k).fit(train_2) for k in k_range]

# Pull out the cluster centers for each model
centroids = [X.cluster_centers_ for X in k_means_var]

# Calculate the Euclidean distance from 
# each point to each cluster center
k_euclid = [cdist(train_2, cent, 'euclidean') for cent in centroids]
dist = [np.min(ke,axis=1) for ke in k_euclid]

# Total within-cluster sum of squares
wcss = [sum(d**2) for d in dist]

# The total sum of squares
tss = sum(pdist(train_2)**2)/train_2.shape[0]

# The between-cluster sum of squares
bss = tss - wcss

# elbow curve
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(k_range, bss/tss*100, 'b*-')
ax.set_ylim((0,100))
plt.grid(True)
plt.xlabel('Number of clusters')
plt.ylabel('Percentage of variance explained')
plt.title('Variance Explained vs. k')

num_clusters = 6
k_means = KMeans(n_clusters=num_clusters)
kMeans = k_means.fit(train_2)
clusters_labels = kMeans.predict(train_2) + 1

df_clusters = pd.DataFrame()
df_clusters['shop_1'] = train_2[:,0]
df_clusters['shop_2'] = train_2[:,1]
df_clusters['shop_3'] = train_2[:,2]
df_clusters['shop_4'] = train_2[:,3]
df_clusters['shop_5'] = train_2[:,4]
df_clusters['cluster'] = clusters_labels

df_clusters.head()

matrix_clusters = np.zeros((num_clusters,5))
for m in range(num_clusters):
    #print "Cluster " , m+1
    for k in range(5):
        text = 'shop_' + str(k+1) 
        sum_ = sum((df_clusters.cluster == m+1) & (df_clusters['shop_'+str(k+1)] != 0 ))
        #print text, sum_
        matrix_clusters[m,k] = sum_
    #print "#######"*2

    
df = pd.DataFrame()
for k in range(5):
    df['Shop ' + str(k+1)] = matrix_clusters[:,k]
df.index = range(1,num_clusters+1) #['Cluster '+ str(i) for i in range(1,6)]

df

df.plot(kind='bar',
           title='Distribution of only-one-store buyers',stacked=True)
plt.xlabel('Cluster')
plt.ylabel('Number of customers')
plt.show()

# Important references: 
# https://www.quora.com/Is-it-possible-to-use-cross-validation-to-select-the-number-of-clusters-for-k-means-or-the-EM-algorithm-for-mixture-models-of-Gaussians
# http://www.statsoft.com/Textbook/Cluster-Analysis#vfold

# We import tools to split dataset into train and test sets
from sklearn.cross_validation import train_test_split

# We focus on customers who purchased exactly in only two shops and grab only unique products purchased in every store 
# as well as distance to each of them.


unique_2 = data[(data.shops_used == 2)][['distance_shop_1','distance_shop_2',
                                                                           'distance_shop_3','distance_shop_4',
                                                                           'distance_shop_5']].values


#unique_2 = data[(data.shops_used == 2)][['unique_products_purchased_shop_1',
#                                                                           'unique_products_purchased_shop_2',
#                                         'unique_products_purchased_shop_3','unique_products_purchased_shop_4',
#                                         'unique_products_purchased_shop_5','distance_shop_1','distance_shop_2',
#                                                                           'distance_shop_3','distance_shop_4',
#                                                                           'distance_shop_5']].values


# We split dataset into training set (70%) and test set (30%)
train_2, test_2 = train_test_split(unique_2, train_size=0.7)
# We get an idea of the number of records in the training set
print train_2.shape

print train_2.mean(axis=0)
print train_2.std(axis=0)
train_2 = preprocessing.scale(train_2)
print train_2.mean(axis=0)
print train_2.std(axis=0)

from scipy.spatial.distance import cdist, pdist
from matplotlib import pyplot as plt

# Determine your k range
k_range = range(1,10)

# Fit the kmeans model for each n_clusters = k
k_means_var = [KMeans(n_clusters=k).fit(train_2) for k in k_range]

# Pull out the cluster centers for each model
centroids = [X.cluster_centers_ for X in k_means_var]

# Calculate the Euclidean distance from 
# each point to each cluster center
k_euclid = [cdist(train_2, cent, 'euclidean') for cent in centroids]
dist = [np.min(ke,axis=1) for ke in k_euclid]

# Total within-cluster sum of squares
wcss = [sum(d**2) for d in dist]

# The total sum of squares
tss = sum(pdist(train_2)**2)/train_2.shape[0]

# The between-cluster sum of squares
bss = tss - wcss

# elbow curve
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(k_range, bss/tss*100, 'b*-')
ax.set_ylim((0,100))
plt.grid(True)
plt.xlabel('Number of clusters')
plt.ylabel('Percentage of variance explained')
plt.title('Variance Explained vs. k')

num_clusters = 6
k_means = KMeans(n_clusters=num_clusters)
kMeans = k_means.fit(test_1)
clusters_labels = kMeans.predict(test_1) + 1

df_clusters = pd.DataFrame()
df_clusters['shop_1'] = test_1[:,0]
df_clusters['shop_2'] = test_1[:,1]
df_clusters['shop_3'] = test_1[:,2]
df_clusters['shop_4'] = test_1[:,3]
df_clusters['shop_5'] = test_1[:,4]
df_clusters['cluster'] = clusters_labels

df_clusters.head()

matrix_clusters = np.zeros((num_clusters,5))
for m in range(num_clusters):
    #print "Cluster " , m+1
    for k in range(5):
        text = 'shop_' + str(k+1) 
        sum_ = sum((df_clusters.cluster == m+1) & (df_clusters['shop_'+str(k+1)] != 0 ))
        #print text, sum_
        matrix_clusters[m,k] = sum_
    #print "#######"*2

    
df = pd.DataFrame()
for k in range(5):
    df['Shop ' + str(k+1)] = matrix_clusters[:,k]
df.index = range(1,num_clusters+1) #['Cluster '+ str(i) for i in range(1,6)]

df

df.plot(kind='bar',
           title='Distribution of only-two-store buyers',stacked=True)
plt.xlabel('Cluster')
plt.ylabel('Number of customers')
plt.show()

from sklearn.decomposition import PCA


unique_2 = data[(data.shops_used == 2)][['unique_products_purchased_shop_1',
                                                                           'unique_products_purchased_shop_2',
                                         'unique_products_purchased_shop_3','unique_products_purchased_shop_4',
                                         'unique_products_purchased_shop_5','distance_shop_1','distance_shop_2',
                                                                           'distance_shop_3','distance_shop_4',
                                                                           'distance_shop_5']].values


# We split dataset into training set (70%) and test set (30%)
train_2, test_2 = train_test_split(unique_2, train_size=0.7)
# We get an idea of the number of records in the training set
print train_2.shape

print "Feature normalization"
print ""
print train_2.mean(axis=0)
print train_2.std(axis=0)
train_2 = preprocessing.scale(train_2)
print train_2.mean(axis=0)
print train_2.std(axis=0)

# Number of components to consider
comp_ = 10

# Generate model
pca = PCA(n_components=comp_)

# Fit model
pca2 = pca.fit_transform(train_2)

# Print out useful information reagarding the variance captured by the 
# number of components chosen


#print pca.explained_variance_
#print "Ratio of explained variance"
#print pca.explained_variance_ratio_
print "Cumulative sum of ratio of explained variance"
pca_results = pca.explained_variance_ratio_.cumsum()
for k in range(10):
    print round(pca_results[k]*100.,2), "% explained by " +str(k+1)+ " components"

# Number of components to consider
comp_ = 6

# Generate model
pca = PCA(n_components=comp_)

# Fit model
pca2 = pca.fit_transform(train_2)

# Print out useful information reagarding the variance captured by the 
# number of components chosen


#print pca.explained_variance_
print "Ratio of explained variance"
print pca.explained_variance_ratio_
print "Cumulative sum of ratio of explained variance"
print pca.explained_variance_ratio_.cumsum()

# Determine your k range
k_range = range(1,10)

# Fit the kmeans model for each n_clusters = k
k_means_var = [KMeans(n_clusters=k).fit(pca2) for k in k_range]

# Pull out the cluster centers for each model
centroids = [X.cluster_centers_ for X in k_means_var]

# Calculate the Euclidean distance from 
# each point to each cluster center
k_euclid = [cdist(pca2, cent, 'euclidean') for cent in centroids]
dist = [np.min(ke,axis=1) for ke in k_euclid]

# Total within-cluster sum of squares
wcss = [sum(d**2) for d in dist]

# The total sum of squares
tss = sum(pdist(pca2)**2)/pca2.shape[0]

# The between-cluster sum of squares
bss = tss - wcss

# elbow curve
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(k_range, bss/tss*100, 'b*-')
ax.set_ylim((0,100))
plt.grid(True)
plt.xlabel('Number of clusters')
plt.ylabel('Percentage of variance explained')
plt.title('Variance Explained vs. k')

num_clusters = 6
k_means = KMeans(n_clusters=num_clusters)
kMeans = k_means.fit(pca2)
clusters_labels = kMeans.predict(pca2) + 1


df_clusters = pd.DataFrame()
for k in range(comp_):
    df_clusters['comp_' + str(k+1)] = pca2[:,k]
    df_clusters['comp_' + str(k+1)] = pca2[:,k]
df_clusters['cluster'] = clusters_labels

df_clusters.head()

matrix_clusters = np.zeros((num_clusters,comp_))
for m in range(num_clusters):
    #print "Cluster " , m+1
    for k in range(comp_):
        text = 'shop_' + str(k+1) 
        sum_ = sum((df_clusters.cluster == m+1) & (df_clusters['comp_'+str(k+1)] != 0 ))
        #print text, sum_
        matrix_clusters[m,k] = sum_
    #print "#######"*2

    
df = pd.DataFrame()
for k in range(comp_):
    df['Comp ' + str(k+1)] = matrix_clusters[:,k]
df.index = range(1,num_clusters+1) #['Cluster '+ str(i) for i in range(1,6)]

df

df.plot(kind='bar',
           title='Distribution of only-two-store buyers',stacked=True)
plt.xlabel('Cluster')
plt.ylabel('Number of customers')
plt.show()

# Reference 
# http://sebastianraschka.com/Articles/2015_pca_in_3_steps.html


#unique_2 = data[(data.shops_used == 2)][['unique_products_purchased_shop_1',
#                                         'unique_products_purchased_shop_2',
#                                         'unique_products_purchased_shop_3',
#                                         'unique_products_purchased_shop_4',
#                                         'unique_products_purchased_shop_5',
#                                         'distance_shop_1','distance_shop_2',
#                                         'distance_shop_3','distance_shop_4',
#                                         'distance_shop_5']].values

unique_2 = data[(data.shops_used == 2)][['unique_products_purchased_shop_1',
                                         'unique_products_purchased_shop_2',
                                         'unique_products_purchased_shop_3',
                                         'unique_products_purchased_shop_4',
                                         'unique_products_purchased_shop_5',
                                         'distance_shop_1','distance_shop_2',
                                         'distance_shop_3','distance_shop_4',
                                         'distance_shop_5']].values

#unique_2 = data[(data.shops_used == 1)].values

unique_2 = unique_2.astype(float)
# We split dataset into training set (70%) and test set (30%)
train_2, test_2 = train_test_split(unique_2, train_size=0.7)
# We get an idea of the number of records in the training set
print train_2.shape

print "Feature normalization"
print ""
print train_2.mean(axis=0)
print train_2.std(axis=0)
train_2 = preprocessing.scale(train_2)
print train_2.mean(axis=0)
print train_2.std(axis=0)

# Number of components to consider
comp_ = 10

# Generate model
pca = PCA(n_components=comp_)

# Fit model
pca2 = pca.fit_transform(train_2)

# Print out useful information reagarding the variance captured by the 
# number of components chosen


#print pca.explained_variance_
#print "Ratio of explained variance"
#print pca.explained_variance_ratio_
print "Cumulative sum of ratio of explained variance"
pca_results = pca.explained_variance_ratio_.cumsum()
for k in range(comp_):
    print round(pca_results[k]*100.,3), "% explained by " +str(k+1)+ " components"



