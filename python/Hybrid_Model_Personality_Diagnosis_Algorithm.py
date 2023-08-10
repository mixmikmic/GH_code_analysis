import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

r_cols = ['user_id','movie_id', 'rating','unix_timestamp']
ratings=pd.read_csv('ratings.csv', sep=',', names=r_cols, encoding='latin-1',skiprows=1)
ratings.head()

#from sklearn.cross_validation import train_test_split
#train, test = train_test_split(ratings, test_size=0.2) #splitting data into test and train
#print train.shape,test.shape

#train_df2 = train.pivot(index = 'user_id', columns ='movie_id', values = 'rating')
#test_df2=test.pivot(index = 'user_id', columns ='movie_id', values = 'rating')
#train_df2.shape
#print sum(train_df2.count())
#test_df.shape

ratings_df = ratings.pivot(index = 'user_id', columns ='movie_id', values = 'rating')
ratings_df.head()#67189066 matrix
#print len(ratings_df)
print ratings_df.shape
train_df=ratings_df
#test_df=ratings_df
train_df.count()
print sum(train_df.count())
ratings_df.to_csv('ratings_df.csv')
print sum(test_df.count())
ratings_df.iloc[8,0]

import random
ix = [(row, col) for row in range(ratings_df.shape[0]) for col in range(ratings_df.shape[1])]
pointer_list=[] #pointers_list - the list that contains all tuples of ratings in the ratings universe
for row,col in ix:
    if ratings_df.iat[row,col]>0:
        pointer_list.append((row,col))
print 'len(pointer_list)',len(pointer_list)
sample_20=random.sample(pointer_list, int(round(.2*len(pointer_list))))#20% of sample overwritten with NaN
print 'len(sample_20)',len(sample_20)

sample_80=list(set(pointer_list)-set(sample_20))
print len(sample_80)#rest of the 8-% sample from initial ratings set
print sum(ratings_df.count())

for row, col in sample_20:
    train_df.iat[row, col] = np.nan #train_df gets updated with 80% of the data
print train_df.shape,sum(train_df.count()) 

train_df.to_csv('train.csv')

#test_df=ratings_df
#print sum(test_df.count())
ratings_df = ratings.pivot(index = 'user_id', columns ='movie_id', values = 'rating')
test_df=ratings_df
print sum(test_df.count())
print sum(ratings_df.count())

for row, col in sample_80:
    test_df.iat[row, col] = np.nan
print test_df.shape,sum(test_df.count())

test_df.to_csv('test.csv')

ratings_df = ratings.pivot(index = 'user_id', columns ='movie_id', values = 'rating')
print sum(ratings_df.count())

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
count_ratings = pd.value_counts(ratings['rating'], sort = False).sort_index()
count_ratings.plot(kind = 'bar')
plt.rc("font",size=15)
plt.title("Ratings Distribution")
plt.xlabel("Rating")
plt.ylabel("Count")
plt.show()

ratings_matrix = ratings_df.as_matrix()#changing pandas df to numpy matrix form, this is a ratings user-item matrix.
#print ratings_matrix.view(type=np.matrix)
user_mean=np.nanmean(ratings_matrix, axis=1) #computing mean of each user - only for items the user has rated 
ratings_demeaned = ratings_matrix - user_mean.reshape(-1, 1)#matrix where mean is subtracted from ratings
ratings_demeaned.view(type=np.matrix)
#np.savetxt('ratings_matrix.csv', ratings_matrix, delimiter=',')
print ratings_matrix

#ratings_df2 = ratings.pivot(index = 'user_id', columns ='movie_id', values = 'rating').fillna(0)
#filled NaN values with 0 for easy computing of simlilarity
from numpy import *
ratings_demeaned_wo_nan=ratings_demeaned
where_are_NaNs = isnan(ratings_demeaned_wo_nan)
ratings_demeaned_wo_nan[where_are_NaNs] = 0
ratings_demeaned_wo_nan.view(type=np.matrix)

from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
ratings_sparse = sparse.csr_matrix(ratings_demeaned_wo_nan)#sparse gives output of only those values which have numbers
#other than 0's. this way we can compute cosine similarity between user-user.
#print ratings_sparse
similarities = cosine_similarity(ratings_sparse)#gives user-user similarity
print similarities,len(similarities)
print "-----------"
#similarities2=cosine_similarity(ratings_sparse.transpose())#computes user-user similarity
#print similarities2,len(similarities2)

train_matrix = train_df.as_matrix()
test_matrix = test_df.as_matrix()
#np.savetxt('train_matrix.csv', train_matrix, delimiter=',')
#np.savetxt('test_matrix.csv', test_matrix, delimiter=',')

user_train_mean=np.nanmean(train_matrix, axis=1) #computing mean of each user - only for items the user has rated 
train_demeaned = train_matrix - user_train_mean.reshape(-1, 1)#matrix where mean is subtracted from ratings
train_demeaned.view(type=np.matrix)
#pqr=pd.DataFrame(train_demeaned)
#print sum(pqr.count())

from numpy import *
train_demeaned_wo_nan=train_demeaned
where_are_NaNs = isnan(train_demeaned_wo_nan)
train_demeaned_wo_nan[where_are_NaNs] = 0
train_demeaned_wo_nan.view(type=np.matrix)

from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
train_sparse = sparse.csr_matrix(train_demeaned_wo_nan)#sparse gives output of only those values which have numbers
#other than 0's. this way we can compute cosine similarity between user-user.
#print ratings_sparse
similarities_train = cosine_similarity(train_sparse)#gives user-user similarity
print similarities_train,len(similarities_train)
print "-----------"
#similarities2=cosine_similarity(ratings_sparse.transpose())#computes user-user similarity
#print similarities2,len(similarities2)

#print len(similarities[0]),similarities[0]
#user_nn50_index=[]#top 50 nearest neighbouring users of active user
#for i in range(len(similarities)):   #similarities[i] is array of similarity values of active user with all users
#    a=sorted(range(len(similarities[i])), key=lambda j: similarities[i][j], reverse=True)[1:51]
#    user_nn50_index.append(a)
#print user_nn50_index

def user_nn(i,j):  #i is the index of user and not user value i.e. if for user 550, 'i' is 549 in the matrix
                   #j is the index of the item --- rating of item 'j' for user 'i' should exist and needs to be
                    #into the return of user_nn
    user_nn_index=[]#neareast neighbouring users of active user but <50 and similarity strictly positive (>0)
    b=[]
    c=[]
    d=[]
    for s in range(len(ratings_matrix)):#this loop filters out users who have rated item 'j'
        if ratings_matrix[s][j]>0:
            c.append(s)
    #print 'c=',c
    for p in range(len(similarities[i])):#this loop filters out user-user similarity > 0
        if similarities[i][p]>0:
            b.append(p)
    #print 'b=',b
    d=list(set(b).intersection(c))
    #print 'Xn',len(d),d
    #print b,[similarities[i][j] for j in b]
    user_nn_index=sorted(d, key=lambda q: similarities[i][q], reverse=True)[1:51]
    return user_nn_index
    
    #return len(user_nn_index),user_nn_index,[ratings_matrix[r][j] for r in user_nn_index]
    #print user_nn_index, [similarities[i][j] for j in user_nn_index]
    #if len(user_nn_index)>51:  #51 because user-user similarity for user itself is 1.. 
        #correlation of this 1 will be removed in future lines
        #user_nn_index=user_nn_index[:51]
        
    #return user_nn_index[1:],[similarities[i][j] for j in user_nn_index[1:]]
        #b=(range(len(similarities[i])), key=lambda j: j>0)
#print (user_nn50_index)
user_nn(26,13)

def common_items(i,j):  #users with index i and j, function gives commonly rated items's index from user-item matrix
    common_items_index=[]
    for p in range(len(ratings_matrix[i])):
        #print ratings_matrix[i][p],ratings_matrix[j][p]
        if ratings_matrix[i][p]>0 and ratings_matrix[j][p]>0:
            common_items_index.append(p)
    return common_items_index
common_items(19,293)

import math
def pd_rating(a,j):   #a is index of the active user, j is the item for which rating needs to be predicted
    user_nn_index=[]
    master_common_items_index=[]
    common_items_index=[]
    sigma=2
    prob=[]
    h_value=range(1,6)
    for h in range(1,6):
        user_nn_index=user_nn(a,j)#this is a list of all nearest neighbouring users
        #p=[]
        #for x in user_nn_index:
            #p=common_items(a,x)
            #master_common_items_index.append(p)#list with common items b/w each pair of users
        q=0#rating vector term
        power=0
        exponent=0
        #print 'user_nn_index',user_nn_index
        for r in user_nn_index:
            common_items_index=common_items(a,r)
            #print 'common_items_index',common_items_index
            for x in common_items_index:
                q+=(ratings_matrix[a][x]-ratings_matrix[r][x])**2
                #print q,a,ratings_matrix[a][x],r,ratings_matrix[r][x]
            power=(h-ratings_matrix[r][j])**2+q
            #print ratings_matrix[r][j]
            exponent+=math.exp(-power/2**(sigma**2))
        #exponent=exponent/len(user_nn_index)
        exponent=exponent
        prob.append(exponent)
        #print 'h=',h,prob
    dict_pair=dict(zip(prob,h_value))
    #print dict_pair
    return dict_pair[max(prob)]

pd_rating(19,5)
        
    
    

def user_train_nn(i,j):  #i is the index of user and not user value i.e. if for user 550, 'i' is 549 in the matrix
                   #j is the index of the item --- rating of item 'j' for user 'i' should exist and needs to be
                    #into the return of user_nn
    user_train_nn_index=[]#neareast neighbouring users of active user but <50 and similarity strictly positive (>0)
    b=[]
    c=[]
    d=[]
    for s in range(len(train_matrix)):#this loop filters out users who have rated item 'j'
        if train_matrix[s][j]>0:
            c.append(s)
    #print 'c=',c
    for p in range(len(similarities_train[i])):#this loop filters out user-user similarity > 0
        if similarities_train[i][p]>0:
            b.append(p)
    #print 'b=',b
    d=list(set(b).intersection(c))
    #print 'Xn',len(d),d
    #print b,[similarities[i][j] for j in b]
    user_train_nn_index=sorted(d, key=lambda q: similarities_train[i][q], reverse=True)[1:51]
    return user_train_nn_index
user_train_nn(19,7)

def common_train_items(i,j):  #users with index i and j, function gives commonly rated items's index from user-item matrix
    common_train_items_index=[]
    for p in range(len(train_matrix[i])):
        #print ratings_matrix[i][p],ratings_matrix[j][p]
        if train_matrix[i][p]>0 and train_matrix[j][p]>0:
            common_train_items_index.append(p)
    return common_train_items_index

import math
def pd_train_rating(a,j):   #a is index of the active user, j is the item for which rating needs to be predicted
    user_train_nn_index=[]
    master_train_common_items_index=[]
    common_train_items_index=[]
    sigma=2
    prob=[]
    values=[0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0]
    h_value=values
    for h in values:
        user_train_nn_index=user_train_nn(a,j)#this is a list of all nearest neighbouring users
        #p=[]
        #for x in user_nn_index:
            #p=common_items(a,x)
            #master_common_items_index.append(p)#list with common items b/w each pair of users
        q=0#rating vector term
        power=0
        exponent=0
        #print 'user_train_nn_index',user_train_nn_index
        for r in user_train_nn_index:
            common_train_items_index=common_train_items(a,r)
            #print 'common_items_index',common_items_index
            for x in common_train_items_index:
                q+=(train_matrix[a][x]-train_matrix[r][x])**2
                #print q,a,ratings_matrix[a][x],r,ratings_matrix[r][x]
            power=(h-train_matrix[r][j])**2+q
            #print ratings_matrix[r][j]
            exponent+=math.exp(-power/2**(sigma**2))
        #exponent=exponent/len(user_train_nn_index)
        exponent=exponent
        prob.append(exponent)
        #print 'h=',h,prob
    dict_pair=dict(zip(prob,h_value))
    #print dict_pair
    return dict_pair[max(prob)]

pd_train_rating(19,7)

i=0
sample_5k_1=[]
sample_5k_2=[]
sample_5k_3=[]
sample_5k_4=[]
for row,col in sample_20:
    i+=1
    if i<=5000:
        sample_5k_1.append((row,col))
    elif 5001<=i<=10000:
        sample_5k_2.append((row,col))
    elif 10001<=i<=15000:
        sample_5k_3.append((row,col))
    else:
        sample_5k_4.append((row,col))
print len(sample_5k_1),len(sample_5k_2),len(sample_5k_3),len(sample_5k_4)

predicted_rating=[]
m=[]
i=0
for row, col in sample_5k_1:
    m=pd_train_rating(row, col)
    i+=1
    print i
    if m>0:
        predicted_rating.append(m)
    else:
        predicted_rating.append(None)
actual_rating=[]
n=[]
for row, col in sample_5k_1:
    n=ratings_df.iloc[row,col]
    actual_rating.append(n)
print predicted_rating
print actual_rating

print len(predicted_rating),len(actual_rating)

predicted_rating1=predicted_rating
actual_rating1=actual_rating

from sklearn.metrics import mean_squared_error
from math import sqrt

rmse = sqrt(mean_squared_error(actual_rating, predicted_rating))
print rmse

predicted_rating2=[]
m=[]
i=0
for row, col in sample_5k_2:
    m=pd_train_rating(row, col)
    i+=1
    print i
    if m>0:
        predicted_rating2.append(m)
    else:
        predicted_rating2.append(None)
actual_rating2=[]
n=[]
for row, col in sample_5k_2:
    n=ratings_df.iloc[row,col]
    actual_rating2.append(n)
print len(predicted_rating2),len(actual_rating2)

predicted_rating3=[]
m=[]
i=0
for row, col in sample_5k_3:
    m=pd_train_rating(row, col)
    i+=1
    print i
    if m>0:
        predicted_rating3.append(m)
    else:
        predicted_rating3.append(None)
actual_rating3=[]
n=[]
for row, col in sample_5k_3:
    n=ratings_df.iloc[row,col]
    actual_rating3.append(n)
print len(predicted_rating3),len(actual_rating3)

predicted_rating4=[]
m=[]
i=0
for row, col in sample_5k_4:
    m=pd_train_rating(row, col)
    i+=1
    print i
    if m>0:
        predicted_rating4.append(m)
    else:
        predicted_rating4.append(None)
actual_rating4=[]
n=[]
for row, col in sample_5k_4:
    n=ratings_df.iloc[row,col]
    actual_rating4.append(n)
print len(predicted_rating4),len(actual_rating4)

predicted_rating=predicted_rating1+predicted_rating2+predicted_rating3+predicted_rating4
actual_rating=actual_rating1+actual_rating2+actual_rating3+actual_rating4
print len(predicted_rating),len(actual_rating)

from sklearn.metrics import mean_squared_error
from math import sqrt

rmse = sqrt(mean_squared_error(actual_rating, predicted_rating))
print rmse



