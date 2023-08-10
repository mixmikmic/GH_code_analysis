import csv
import pandas
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime
import matplotlib
import seaborn as sns
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition

users = pandas.read_csv("data/users.csv", header=None)
conversions = pandas.read_csv("data/conversions.csv", header=None)
items = pandas.read_csv("data/items.csv", header=None)
users_ads = pandas.read_csv("data/users_ads.csv", header=None)
views = pandas.read_csv("data/views.csv", header=None, low_memory=False)

users.columns = ['userId', 'registerCountry', 'signupTime']
conversions.columns = ['userId', 'itemId', 'price', 'quantity', 'timestamp']
items.columns = ['itemId', 'style', 'personality', 'color', 'theme', 'price', 'category']
users_ads.columns = ['userId', 'utmSource', 'utmCampaign', 'utmMedium', 'utmTerm', 'utmContent']
views.columns = ['userId', 'itemId', 'timestamp', 'pagetype']

users.signupTime = pandas.to_datetime(users.signupTime)
conversions.timestamp = pandas.to_datetime(conversions.timestamp)
views.timestamp = pandas.to_datetime(views.timestamp)

from scipy.sparse import lil_matrix, kron,identity
from scipy.sparse.linalg import lsqr

users.signupTime = pandas.to_datetime(users.signupTime)
conversions.timestamp = pandas.to_datetime(conversions.timestamp)

item_indices = items.itemId.value_counts().index
item_indices = pandas.Series(range(len(item_indices)), index = item_indices)

user_indices = users.userId.value_counts().index
user_indices = pandas.Series(range(len(user_indices)), index = user_indices)

def users_as_real_vectors(users_df):
    user_number_of_purchases = lil_matrix((user_indices.size, item_indices.size))
    for index, row in conversions.iterrows():
        if not (row.itemId in item_indices.index) or not (row.userId in user_indices.index):
            continue
        item_index = item_indices[row.itemId]
        user_index = user_indices[row.userId]
        user_number_of_purchases[user_index, item_index] += row.quantity
    print(user_number_of_purchases.size)
    return user_number_of_purchases

usr = users_as_real_vectors(users)

import random
def random_subset(df, percent):
    new_size  = int(percent*len(df.index))
    subset = random.sample(set(df.index), new_size)
    return df.ix[subset]

users_small = random_subset(users, 0.2)
items_small = random_subset(items, 0.2)
users_ads_small = random_subset(users_ads, 0.2)
conversions_small = random_subset(conversions, 0.2)
# views_small = random_subset(views, 0.2)

full_info_conversions = conversions.merge(users, how='inner', on='userId')
full_info_conversions = full_info_conversions.merge(items, how='inner', on='itemId')
print(full_info_conversions.columns.values)

print(full_info_conversions.index.size)
print(conversions.index.size)
print(full_info_conversions.userId.value_counts().index.size)

full_info_views_conversions = full_info_conversions.merge(views, how='inner', on='userId')
print(full_info_views_conversions.columns.values)
print(views.columns.values)

filter = ~full_info_conversions.userId.isin(full_info_views_conversions.userId)
zero_views_and_have_conversions = full_info_conversions[filter]
zero_views_and_have_conversions['spending'] = zero_views_and_have_conversions.price_y * zero_views_and_have_conversions.quantity

def mean_or_zero(series):
    if series.size == 0:
        return 0.0
    return series.mean()
import math

def charts_max_k_views_bought_something(period_of_time_days, k):
    df = full_info_views_conversions[full_info_views_conversions.timestamp_x <= full_info_views_conversions.signupTime + pandas.Timedelta(period_of_time_days)]
    df['spending'] = df.price_y * df.quantity

    sns.set()
    f, (axes) = sns.plt.subplots(3, sharex=False, sharey=False)
    main_title = "Charts binned by number of initial views (max " + str(k) + " views) during first " + str(period_of_time_days) + " days after registration" 
    f.suptitle(main_title,fontsize=20)
    f.set_size_inches(15,30)
    f.tight_layout(pad=1, w_pad=1, h_pad=13)
    plt.subplots_adjust(top=0.91)
    
    sum_df = pandas.DataFrame()
    sum_df['userId'] = df.userId
    sum_df['quantity'] = df.quantity
    sum_df['spending']  = df.spending
    sum_df['number_views'] = np.ones(sum_df.spending.size)
    
    sum_df = sum_df.groupby('userId').sum()

    mean_spending_k=[]
    for i in range(k):
        mean_spending_k.append((sum_df[sum_df.number_views == i]).spending.mean())
    if math.isnan(mean_spending_k[0]):
        mean_spending_k[0] = zero_views_and_have_conversions.spending.mean()

    mean_spending_k_series = pandas.Series(mean_spending_k, index=range(k))
    mean_spending_k_series.plot(ax=axes[0],kind='bar')
    axes[0].set_title("Average spending", fontsize=16)
    axes[0].set_xlabel("Number of views", fontsize=13)
    axes[0].set_ylabel("Mean spending", fontsize=13)
    
    mean_quantity_k=[]
    for i in range(k):
        mean_quantity_k.append((sum_df[sum_df.number_views == i]).quantity.mean())
    if math.isnan(mean_quantity_k[0]):
        mean_quantity_k[0] = zero_views_and_have_conversions.quantity.mean()

    mean_quantity_k = pandas.Series(mean_quantity_k, index=range(k))
    mean_quantity_k.plot(ax=axes[1],kind='bar')
    axes[1].set_title("Mean quantity of bought objects", fontsize=16)
    axes[1].set_xlabel("Number of views", fontsize=13)
    axes[1].set_ylabel("Mean quantity", fontsize=13)

    amount_of_people_k=[]
    for i in range(k):
        amount_of_people_k.append((sum_df[sum_df.number_views == i]).quantity.size)
    amount_of_people_k[0] = zero_views_and_have_conversions.userId.size
    amount_of_people_k = pandas.Series(mean_quantity_k, index=range(k))
    amount_of_people_k.plot(ax=axes[2],kind='bar')
    axes[2].set_title("Amount of people who had x views and have transactions", fontsize=16)
    axes[2].set_xlabel("Number of views", fontsize=13)
    axes[2].set_ylabel("Amount of people", fontsize=13) 
    
    plt.show()

charts_max_k_views_bought_something(10, 20)



