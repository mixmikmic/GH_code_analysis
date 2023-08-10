from pymongo import MongoClient
from requests.auth import HTTPProxyAuth
from nauk_page import get_data
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')

client = MongoClient()
db = client['jobs_database']
coll = db['naukri_jobs']
#coll.drop
print("No of rows in the dataset = {}".format(coll.count()))

cursor = coll.find()
df = pd.DataFrame(list(cursor))

print(df.head())

print(df.dtypes, df.shape)

l = df['Experience']

import re
data_in_str = [re.sub("[^0-9-]","",p) for p in l] 

small_list = []
exp_list_cleaned = []

print(data_in_str[0])
a = data_in_str[0].split('-')
b = a[0]
c = a[1]
print(a,b,c)

exp_list_cleaned = []
for p in data_in_str:
    a = p.split('-')
    b = a[0]
    c = a[1]
    exp_list_cleaned.append([int(b), int(c)])

print(len(exp_list_cleaned))
print(type(exp_list_cleaned))
print((exp_list_cleaned[0]))

df['clean_exp'] = exp_list_cleaned

df['freq_exp'] = df.groupby('Experience')['Experience'].transform('count')

df.head()

dic_exp = {}

for i in range(len(df)):
    tup = tuple(exp_list_cleaned[i])
    freq = df['freq_exp'][i]
    dic_exp.update({tup:freq})

print(len(dic_exp))
print(dic_exp)

print(dic_exp.keys())
list_keys = list(dic_exp.keys())

list_freq = list(dic_exp.values())

avg = [int((l[0]+l[1])/2) for l in list_keys]

#0-2, 2-5, 5-8, 8-13, 13-20
#The above are the ranges in which the groups will be divided.

range1 , range2, range3, range4, range5 = 0, 0, 0, 0, 0
for i in range(len(avg)):
    if avg[i]>=0 and avg[i]<=2:
        range1 = range1 + list_freq[i]
    if avg[i]>=2 and avg[i]<=5:
        range2 = range2 + list_freq[i]
    if avg[i]>=5 and avg[i]<=8:
        range3 = range3 + list_freq[i]
    if avg[i]>=8 and avg[i]<=13:
        range4 = range4 + list_freq[i]
    if avg[i]>=13:
        range5 = range5 + list_freq[i]
        
print(range1, range2, range3, range4, range5)

y_exp = []
y_exp.append(range1), y_exp.append(range2), y_exp.append(range3), y_exp.append(range4)
y_exp.append(range5)

name_list = ["0-2","2-5","5-8","8-13",">13"]
p = [i for i in range(len(y_exp))]

import matplotlib.pyplot as plt
x = plt.bar(p, y_exp, align='center', alpha = 0.5, color = 'red')
plt.xticks(p, name_list)
plt.xlabel('Average Experience')
plt.ylabel('No. of jobs')
plt.show()

colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'pink']
plt.pie(y_exp, labels=name_list, colors=colors,  autopct='%1.1f%%')
plt.axis('equal')
plt.show()

df.head()

del df['freq_exp']
del df['clean_exp']

df.head()

dic_location = {}

df['freq_loc'] = df.groupby('Location')['Location'].transform('count')
for i in range(len(df)):
    loc = df['Location'][i]
    freq = df['freq_loc'][i]
    dic_location.update({loc: freq})

loc_list_new = list(dic_location.keys())
loc_freq_new = list(dic_location.values())

import operator
sorted_dic = sorted(dic_location.items(), key = operator.itemgetter(1), reverse=True)

def get_top_n(sorted_dictionary, n, list_key, list_val):
    for i in range(n):
        list_key.append(sorted_dictionary[i][0])
    for i in range(n):
        list_val.append(sorted_dictionary[i][1])
    return list_key, list_val

list_key = []
list_val = []
loc_key, loc_val = get_top_n(sorted_dic, 8, list_key, list_val)

plt.figure(figsize = (13,4))
p = [i for i in range(len(loc_key))]
x = plt.bar(p, loc_val, align='center', alpha = 0.5, color = 'purple')
plt.xticks(p, loc_key)
plt.xlabel('Location')
plt.ylabel('No of jobs')
plt.show()

x = list(df['Requirements'])
from collections import defaultdict
dic = defaultdict(int)

for i in range(len(x)):
    for req in x[i]:
        dic[req] += 1

req_sorted = sorted(dic.items(), key = operator.itemgetter(1), reverse=True)

key_req = []
val_req = []

key_req, val_req = get_top_n(req_sorted, 10, key_req, val_req)

plt.figure(figsize = (13,4))
p = [i for i in range(len(key_req))]
x = plt.bar(p, val_req, align = 'center' , alpha = 0.5)
plt.xticks(p, key_req)
plt.xlabel('Skills')
plt.ylabel('Number of jobs')
plt.show()



