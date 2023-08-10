import pickle
import pandas as pd
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

abilify = pickle.load( open( "abilify.p", "rb" ) )
reviews = abilify

for review in reviews:
    # fix ageRange to interval
    try:
        temp_ar = review['ageRange'].split('-')
        review['ageRange'] = [int(temp_ar[0]), int(temp_ar[1])]
    except:
        if type(review['ageRange']) != type([]):
#             print(review['ageRange'])
            continue
            
    # fix medDuration to interval of months **[, ) interval
    try:
        temp_md = review['medDuration'].split('less than ') # "less than 1 month", :"x to less than y years"
        temp_md2 = temp_md[-1].split(' ')
        if temp_md2[-1]== 'years': # x to less than y years
            beg_num = int(temp_md[0].split(' ')[0])*12
            end_num = int(temp_md2[0]) *12
        elif temp_md2[-1] == 'month': # less than 1 month
            beg_num = 0
            end_num = 1
        elif temp_md2[-1] == 'months': # x to y months
            temp_md = review['medDuration'].split(' to ')
            beg_num = int(temp_md[0])
            end_num = int(temp_md[1].split(' ')[0])
        review['medDuration_int'] = [beg_num, end_num]
        review['medDuration'] = str([beg_num, end_num])
    except:
        if type(review['medDuration']) != type([]):
#             print('wait, what?', review['medDuration'])
            continue
        
    for value in ['effectiveness', 'ease_of_use', 'satisfaction', 'genRating','upVotes']:
        try:
            review[value] = int(review[value])
        except:
            continue        

reviews_df = pd.DataFrame(reviews)

import numpy as np

def bins_labels(bins, **kwargs):
    bin_w = (max(bins) - min(bins)) / (len(bins) - 1)
    plt.xticks(np.arange(min(bins)+bin_w/2, max(bins), bin_w), bins, **kwargs)
    plt.xlim(bins[0], bins[-1])


fig, ((ax1, ax2)) = plt.subplots(1,2)
bins = range(8)
bins_labels(bins, fontsize=14)
reviews_df['ease_of_use'].plot.hist(bins = bins, edgecolor="k", ax = ax1)
ax1.set_title('ease of use -abilify')

# reviews_df['ease_of_use'].plot.box()

reviews_df[['ease_of_use', 'upVotes']]

plt.scatter(reviews_df.ease_of_use, reviews_df.upVotes)
plt.xlabel('ease of use score')
plt.ylabel('Up Votes')

meddur = reviews_df.groupby(by = ['medDuration'])
meddur.groups.keys()

conversion_to = {key:ik for ik, key in enumerate(meddur.groups.keys())}
conversion_back = {ik:key for ik, key in enumerate(meddur.groups.keys())}

# reviews_df.groupby('medDuration').ease_of_use.hist(stacked = False)

def hist(x):
    h, e = np.histogram(x.dropna(), range=(0, 10))
    e = e.astype(int)
    return pd.Series(h, zip(e[:-1], e[1:]))

kw = dict(stacked=True, width=1, rot=45)
bins = range(8)
bins_labels(bins, fontsize=14)
reviews_df.groupby('medDuration').ease_of_use.plot.hist(stacked = False, alpha = .6, bins = bins, edgecolor="k", legend = True)#(**kw)



# reviews_df['ease_of_use'].plot.hist(bins = bins, edgecolor="k", ax = ax1)
# ax1.set_title('ease of use -abilify')

def make_hist(tag):
    kw = dict(stacked=True, width=1, rot=45)
    bins = range(8)
    bins_labels(bins, fontsize=14)

#     tag = 'ease_of_use'
    plt.hist(reviews_df[reviews_df["medDuration"]=='[0, 1]'][tag].reset_index(drop=True), alpha=0.6, label='[0, 1]', bins = bins, edgecolor="k")
    plt.hist(reviews_df[reviews_df["medDuration"]=='[1, 6]'][tag].reset_index(drop=True), alpha=0.6, label="[1, 6]", bins = bins, edgecolor="k", width=.35)
    plt.hist(reviews_df[reviews_df["medDuration"]=='[12, 24]'][tag].reset_index(drop=True), alpha=0.6, label="[12, 24]", bins = bins, edgecolor="k", width =.45)
    plt.hist(reviews_df[reviews_df["medDuration"]=='[24, 60]'][tag].reset_index(drop=True), alpha=0.6, label="[24, 60]", bins = bins, edgecolor="k", width=.25)
    plt.hist(reviews_df[reviews_df["medDuration"]=='[60, 120]'][tag].reset_index(drop=True), alpha=0.6, label="[60, 120]", bins = bins, edgecolor="k", width=.85)


    plt.legend(loc = 'best')
    # plt.axes = [[0,1],[0,10]]
    plt.axis([0.5, 6.5, 0, 350])
    plt.xlabel(tag+' rating')
    plt.ylabel('count')
    

make_hist('ease_of_use')

make_hist('satisfaction')

make_hist('effectiveness')



grouped =reviews_df.groupby(['medDuration', 'satisfaction'])
grouped.describe()

coded_meddur = []
satisfaction = []
for ik in range(len(reviews_df['medDuration'])):
    try:
        coded_meddur.append(conversion_to[reviews_df['medDuration'].iloc[ik]])
        satisfaction.append(reviews_df['satisfaction'].iloc[ik])
    except: 
        continue

plt.scatter(coded_meddur,satisfaction, alpha = .03 )
plt.xticks(coded_meddur, [conversion_back[tick] for tick in coded_meddur])
plt.xlabel('medication duration (months)')
plt.ylabel('satisfaction rating')



