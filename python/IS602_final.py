import os
import ads
import scipy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import csv

os.environ["ADS_DEV_KEY"] = "kNUoTurJ5TXV9hsw9KQN1k8wH4U0D7Oy0CJoOvyw" # (open('dev_key.txt','r')).read() for text file containing just dev key

ads.config.token = 'ADS_DEV_KEY'

x = raw_input('first search term --> ') # the first term you want to search 

y = raw_input('second search term --> ') # the second term you want to search for comparison

z = raw_input('third search term --> ') # the third term you want to search for comparison

papers1 = list(ads.SearchQuery(title= x, fq="bibgroup:cfa", sort= "pubdate"))

papers2 = list(ads.SearchQuery(title= y, fq="bibgroup:cfa", sort= "pubdate"))

papers3 = list(ads.SearchQuery(title= z, fq="bibgroup:cfa", sort= "pubdate"))

print papers1[0].author
print type(papers1)

# Find the mean number of authors for the whole set of results of cfa papers1
s = []
for i in papers1:
    auth_per_paper1 = [len(i.author)]
    s.append(auth_per_paper1)
avg1 = scipy.mean(s)

# Find the mean number of authors for the whole set of results of cfa papers2
s2 = []
for i in papers2:
    auth_per_paper2 = [len(i.author)]
    s2.append(auth_per_paper2)
avg2 = scipy.mean(s2)

# Find the mean number of authors for the whole set of results of cfa papers3
s3 = []
for i in papers3:
    auth_per_paper3 = [len(i.author)]
    s3.append(auth_per_paper3)
avg3 = scipy.mean(s3)

print "Average number of Authors per topic"
print "==================================="
print x, ": ", avg1
print y, ": ", avg2
print z, ": ", avg3

new_s = [i[0] for i in s]
new_s2 = [i[0] for i in s2]
new_s3 = [i[0] for i in s3]

rows = zip(new_s,new_s2,new_s3)

with open('search.csv', 'wb') as f:
    writer = csv.writer(f)
    for row in rows:
        writer.writerow(row)

results = pd.read_csv('search.csv')
results.columns = [x,y,z]

m_results = pd.melt(results)

m_results.columns = ['Search','Authors']

get_ipython().magic('matplotlib inline')
sns.set_context("poster")
plt.figure(figsize=(7, 8))
v_plot = sns.violinplot(x="Search", y="Authors", data=m_results, palette="Blues_d", scale="count")
v_plot.set_title('Number of Authors per Search Topic')

