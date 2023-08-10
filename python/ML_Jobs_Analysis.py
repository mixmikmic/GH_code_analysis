import pandas as pd
from pandas import DataFrame
import cPickle as pickle
from collections import defaultdict
import re

with open('naukri_dataframe.pkl', 'r') as f:
    naukri_df = pickle.load(f)   

naukri_df.head(3)

naukri_df['Location'].value_counts()[:10]

print naukri_df.ix[499,'Location']

uniq_locs = set()
for loc in naukri_df['Location']:
    uniq_locs = uniq_locs.union(set(loc.split(',')))
    
uniq_locs = set([item.strip() for item in uniq_locs])

locations_str = '|'.join(naukri_df['Location']) # All locations into a single string for pattern matchings 
loc_dict = defaultdict(int)
for loc in uniq_locs:
    loc_dict[loc] = len(re.findall(loc, locations_str))

# Take the top 10 most frequent job locations
jobs_by_loc = pd.Series(loc_dict).sort_values(ascending = False)[:10]

print jobs_by_loc

jobs_by_loc['Bengaluru'] = jobs_by_loc['Bengaluru'] + jobs_by_loc['Bengaluru / Bangalore'] 
jobs_by_loc['Delhi NCR'] = jobs_by_loc['Delhi NCR'] + jobs_by_loc['Delhi'] + jobs_by_loc['Noida'] + jobs_by_loc['Gurgaon'] 
jobs_by_loc.drop(['Bengaluru / Bangalore','Delhi','Noida','Gurgaon'], inplace=True)
jobs_by_loc.sort_values(ascending = False, inplace=True)
print jobs_by_loc

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
sns.set_style("darkgrid")

bar_plot = sns.barplot(y=jobs_by_loc.index,x=jobs_by_loc.values,
                        palette="muted",orient = 'h')                        
plt.title("Machine Learning Jobs by Location")
plt.show()

jobs_by_companies = naukri_df['Company Name'].value_counts()[:10]

bar_plot = sns.barplot(y=jobs_by_companies.index,x=jobs_by_companies.values,
                        palette="YlGnBu",orient = 'h')
plt.title("Machine Learning Jobs by Companies")
plt.show()

salary_list = []
exp_list = []
for i in range(len(naukri_df['Salary'])):
    salary = naukri_df.ix[i, 'Salary']
    exp = naukri_df.ix[i, 'Experience']
    if 'INR' in salary:
        salary_list.append((int(re.sub(',','',salary.split("-")[0].split("  ")[1])) + int(re.sub(',','',salary.split("-")[1].split(" ")[1])))/2.0)
        exp_list.append((int(exp.split("-")[0]) + int(exp.split("-")[1].split(" ")[1]))/2.0)
    i+=1

plot_data = pd.DataFrame({'Experience':exp_list,'Salary':salary_list})

sns.jointplot(x = 'Experience', y = 'Salary', data=plot_data, kind='reg', color='maroon')
plt.ylim((0,6000000))
plt.xlim((0,16))
plt.show()

import nltk
from nltk.tokenize import word_tokenize

from collections import Counter

tokens = [word_tokenize(item) for item in naukri_df['Doctorate'] if 'Ph.D' in item]
jobs_by_phd = pd.Series(Counter([item for sublist in tokens for item in sublist if len(item) > 4])).sort_values(ascending = False)[:8]
bar_plot = sns.barplot(y=jobs_by_phd.index,x=jobs_by_phd.values,
                        palette="BuGn",orient = 'h')
plt.title("Machine Learning Jobs PhD Specializations")
plt.show()

skills = pd.Series(Counter('|'.join(naukri_df['Skills']).split('|'))).sort_values(ascending = False)[:25]
sns.color_palette("OrRd", 10)
bar_plot = sns.barplot(y=skills.values,x=skills.index,orient = 'v')
plt.xticks(rotation=90)
plt.title("Machine Learning In-Demand Skill Sets")
plt.show()

from wordcloud import WordCloud, STOPWORDS

jd_string = ' '.join(naukri_df['Job Description'])

wordcloud = WordCloud(font_path='/home/hareesh/Github/naukri-web-scraping/Microsoft_Sans_Serif.ttf',
                          stopwords=STOPWORDS,background_color='white', height = 1500, width = 2000).generate(jd_string)

plt.figure(figsize=(10,15))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()



