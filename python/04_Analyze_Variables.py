get_ipython().magic('matplotlib inline')
from IPython.core.pylabtools import figsize
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
plt.style.use('ggplot')
figsize(11,9)

import scipy.stats as stats

import pymc as pm

import requests

from os.path import join, dirname
import os
from dotenv import load_dotenv, find_dotenv

dotenv_path = join(dirname('__file__'), '.env')
load_dotenv(dotenv_path)

API_KEY = os.environ.get("API_KEY")

git_logs_filename = 'data/popular_open_source_logs.csv'
columns = ['timestamp', 'project', 'email', 'lines_inserted', 'lines_removed']
git_logs = pd.read_csv(git_logs_filename, index_col='timestamp', usecols=columns)

posts_filename = "data/posts-2016-06-08-21-35-42.csv"
columns = ['Author', 'Time', 'Text', 'ProfileUrl', 'PostUrl', 'Lang',
           'Sentiment']
posts = pd.read_csv(posts_filename, parse_dates=['Time'], index_col='Time', usecols=columns)
posts['HourOfDay'] = posts.index.hour
posts['DayOfWeek'] = posts.index.dayofweek

alternate_posts_filename = "data/alt_posts-2016-06-06.csv"
columns = ['UniversalMessageId', 'SenderUserId', 'Title', 'Message',
           'CreatedTime', 'Language', 'LanguageCode', 'CountryCode',
           'MediaTypeList', 'Permalink', 'Domain', 'Spam', 'Action Time', 'Location']
alternate_posts = pd.read_csv(alternate_posts_filename,
                              usecols=columns,
                              index_col='CreatedTime',
                              parse_dates=['CreatedTime'])
alternate_posts.MediaTypeList.fillna(value='Unknown', inplace=True)
alternate_posts.SenderUserId.fillna(value='Unknown', inplace=True)

# This is a County Business Patterns API endpoint
url = "http://api.census.gov/data/2014/cbp?key=%s&get=EMP,ESTAB,EMPSZES,EMPSZES_TTL,PAYANN&for=state:*" % (API_KEY)
result = requests.get(url)
result.reason
cbp = None
if result.ok:
    data = result.json()
    cbp = pd.DataFrame(data[1:], columns=data[0])
print(result.reason)

git_logs.dtypes

git_logs.head()

git_logs['lines_diff'] = git_logs['lines_inserted'].shift(-1) - git_logs['lines_removed']
committer_group = git_logs.groupby(['email'])
lines_committed = pd.DataFrame(data=committer_group['lines_diff'].count())

lines_committed.describe()

lines_committed.lines_diff.hist()

boundary = lines_committed.lines_diff.quantile(0.80)
lower_portion = lines_committed.loc[lines_committed.lines_diff <= boundary]
lower_portion.describe()

lower_portion.hist()

upper_portion = lines_committed.loc[lines_committed.lines_diff > boundary]
upper_portion.describe()

log_count = lines_committed.lines_diff.apply(np.log10)
log_count.hist()
log_count.describe()

lines_committed.boxplot(return_type='axes')

git_logs.columns

project_group = git_logs.groupby('project')
commit_counts = project_group.project.apply(lambda x: x.count())
commit_count_portion = list(map(lambda x: x / commit_counts.sum(), commit_counts))
count_df = pd.DataFrame(data={'commit_count': commit_counts, 'proportion': commit_count_portion})
count_df

count_df.proportion.sort_values(ascending=False).plot(kind='bar')

pd.tools.plotting.scatter_matrix(posts, alpha=0.2)
True

pd.tools.plotting.scatter_matrix(git_logs, alpha=0.2, diagonal='kde')
git_logs.head()



