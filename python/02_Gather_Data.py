get_ipython().magic('matplotlib inline')
from IPython.core.pylabtools import figsize
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
plt.style.use('ggplot')
figsize(11,9)

import scipy.stats as stats

import pymc as pm

git_logs_filename = 'data/popular_open_source_logs.csv'
df = pd.read_csv(git_logs_filename)

alternate_posts_filename = "data/alt_posts-2016-06-06.csv"
alternate_posts = pd.read_csv(alternate_posts_filename)
alternate_posts.head()

import requests

from os.path import join, dirname
import os
from dotenv import load_dotenv, find_dotenv

dotenv_path = join(dirname('__file__'), '.env')
load_dotenv(dotenv_path)

API_KEY = os.environ.get("API_KEY")

# This is a County Business Patterns API endpoint
url = "http://api.census.gov/data/2014/cbp?key=%s&get=EMP,ESTAB,EMPSZES,EMPSZES_TTL,PAYANN&for=state:*" % (API_KEY)
result = requests.get(url)
result.reason
cbp = None
if result.ok:
    data = result.json()
    cbp = pd.DataFrame(data[1:], columns=data[0])
print(result.reason)

cbp.head()

columns = ['timestamp', 'project', 'email', 'lines_inserted', 'lines_removed']
git_logs = pd.read_csv(git_logs_filename, index_col='timestamp', usecols=columns)
git_logs.head()

git_logs.sort_values(by='project', ascending=False, inplace=True)
git_logs.head()

git_logs.dtypes

posts_filename = "data/posts-2016-06-08-21-35-42.csv"
columns = ['Author', 'Time', 'Text', 'ProfileUrl', 'PostUrl', 'Lang',
           'Sentiment']
posts = pd.read_csv(posts_filename, parse_dates=['Time'], index_col='Time', usecols=columns)
posts['HourOfDay'] = posts.index.hour
posts['DayOfWeek'] = posts.index.dayofweek
posts.head()

posts.dtypes



