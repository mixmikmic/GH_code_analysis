# 11/5 Get Raw Player Bios
# Cameron Yick

# For data manipulation
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
PROJ_ROOT = os.pardir

# For Scraping
from bs4 import BeautifulSoup
import requests
import time

# For munging
import re
import json

# Backoff time for large scrapes
THROTTLE_TIME = .05

# ipython magics
# %load_ext watermark
get_ipython().magic('matplotlib inline')
# a nice bonus for making progress bars in ipython

from tqdm import tqdm, tqdm_pandas, tqdm_notebook

# Initialize reused objects
tqdm.pandas()
s= requests.Session()
s.headers.update({'User-agent': 'Mozilla/5.0'})

# playerPath = os.path.join(PROJ_ROOT, "data", "interim", "players.json")

# use with temp auth token

ghPath = "https://raw.githubusercontent.com/hydrosquall/yale_athletics_data/master/data/interim/harvard_players.json?token=AImmM-Q_QprW_hYT6_KwnNcP21It4pS7ks5YPkJ2wA%3D%3D"

# use online read path for usage with a distributed crawler
df = pd.read_json(ghPath)

df.info()

# There is are 10806 unique bio links... hm. Scrape now, clean later.
df['Name_link'].unique().shape

df.duplicated(subset=['Name_link'],keep=False).sum()

# how did this guy show up twice?!
df[df['Name_link'] == "/sports/bsb/2008-09/bios/albright_tyler"]

df[df.duplicated(subset=['Name_link'],keep=False)].head()

df['Name_link'].head()

# Let's make some data soup!

def makeAthleteSoup(link, session=s):
    if not link:
        return None
    time.sleep(THROTTLE_TIME)
    result = session.get("http://www.gocrimson.com" + link)
    if result.status_code is 200:
        return BeautifulSoup(result.content, 'lxml').find('div', class_="bio-wrap")
    else:
        print("Site is down!")

bios = df['Name_link'].unique()

df['Name_link'].head()

bios = pd.Series(bios)

soups = bios.progress_apply(lambda x: makeAthleteSoup(x))

# the join key is the bio URL
soups.index = bios
soups.head()

pd.DataFrame(soups).to_csv('harvard_bios.csv', encoding="utf-8")

get_ipython().system('du harvard_bios.csv')

get_ipython().system('gzip -c harvard_bios.csv > harvard_bios.csv.gz')

get_ipython().system('du harvard_bios.csv.gz')

d = pd.read_csv('harvard_bios.csv.gz', compression='gzip')

d.info()

type(d['0'].iloc[4])

# check the lengths of these bio fields
test = d['0'].apply(lambda x: len(x) if (type(x) == str) else 0)
plt.plot(test)

