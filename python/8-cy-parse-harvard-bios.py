# Use

from bs4 import BeautifulSoup
import pandas as pd
import re
from tqdm import tqdm, tqdm_pandas, tqdm_notebook

tqdm.pandas()

get_ipython().system('ls | grep bios')


df = pd.read_csv('../data/raw/harvard_bios.csv.gz', compression='gzip')
df.info()

df.head()

# the output needs to be handled 1 at a time, and then dumped out carefully.
# df['soup'] = df['0'].progress_apply(lambda x: BeautifulSoup(x, 'lxml') )

# tSoup = df['0'].head().progress_apply(lambda x: BeautifulSoup(x, 'lxml') )

# ts = tSoup.iloc[0]

import json

def tableToDict(soup):
    "Given a headerless table, return a dict with all the fields"
    table = {}
    for row in soup.find_all('tr'):
        cells = row.find_all('td')
        table[str(cells[0].text.strip())] = cells[1].text.strip()
        
    return table
    
def soupToDict(soup):
    "Given a player's soup blob, parse their header"
    playerData = tableToDict(soup.select('div.player-info')[0].find('table'))
    playerData['name'] = soup.select('div.player-name span.name')[0].text.strip()
    return playerData
def getSynopsis(soup):
    "Given a player's soup, split their bio information into     chunks for each strong header"
    
    # Get all top level children
#     bioPs = soup.select('div.synopsis')[0].find_all('p', recursive=False)
    
#     blocks= []
#     # if their blocks are malformed, skip that block.
#     for i, val in enumerate(bioPs):
#         try:
#             blocks[val.select('strong')[0].text] = val.get_text()
#         except:
#             printf("Text not detected")
        
    
    # Given the variation in blob types, just return
    # single blob for now.
    # Ryan Brenner: b instead of strong tags
    # Jackson Stallings (Junior Year) - UL blobs mixed with P tags
    # Jackson Stallings (Freshman) - Clean example
    
    # Sometimes, no bio
#     http://yalebulldogs.com/sports/c-sail/2016-17/bios/buehler_patrick_nu3o?view=news
    bio = soup.find('div',class_='synopsis')
    return bio.get_text() if bio else None

# ts.select('div.synopsis')[0].find_all('b')

# getSynopsis(ts)

# soupToDict(ts)

# df['types'] = df['0'].map(type)
# df['types'].describe()

# df['types']

df['headers'] = df['0'].progress_apply(lambda x:                                        json.dumps(soupToDict(BeautifulSoup(x, 'lxml')), encoding='utf-8') if (type(x) == str) else None)

# df['synopsis'] = df['headers'].progress_apply(lambda x: 0)

df['synopsis'] = df['raw'].progress_apply(lambda x: getSynopsis(BeautifulSoup(x, 'lxml')) if (type(x) == str) else "")

# df.columns = ['url', 'raw', 'headers', 'synopsis']

# df.index = df['url']

df.head()

df[['synopsis', 'headers']].to_csv('../data/interim/harvard_bio_parsed.csv', encoding='utf-8')

# df2 = pd.read_csv('../data/interim/harvard_bio_parsed.csv')

# !gzip -c harvard_bio_parsed.csv > harvard_bio_parsed.csv.gz

# if we redo this, batching it in databricks / spark job will save a lot of time.

get_ipython().system('ls -lS | grep parsed')

