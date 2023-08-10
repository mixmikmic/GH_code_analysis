get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

import pandas as pd
import numpy as np
import gensim
import sklearn
import nltk
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

import os
import sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

df_training = pd.read_csv('../../data/training_dataset.csv')

df_training.head(3)

from internal_displacement.scraper import Scraper

scraper = Scraper()

def progress(queue, done):
    queue = len(queue)
    done = len(done)
    print(done/queue * 100)

content = []
n=0
for url in df_training['URL']:
    n += 1
    try:
        article = scraper.scrape(url, scrape_pdfs=False)
        text = article[0]
    except:
        text = 'fail'
    print(n)
    content.append(text)

df_training['text'] = content

df_training.head(3)

df_training.to_csv('../../data/idmc_training_data_text_nopdf.csv')

df_cf = pd.read_csv('../../data/crowdflower/cf_1_category_training.csv')
df_cf.drop(['Unnamed: 0', 'title'], axis=1, inplace=True)

df_training = pd.read_csv('../../data/idmc_training_data_text_nopdf.csv')
df_training.drop(['Country_or_region', 'Unnamed: 0'], axis=1, inplace=True)

df_cf.head(1)

df_training.head(1)

mask = df_training['text'].str.len() > 200
df_training = df_training[mask]

df_training['Tag'] = df_training['Tag'].map({'Conflict and violence': 'conflict', 'Disasters': 'disaster'})

df_training = df_training.rename(index=str, columns={"Tag": "category", "URL": "url"})
df_cf = df_cf.rename(index=str, columns={"paragraph": "text"})

df_training = df_training[['category', 'text', 'url']]

df_cf['category'] = df_cf['category'].map({'conflict_or_violence': 'conflict', 'disaster': 'disaster', 'other': 'other'})

df_training_final = pd.concat([df_training, df_cf])

df_training_final['category'].value_counts()

df_training_final.to_csv('../../data/classification_training.csv')

df = pd.read_csv('../../data/classification_training.csv')

import textacy

def check_language(article):
    '''Identify the language of the article content
    and update the article property 'language'
    Parameters
    ----------
    article:        the content of the article:String
    '''
    try:
        language = textacy.text_utils.detect_language(article)
    except ValueError:
        language = 'na'
    return language

langs = [check_language(l) for l in df['text']]

df['langs'] = langs

df = df[df['langs'] == 'en']

len(df)

df.to_csv('../../data/classification_training.csv')



