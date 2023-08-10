import os
import glob 
from datetime import datetime,date

import requests
from bs4 import BeautifulSoup
import re
import nltk
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
import pickle
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from IPython.display import display

from utils import *


# Figures inline and set visualization style
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()
sns.set_style("whitegrid")

pathSEG='./data/'

url='http://library.seg.org/loi/gpysa7' # List of volumes

# Make the request 
r = requests.get(url)

# Extract HTML from Response object and print
html = r.text
#print html

# Create a BeautifulSoup object from the HTML
soup = BeautifulSoup(html, "html5lib")

# Create tokenizer to find weblinks for all volumes of Geophysics
tokenizer = RegexpTokenizer('"((http)s?://library.seg.org/toc/gpysa7/[0-9].*?)"')
volumes = tokenizer.tokenize(html)

# Remove first volume as it contains articles that have just been accepted.
volumes = volumes[1:] 

print('Number of Geophysics Volumes: %d ' % len(volumes))
#print volumes

volume = 'https://library.seg.org/toc/gpysa7/82/2'

r    = requests.get(volume)
html = r.text
cat  = find_categories(r.text)

r = requests.get('https://doi.org/10.1190/geo2016-0138.1')

html = r.text
soup = BeautifulSoup(html, "html5lib")
info = soup.findAll('meta')
#print info

# authors
author = filter(lambda x: 'dc.Creator' in str(x), info)
#print author
author  = map(lambda x: str(x).split('"')[1].decode('utf8'), author)
print('Authors:',author)

# keywords
keywords = filter(lambda x: 'dc.Subject' in str(x), info)
#print keywords
keywords = map(lambda x: str(x).split('"')[1].decode('utf8'), keywords)
#print keywords
keywords = map(lambda x: str(x).split(';'), keywords)[0]
print('Keywords:',keywords)


# abstract
abstract = filter(lambda x: 'dc.Description' in str(x), info)
#print abstract
abstract = map(lambda x: str(x).split('"')[1].decode('utf8'), abstract)[0][8:]
print abstract
print('Abstract:',abstract)


# publication history
info = soup.findAll(text=re.compile("Received:|Accepted:|Published:"))
print info
received, accepted, published = get_pubhistory(info)
print received, accepted, published

# countries
info = soup.findAll('span')
country = filter(lambda x: 'country' in str(x), info)
country = map(lambda x: str(x).split('>')[1].split('<')[0].decode('utf8'), country)
print country
print('Country:',country)


# affiliations
info = soup.findAll('span')
affiliation = filter(lambda x: 'class="institution"' in str(x), info)
print affiliation
affiliation = map(lambda x: str(x).split('>')[1].split('<')[0].decode('utf8'), affiliation)
print affiliation
print('Affiliation:',affiliation)


# citations
info = soup.findAll('div', { "class" : "citedByEntry" })
ncitations = len(info)
print('Ncitations:',ncitations)

scrapedvolumes = ['82']  # list of volumes to scrape
ndois          = -1      # number of dois to process, if -1 all dois


for scrapedvolume in scrapedvolumes:

    selvolumes = filter(lambda x: scrapedvolume in str(x), [volume[0] for volume in volumes])
    print ('Selected volumes %s' % selvolumes)

    for ivolume,volume in enumerate(selvolumes):

        print('Volume %s' % volume)

        # Create folder to save useful info
        vol, issue = volume.split('/')[-2:]

        folder='/'.join(volume.split('/')[-2:]) 
        if not os.path.exists(pathSEG+folder):
            os.makedirs(pathSEG+folder)

        # Initialize containers
        df_seg    = pd.DataFrame()
        titles    = []
        authors   = []
        countries = []
        affiliations = []
        keywords  = []
        abstracts = []

        # make request
        r = requests.get(volume)
        html = r.text
        #print html

        # find categories for each doi
        categories = find_categories(html)

        # find all dois
        dois = re.findall('"((https)s?://doi.*?)"', html)
        #print dois

        # remove first doi as it is ' This issue of Geophysics '
        #dois = dois[1:]
        dois = dois[:len(categories)]

        # loop over dois and extract info
        for idoi, doi in enumerate(dois[:ndois]):

            # sleep for some time to avoid being found web scraping ;)
            time_sleep=np.round(
                np.random.uniform(0,10))
            print('Sleep for %d' % time_sleep)
            time.sleep(time_sleep)
            
            # Rearrange doi to work also with old volumes
            doi = '/'.join(['http://library.seg.org/doi/abs','/'.join(doi[0].split('/')[-2:])])
            print('DOI %s' % doi)
            
            # Make the request 
            r = requests.get(doi)

            # Extract HTML from Response object
            html = r.text
            #print html

            # Create a BeautifulSoup object from the HTML
            soup = BeautifulSoup(html, "html5lib")


            # GET USEFUL INFO #
            info    = soup.findAll('meta')
            infopub = soup.findAll(text=re.compile("Received:|Accepted:|Published:"))
            infoaff = soup.findAll('span')


            # Get title
            title = soup.title.string.split('GEOPHYSICS')[0][18:-3]
            print('Title: %s' % title)
            titles.append(title)

            # Get category
            category = categories[idoi]
            print('Category: %s' % category)


            # Get authors
            author    = filter(lambda x: 'dc.Creator' in str(x), info)
            author_df = map(lambda x: str(x).split('"')[1], author)
            author    = map(lambda x: str(x).split('"')[1].decode('utf8'), author)

            print('Authors: %s' % author)
            authors.extend(author)


            # Get keywords
            keyword     = filter(lambda x: 'dc.Subject' in str(x), info)
            if len(keyword)>0:
                keyword_df  = map(lambda x: str(x).split('"')[1], keyword)#.decode('utf8')
                keyword     = map(lambda x: str(x).split('"')[1], keyword)
                keyword     = map(lambda x: str(x).split(';'), keyword)[0]
            else:
                keyword_df='-'
                keyword='-'
            print('Keywords: %s' % keyword)
            keywords.extend(keyword)


            # Get abstracts
            abstract = filter(lambda x: 'dc.Description' in str(x), info)
            if len(abstract)>0:
                abstract = map(lambda x: str(x).split('"')[1].decode('utf8'), abstract)[0][8:]
            else:
                abstract='-'
            #print('Abstract: %s' % abstract)
            abstracts.extend(abstract)


            # Get countries
            country    = filter(lambda x: 'country' in str(x), infoaff)
            country_df = map(lambda x: str(x).split('>')[1].split('<')[0], country)
            country    = map(lambda x: str(x).split('>')[1].split('<')[0].decode('utf8'), country)

            print('Countries: %s' % country)
            countries.extend(country)


            # Get affiliations
            affiliation    = filter(lambda x: 'institution' in str(x), infoaff)
            affiliation_df = map(lambda x: str(x).split('>')[1].split('<')[0], affiliation)
            affiliation    = map(lambda x: str(x).split('>')[1].split('<')[0].decode('utf8'), affiliation)

            print('Affiliations: %s' % affiliation)
            affiliations.extend(affiliation)


            # Get publication history
            pubhistory = get_pubhistory(infopub)
            print('Publication history: %s\n' % str(pubhistory))


            # Get number of citations
            citations = soup.findAll('div', { "class" : "citedByEntry" })
            ncitations = len(citations)
            print('Number of citations: %d\n' % ncitations)


            # check that I am not being banned by website...
            #if len(author)==0:
            #    print('Last DOI %s')
            #    raise Exception('No Authors')

            df_seg = df_seg.append(pd.DataFrame({'Title'         : title.encode('utf8'), 
                                                 'Category'      : category.encode('utf8'),
                                                 'Authors'       : ('; ').join(author_df),
                                                 'Countries'     : ('; ').join(country_df),
                                                 'Affiliations'  : ('; ').join(affiliation_df),
                                                 'Keywords'      : keyword_df[0],
                                                 'Received'      : pd.Timestamp(pubhistory[0]),
                                                 'Accepted'      : pd.Timestamp(pubhistory[1]),
                                                 'Published'     : pd.Timestamp(pubhistory[2]),
                                                 'Volume'        : vol,
                                                 'Issue'         : issue,
                                                 'Ncitations'    : ncitations}, index=[0]), ignore_index=True)


        # save dataframe
        df_seg.to_csv(pathSEG+folder+'/df_SEG.csv')

        # loop through titles and get all words
        words_title = words_from_text(titles)
        #print words_title
        #words_title = [x.encode('utf-8') for x in words_title]

        # loop through abstracts and get all words
        words_abstract = words_from_text(abstracts)
        #print words_abstract

        # Save words and authors into pickles
        with open(pathSEG+folder+'/wordstitle_SEG', 'wb') as fp:
            pickle.dump(words_title, fp)

        with open(pathSEG+folder+'/wordsabstract_SEG', 'wb') as fp:
            pickle.dump(words_abstract, fp)

        with open(pathSEG+folder+'/authors_SEG', 'wb') as fp:
            pickle.dump(authors, fp)

        with open(pathSEG+folder+'/countries_SEG', 'wb') as fp:
            pickle.dump(countries, fp)

        with open(pathSEG+folder+'/affiliations_SEG', 'wb') as fp:
            pickle.dump(affiliations, fp)

