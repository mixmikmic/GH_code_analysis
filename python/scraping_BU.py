import numpy as np
import pandas as pd
import string

bu_words = ["again",
"also",
"ask",
"all",
"because",
"boy",
"but",
"can",
"come",
"deaf",
"different",
"done",
"drink",
"drive",
"eat",
"speak",
"speech (as in giving a speech)",
"excuse me",
"family",
"father",
"few",
"fine",
"fingerspell",
"food",
"for",
"friend",
"get",
"girl",
"give",
"good",
"have",
"he",
"hearing",
"hello",
"help",
"home",
"how",
"if",
"Internet",
"know",
"like (feeling)",
"little",
"live",
"locate",
"man",
"many",
"me",
"meet",
"more",
"my",
"name",
"need",
"new",
"no",
"not",
"ok",
"old",
"other",
"people",
"play",
"please",
"remember",
"same",
"say",
"school",
"see",
"she",
"should",
"signing",
"slow",
"some",
"sorry",
"take",
"tell",
"texting",
"thank you",
"that",
"their",
"they",
"think",
"time",
"tired",
"toilet",
"understand",
"use",
"wait",
"water",
"what",
"when",
"where",
"which",
"who",
"why",
"will",
"with",
"woman",
"work",
"write",
"yes",
"you",
"your",]

word_url = []
for word in bu_words:
    bu_string = "http://secrets.rutgers.edu/dai/queryPages/search/result.php?type=whole&key=HOW&variant_name=HOW&demonstrator=All&is_main=1"
    word_url.append(bu_string.replace("HOW", word))

az_url = []
for letter in string.ascii_uppercase:
    bu_string = "http://secrets.rutgers.edu/dai/queryPages/search/result.php?type=whole&key=HOW&variant_name=HOW&demonstrator=All&is_main=1"
    az_url.append(bu_string.replace("HOW", letter))

# ONE WAY to do it!
from urllib.request import urlopen
import re

az_all = []
az_len = []

for url in az_url:
    #connect to a URL
    website = urlopen(url)

    #read html code
    html = website.read().decode('utf-8')

    #use re.findall to get all the links)
    linkzz = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', html)
    az_len.append(len(linkzz))
    az_all.append(linkzz)

############# BEAUTIFUL SOUP
###########################
from urllib.request import urlopen
from bs4 import BeautifulSoup

###Extract href for words
word_dl = []
word_len = []
count = 0

for url in word_url:
    
    conn = urlopen(url)
    html = conn.read()
    soup = BeautifulSoup(html, 'lxml')
    word_links = soup.find_all('a')
    word_len.append(len(word_links))
    
    for tag in word_links:
        link = tag.get('href',None)
        #if link is not None:
        word_dl.append((link, bu_words[count]))
    
    count += 1

#remove initial characters

final_words  =[(word[1], word[0][22:]) for word in word_dl]

#check all are 'mov'

check=[]
for word in final_words:
    check.append((word[1]).endswith('mov'))
    
sum(check) == len(final_words)

###Extract href for letters
az_dl = []
az_len = []
letters = string.ascii_uppercase
count = 0

for url in az_url:
    
    conn = urlopen(url)
    html = conn.read()
    soup = BeautifulSoup(html, 'lxml')
    az_links = soup.find_all('a')
    az_len.append(len(az_links))
    
    for tag in az_links:
        link = tag.get('href',None)
        #if link is not None:
        az_dl.append((link, letters[count]))
        

        count += 1

final_az  =[(letter[1], letter[0][22:]) for letter in az_dl]

final_az

#check all are 'mov'

check=[]
for letter in final_az:
    check.append(letter.endswith('mov'))
    
sum(check) == len(final_az)

get_ipython().system('pip install tqdm')

import os
import logging
import hashlib
from io import BytesIO

from PIL import Image
import requests
from urllib.request import urlretrieve

logger = logging.getLogger(__name__)


def hash_url(url):
    hash_object = hashlib.md5(url.encode())
    return hash_object.hexdigest()


def download_image(url, path, timeout=0.5):
    filename = os.path.join(path, hash_url(url) + ".mov")
    if not os.path.exists(filename):
        try:
            path, _ = urlretrieve(url, filename)
        except Exception as e:
            print('Error: {}'.format(e))
            print('For url: {}'.format(url))
            path = ''
    else:
        print("duplicate, skipping!")
    return path

# Download letters

from tqdm import tqdm_notebook 
import os
for l, url in tqdm_notebook(final_az):
    path = os.path.join("../train_data/asllvd/", l)
    os.makedirs(path, exist_ok=True)
    download_image(url, path)
# Download letters

# Download words

from tqdm import tqdm_notebook 
import os
for l, url in tqdm_notebook(final_words):
    path = os.path.join("../train_data/asllvd/", l)
    os.makedirs(path, exist_ok=True)
    download_image(url, path)

