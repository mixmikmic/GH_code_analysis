import os
import xml.etree.ElementTree
import shelve
import logging
import json
import lzma

import pandas
import requests

path = os.path.join('data', 'issn.tsv')
issn_df = pandas.read_table(path)
issn_df.head(3)

issns = sorted(set(issn_df.issn))
len(issns)

def get_response_text(issn, cache={}):
    if issn in cache:
        return cache[issn]
    url = f'https://api.elsevier.com/content/serial/title/issn/{issn}'
    params = {
        'httpAccept': 'text/xml',
    }
    response = requests.get(url, params)
    if not response.ok:
        logging.info(f'{response.url} returned {response.status_code}:\n{response.text}')
    text = response.text
    cache[issn] = text
    return text

def get_homepage(text):
    tree = xml.etree.ElementTree.fromstring(text)
    elem = tree.find('entry/link[@ref="homepage"]')
    href = None if elem is None else elem.get('href')
    return href

path = os.path.join('data', 'homepages', 'issn-scopus-api.shelve')
cache = shelve.open(path, protocol=4)

issn_to_url = dict()
for issn in issns:
    text = get_response_text(issn, cache)
    if not text:
        continue
    url = get_homepage(text)
    if not url:
        continue
    issn_to_url[issn] = url
len(issn_to_url)

len(cache)

cache_to_export = dict(cache)

cache.close()

issn_homepage_df = pandas.DataFrame.from_records(list(issn_to_url.items()), columns=['issn', 'homepage'])
issn_homepage_df.head(2)

scopus_homepage_df = issn_df.merge(issn_homepage_df)
scopus_homepage_df = scopus_homepage_df[['scopus_id', 'homepage']].drop_duplicates()
scopus_homepage_df.head(3)

len(scopus_homepage_df)

# Journals with multiple homepage URLs
scopus_homepage_df[scopus_homepage_df.duplicated(keep=False)]

path = os.path.join('data', 'homepages', 'issn-homepages.tsv')
issn_homepage_df.to_csv(path, sep='\t', index=False)

path = os.path.join('data', 'homepages', 'scopus-homepages.tsv')
scopus_homepage_df.to_csv(path, sep='\t', index=False)

path = os.path.join('data', 'homepages', 'issn-scopus-api.json.xz')
with lzma.open(path, 'wt') as write_file:
    json.dump(cache_to_export, write_file, indent=2, sort_keys=True)

