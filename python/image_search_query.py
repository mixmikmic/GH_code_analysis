import os
import json
import requests
from requests_oauthlib import OAuth1


def get_secret(service):
    """Access local store to load secrets."""
    local = os.getcwd()
    root = os.path.sep.join(local.split(os.path.sep)[:3])
    secret_pth = os.path.join(root, '.ssh', '{}.json'.format(service))
    return secret_pth


def load_secret(service):
    """Load secrets from a local store.

    Args:
        server: str defining server

    Returns:
        dict: storing key: value secrets
    """
    pth = get_secret(service)
    secret = json.load(open(pth))
    return secret

BING_API_KEY = load_secret('bing')
NP_API_KEY, NP_API_SECRET = load_secret('noun_project')

def search_bing_for_image(query):
    """
    Perform a Bing image search.

    Args:
        query: Image search query

    Returns:
        results: List of urls from results
    """
    search_params = {'q': query,
                     'mkt': 'en-us',
                     'safeSearch': 'strict'}
    auth = {'Ocp-Apim-Subscription-Key': BING_API_KEY}
    url = 'https://api.cognitive.microsoft.com/bing/v5.0/images/search'
    r = requests.get(url, params=search_params, headers=auth)
    results = r.json()['value']
    urls = [result['contentUrl'] for result in results]
    return urls

def search_np_for_image(query):
    """
    Perform a Noun Project image search.

    Args:
        query: Image search query

    Returns:
        results: List of image result JSON dicts
    """
    auth = OAuth1(NP_API_KEY, NP_API_SECRET)
    endpoint = 'http://api.thenounproject.com/icons/{}'.format(query)
    params = {'limit_to_public_domain': 1,
              'limit': 5}
    response = requests.get(endpoint, params=params, auth=auth)
    urls = [icon['preview_url'] for icon in response.json()['icons']]
    return urls

print(search_np_for_image('magic')[:3])

print(search_bing_for_image('magic')[:3])

from PIL import Image
import matplotlib.pyplot as plt
import urllib
get_ipython().magic('matplotlib inline')

def view_urls(urls):
    """Display the images found at the provided urls"""
    for i, url in enumerate(urls):
        resp = requests.get(url)
        dat = urllib.request.urlopen(resp.url)
        img = Image.open(dat)
        plt.imshow(img)
        plt.axis('off')
        plt.show()

view_urls(search_bing_for_image('magic')[:3])

view_urls(search_np_for_image('magic')[:3])

import pensieve

book1 = pensieve.Doc('../../corpus/book1.txt', doc_id=1)

from pprint import pprint
from numpy.random import randint

rand = randint(len(book1.paragraphs))
print(book1.paragraphs[rand].text)
pprint(book1.paragraphs[rand].words)

import textacy
import networkx as nx

graph = textacy.network.terms_to_semantic_network(book1.paragraphs[400].spacy_doc)
print(book1.paragraphs[400].text)
textacy.viz.draw_semantic_network(graph);

print(book1.paragraphs[400].text)
textacy.keyterms.textrank(book1.paragraphs[400].spacy_doc)

print(book1.paragraphs[654].text)
textacy.keyterms.textrank(book1.paragraphs[654].spacy_doc)

print(book1.paragraphs[400].text)
textacy.keyterms.sgrank(book1.paragraphs[400].spacy_doc)

print(book1.paragraphs[654].text)
textacy.keyterms.sgrank(book1.paragraphs[654].spacy_doc)

print(book1.paragraphs[400].text)
textacy.keyterms.key_terms_from_semantic_network(book1.paragraphs[400].spacy_doc, ranking_algo='divrank')

print(book1.paragraphs[654].text)
textacy.keyterms.key_terms_from_semantic_network(book1.paragraphs[654].spacy_doc, ranking_algo='divrank')

def build_query(par):
    """
    Use TextRank to find the most important words that aren't character names.
    """
    keyterms = textacy.keyterms.textrank(par.spacy_doc)
    for keyterm, rank in keyterms:
        if keyterm.title() not in par.doc.words['people']:
            return keyterm
    return None

par = book1.paragraphs[randint(len(book1.paragraphs))]
print(par.text)
build_query(par)

def submit_query(query):
    """
    Decide which search engine to use based on the part of speech of the query
    """
    doc = textacy.Doc(query, lang='en')
    try:
        urls = search_np_for_image(query)
    except Exception as e:
        urls = search_bing_for_image(query)
    return urls

par = book1.paragraphs[400]
print(par.text)
query = build_query(par)
print(query)
urls = submit_query(query)
view_urls(urls[:1])



