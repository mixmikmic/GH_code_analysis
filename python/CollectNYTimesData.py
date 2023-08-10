#Script calls the API and retrieve URLs for the articles related to gun violence

import re
import sys
import time
import requests
from lucenequerybuilder import Q

urls = list()
regexp = re.compile(r'query')	#regex for urls containg the word query

hits = sys.maxsize
for i in range(1, 2):
    search_query = Q(Q('gun') | Q('gunman'))
    # DATE IS IN YYYYMMDD FORMAT
    parameters = {"api-key": "API_KEY",
                  "q": search_query,
                  "glocations": "United States",
                  "fl": "web_url",
                  "page": i,
                  "begin_date": "20180321",
                  "end_date": "20180329"}
    time.sleep(1)	# delaying api calls due to api time restrictions
    response = requests.get("https://api.nytimes.com/svc/search/v2/articlesearch.json", params=parameters)
    data = response.json()
    try:
        #Getting the number of hits
        hits = data['response']['meta']['hits']
        # print(hits)
    except KeyError:
        pass

    try:
        # print(data)
        for x in data['response']['docs']:
            if regexp.search(x['web_url']):
                print("regex")
                continue

            if '/world/' not in x['web_url'] and '/sports/' not in x['web_url'] and '/video/' not in x['web_url']: #removing unnecessary urls
                urls.append(x['web_url'])
                print(x['web_url'])
    except KeyError:
        print("error")
        pass
    if data['response']['meta']['offset'] > data['response']['meta']['hits']: break # NO more articles available

# SAVING TO FILE
urls_file = open("nytimes_data\\sample_url.txt", "a+")
for i in urls:
    urls_file.write(i + "\n")
urls_file.close()

#Script will retrieve the articles for the URLs collected and parses to get the article body 

import urllib.request
import os
from bs4 import BeautifulSoup

directory = "nytimes_data\\sample_data"
if not os.path.exists(directory):
    os.makedirs(directory)

urls = open("nytimes_data\\sample_url.txt").read()
urls = urls.splitlines()

for index, url in enumerate(urls):
    content = urllib.request.urlopen(url).read()
    soup = BeautifulSoup(content, "html.parser")
    # Opening the file
    file = open('nytimes_data\\sample_data\\ny_article_' + str(index + 1) + ".txt", "w+")

    # Getting the article body and saving to the file
    for a in soup.find_all('p', {'class': 'story-body-text story-content'}):
        file.write(a.text)
    for a in soup.find_all('p', {'class': 'g-body'}):
        file.write(a.text)
    for a in soup.find_all('p', {'class': 'css-1xyeyil e2kc3sl0'}):
        file.write(a.text)
    file.close()
    
print ("Files downloaded")

