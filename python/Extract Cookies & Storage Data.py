import pandas as pd
import json
import sys
sys.path.append('..')
from utils.load_data_util import load_random_data

result = load_random_data(10, seed=42)
result

from urllib.parse import urlparse


def extract_domain_name(data):
  parsed_uri = urlparse( data )
  return '{uri.scheme}://{uri.netloc}/'.format(uri=parsed_uri)


result_cookie = result.loc[(result['symbol'] == "window.document.cookie") & (result['value'].str.contains("="))]
cookiedf = pd.DataFrame(columns=['key', 'value', 'script_url'])
for i, row in result_cookie.iterrows():
    for kv in row['value'].split(";"):
        if "=" in kv:
            keyValueArr = kv.strip().split("=", maxsplit=1)
            cookiedf = cookiedf.append({'key':keyValueArr[0], 'value':keyValueArr[1], 'script_url':row['script_url'], 'location':extract_domain_name(row['location']), 'call_stack':row['call_stack']}, ignore_index=True) 

cookiedf


cookieKey = (cookiedf.groupby(by="key"))
for key, item in cookieKey:
     print(cookieKey.get_group(key), "\n\n")
    

cookieKey.size().reset_index(name='location')

import requests
from bs4 import BeautifulSoup
import sys
import urllib
sys.path.append('..')
from utils.simple_cache import Simple_Json_Cache


def get_cookiepedia_data_by_cookie_key(key): 
    """ returns cookiepedia data or empty string, track empty strings as they are none requests not to make """
    url = 'https://cookiepedia.co.uk/cookies/' + urllib.parse.quote(key)
    page = requests.get(url)
    if page.status_code == 200:
        soup = BeautifulSoup(page.content, 'html.parser')
        node = soup.find(id="content-left")
        if node != None:
            text = node.find('p').get_text()
            if text:
                return text
    return ""

cache = Simple_Json_Cache("cookiepedia")

for key in cookieKey.groups.keys():
    cache_value = cache.get_value(key)
    if not cache_value or cache_value != "": # we assume if cookie value == "" then we have checked cookiepedia and it has no data on the key
        text = get_cookiepedia_data_by_cookie_key(key)
        cache.set_value(key, text)

                    
cache.dump()

# display cookiepedia data
fail_resp_msg = 'There is not yet' # begining of error message when cookiepedia has no data :(
cache = Simple_Json_Cache("cookiepedia")
cache_data = cache.get_cache_file()
filtered_cache = {k:v for (k,v) in cache_data.items() if v.strip() and not v.startswith(fail_resp_msg)}
for (k,v) in filtered_cache.items():
    print(k+": "+v+"\n")

for key, item in cookieKey:
     print(cookieKey.get_group(key).drop_duplicates(subset='script_url'), "\n\n")

test = result.loc[(result['symbol'] == "window.localStorage")]

localStorage = []
for i, row in test.iterrows():
    storageData = json.loads(row['value'])
    localStorage.append({'json': storageData, 'script_url':row['script_url']})

localStorage

from pandas.io.json import json_normalize

storage = []
for row in localStorage:
    if len(row["json"].keys())!=0:
        storage.append(json_normalize(row["json"]))

pd.concat(storage)

