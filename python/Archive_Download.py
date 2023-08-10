import requests
import os
import json
from pathlib import Path

article_search_key = os.getenv('nyt_article_search_key')
print(article_search_key)

import time

p = Path(os.getcwd())
archive_path = str(p.parent) + '/data/article_search/2010/'
page = 0

while page < 101:
    raw_data = requests.get('http://developer.nytimes.com/proxy/https/api.nytimes.com/svc/search/v2/articlesearch.json?api-key='+article_search_key+'&q=China&begin_date=20100101&end_date=20101231&page='+str(page))
    with open(archive_path+str(page)+'_a_s_2010.txt', 'w') as f:
        f.write(json.dumps(raw_data.json()))
    page += 1
    time.sleep(2)



