import requests
from bs4 import BeautifulSoup
import pandas as pd

# I do not understand, why this doesnt work.

from urllib.request import urlopen
html_str = urlopen("http://research.domaintools.com/statistics/tld-counts/").read()

#Stack overflow answers regarding this problem:
#http://stackoverflow.com/questions/2233687/overriding-urllib2-httperror-and-reading-response-html-anyway
#http://stackoverflow.com/questions/15912257/500-error-with-urllib-request-urlopen

#How to read this?

from urllib.request import urlopen
from bs4 import BeautifulSoup
from urllib.error import HTTPError

try:
    html = urlopen("http://research.domaintools.com/statistics/tld-counts/")
    html2 = BeautifulSoup(html, "html.parser")
except urllib.error.HTTPError as e:
    print(e.code)
    if html is None:
        print("url is not found")

BeautifulSoup(html, "html.parser")

html_str = open('data/view-source_research.domaintools.com_statistics_tld-counts_.htm', 'r').read()

html_str = BeautifulSoup(html_str, "html.parser")

domains = html_str.find_all('div', {'class': 'stats-container'})

import re
index = 0
domain_list = []
for domain in domains:
    all_domain_names = domain.find_all('td', {'class': 'name'})
    for item in all_domain_names:
        item = item.string
        item = item.replace(r"\(\.+\)$", '')
        index = index + 1
        domain_dict = {'index': index, 'name': item}
        #print(item)
        domain_list.append(domain_dict)

domain_count_list = []
index = 0
for count in domains:
    all_domain_counts = domain.find_all('td', {'class': 'amount'})
    for item in all_domain_counts:
        item = item.string
        item = item.replace(",", "")
        if item == "N/A":
            item = '0'
        item = int(item)
        index = index + 1
        count_dict = {'index': index, 'count': item}
        domain_count_list.append(count_dict)

#domain_count_list

#{a:1, b:1} {a:1, c:1} into {a:1, b:1, c1}

df = pd.DataFrame(domain_list)
df2 = pd.DataFrame(domain_count_list)
df_TLD_count = df.merge(df2, left_on='index', right_on='index')
df_TLD_count.drop('index', axis=1)
df_TLD_count.head()

Swiss_Code = df_TLD_count[df_TLD_count['name'] == '.ch']
Swiss_Code

df_TLD_count.to_csv('160812_Top_Level_Domains')















