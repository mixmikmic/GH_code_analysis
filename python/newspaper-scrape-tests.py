import newspaper
import csv
import pandas as pd

file = '../../data/training_dataset.csv'

def urls_from_csv(csv_file, column=None):
    '''
    Takes csv directory and returns list of URLs
    '''
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        contents = list(reader)
    
    urls = [line[1] for line in contents[1:]]
    return urls, contents

def urls_to_df(csv_file, column=None):
    '''
    Takes csv directory and returns list of URLs
    '''
    df = pd.read_csv(csv_file)
    df.columns = [x.lower() for x in df.columns]
    urls = list(df['url'])
    return urls, df

urls, contents = urls_from_csv(file)

def remove_newline(text):
    ''' Removes new line and &nbsp characters.
    '''
    text = text.replace('\n', ' ')
    text = text.replace('\xa0', ' ')
    return text

urls

def html_report(link, nlp=False):
    report = {}
    a = newspaper.Article(link)
    a.download()
    a.parse()
    report['domain'] = a.source_url
    report['title'] = a.title
    report['authors'] = a.authors
    report['date_pub'] = a.publish_date
    report['text'] = remove_newline(a.text)
    # tag the type of article
    ## currently default to text but should be able to determine img/video etc
    report['type'] = 'text'
    return report

urls, df = urls_to_df(file)

def scrape_from_urls(urls):
    reports = []
    for url in urls:
        if url[-3:] == 'pdf':
            continue
        else:
            report = html_report(url)
            reports.append(report)
            
    return reports

import requests

from newspaper import Config

config = Config()

user = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.81 Safari/537.36'

config.browser_user_agent = user

url = urls.iloc[2]

a = newspaper.Article(url, config)

a.download()

a.title

a.parse()

a.text

result = requests.get(url)
status = result.status_code

status

result.text

keys = report[0].keys()
with open('data.csv', 'w') as f:
    dict_writer = csv.DictWriter(f, fieldnames=keys)
    dict_writer.writeheader()
    dict_writer.writerows(report)

urls = df_failed['URL']

df_600 = pd.read_csv('../../data/input_urls.csv')

from urllib.parse import urlparse

parsed = urlparse('https://docs.python.org/3/library/urllib.parse.html')

def get_domain(url):
    parsed = urlparse(url)
    domain = parsed.hostname
    return domain

df_600['domain'] = df_600['DocumentIdentifier'].map(lambda x: get_domain(x))

sum(df_600['DocumentIdentifier'].str.contains('pdf'))



