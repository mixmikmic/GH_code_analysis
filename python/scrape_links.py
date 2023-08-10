import time
import os
from bs4 import BeautifulSoup
import requests
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import pandas as pd
browser = webdriver.Chrome()
url = ["https://medium.com/topic/data-science", "https://medium.com/topic/artificial-intelligence",
       "https://medium.com/topic/technology", "https://medium.com/topic/programming"]
browser.get("https://medium.com/topic/data-science")
time.sleep(1)
import ast
from datetime import datetime
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()

all_posts = []
topic_len = []
for u in url:
    
    browser.get(u)
    time.sleep(1)
    
    elem = browser.find_element_by_tag_name('body')
    # print(elem)

    no_of_pagedowns = 3000

    while no_of_pagedowns:
        elem.send_keys(Keys.PAGE_DOWN)
        time.sleep(0.2)
        no_of_pagedowns-=1


    post_elems = browser.find_elements_by_tag_name("a")

    # for post in post_elems:
    #     print(post.get_attribute('href'), len (post.get_attribute('href')))
    post_relevant = [post for post in post_elems if len(post.get_attribute('href')) > 100] 
    topic_len.append(len(post_relevant))
    for i in range(0, len(post_relevant), 3):
        all_posts.append(post_relevant[i].get_attribute('href'))
topic_len

len(all_posts)

para = []
tags = []
bullets = []
images = []
links = []
data_scrapped = pd.DataFrame()

for i in range(len(all_posts)):

    try:
        r  = requests.get(all_posts[i])
        data = r.text
        soup = BeautifulSoup(data, 'html.parser')
        data_scrapped.loc[i,'title'] = soup.title.text[0:-23]
        data_scrapped.loc[i,'claps'] = soup.find('button', class_="button button--chromeless u-baseColor--buttonNormal js-multirecommendCountButton").text
        format = "%Y-%m-%d"
        data_scrapped.loc[i,'datePublished'] =  datetime.strptime(ast.literal_eval(soup.find('script').text)["datePublished"][0:10], format)
        data_scrapped.loc[i,'read_time'] = soup.find("span",class_ = "readingTime")['title'][:-9]
        data_scrapped.loc[i,'followedBy'] = soup.find_all("script")[-1].text.split('"usersFollowedByCount":')[1].split(",")[0]
        data_scrapped.loc[i,'following'] = soup.find_all("script")[-1].text.split('"usersFollowedCount":')[1].split(",")[0]
        
        links.append([item['href'] for item in soup.find_all('a', class_ = "markup--anchor markup--p-anchor")])
        para.append( [hit.text.strip() for hit in soup.find_all('p')])
        tags.append( ast.literal_eval(soup.find('script').text)["keywords"])
        bullets.append( [hit.text.strip() for hit in soup.select('li[class*="graf graf--li"]')])
        images.append( soup.find_all('img', class_="graf-image"))
    except:
        print(i)
    if i%500 == 0:
        print(i)

data_scrapped.dropna(inplace=True)
data_scrapped.reset_index(inplace=True)

data_scrapped['para'] = [[]] * len(data_scrapped)
data_scrapped['tags'] = [[]] * len(data_scrapped)
data_scrapped['bullets'] = [[]] * len(data_scrapped)
data_scrapped['images'] = [[]] * len(data_scrapped)
data_scrapped['links'] = [[]] * len(data_scrapped)

len(para), len(tags), len(bullets), len(images), len(links), data_scrapped.shape

data_scrapped['tags'] = tags
data_scrapped['para'] = para
data_scrapped['bullets'] = bullets
data_scrapped['images'] = images
data_scrapped['links'] = links

del data_scrapped['index']
data_scrapped.shape

data_scrapped = pd.read_csv("data_scraped.csv")

data_scrapped['datePublished'] = data_scrapped['datePublished'].apply(lambda x:datetime.strptime(x,'%Y-%m-%d'))

data_scrapped.drop_duplicates().shape

data_scraped.head(3)

data_scrapped.to_csv('data_scraped.csv', index = False, encoding='utf-8')

data = pd.DataFrame()
for i in range(data_scrapped.shape[0]):
    
    data.loc[i, "title_words"] = len(data_scrapped.loc[i, "title"].split(" "))
    data.loc[i, "days_passed"] = (datetime.now() - data_scrapped.loc[i,'datePublished'] ).days
    data.loc[i, "ct_image"] = len(data_scrapped.loc[i, "images"])
    data.loc[i, "ct_tags"] = len(data_scrapped.loc[i, "tags"])
    data.loc[i, "text"] =  "".join(data_scrapped.loc[i, "para"] + data_scrapped.loc[i, "bullets"])
    data.loc[i, "ct_words"] = len(data.loc[i, "text"].split())
    data.loc[i, "title_emot_quotient"] = abs(sid.polarity_scores(data_scrapped.loc[i, "title"])['compound'])
    data.loc[i, "featured_in_tds"] = 'Towards Data Science' in data_scrapped.loc[i,'tags']
    data.loc[i,'read_time'] = int(data_scrapped.loc[i,'read_time']) 
    data.loc[i,'links'] =  data_scrapped.loc[i,'links']
    data.loc[i,'following'] = int(data_scrapped.loc[i,'following']) 
    data.loc[i,'followedBy'] = int(data_scrapped.loc[i,'followedBy'])

    
def claps2num(x):
    if "K" in x:
        return float(x.replace("K",""))*1000
    else:
        return int(x)
data['claps'] = data_scrapped['claps'].apply(lambda x: claps2num(x))
data["img/word"] = data['ct_image']/data['ct_words']

data.head(3)

data.drop_duplicates().shape

data.to_csv("data.csv",index=False, encoding='utf-8')

data_scrapped.head()



































