## Install Selenium and your browser driver:
# !pip install selenium
# !brew install chromedriver

import requests
from bs4 import BeautifulSoup

gmail_url="https://gmail.com"
soup=BeautifulSoup(requests.get(gmail_url).text)
print(soup.prettify())

print(soup.find(id='Email'))

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
import os

# Specify where the browser driver is located ---- only do this if not in $PATH already
# chromedriver = "./chromedriver"
# os.environ["webdriver.chrome.driver"] = chromedriver

# Initialize a browser
browser = webdriver.Chrome()
browser.get("https://www.gmail.com")

browser.page_source

email = browser.find_element_by_id("Email") # identifierId
email.send_keys("email@host.com")
email.send_keys(Keys.RETURN) 

passwd = browser.find_element_by_name("password")
passwd.send_keys("##########")
passwd.send_keys(Keys.RETURN)

compose_button=browser.find_element_by_xpath('//div[text()="COMPOSE"]')
compose_button.click()

to_field = browser.find_element_by_name("to")
to_field.send_keys("ling@thisismetis.com")

subject = browser.find_element_by_name("subjectbox")
subject.send_keys("Hello from PyData!")

message_body = browser.find_element_by_xpath("//div[@aria-label='Message Body']")
message_body.send_keys("Hello,")
message_body.send_keys(Keys.RETURN)
message_body.send_keys("I am a computer and I just became self aware.")

send_button = browser.find_element_by_xpath("//div[contains(@aria-label, 'Send')]")
send_button.click()

browser.close()

browser = webdriver.Chrome()

url = "http://www.imdb.com"
browser.get(url)

query = browser.find_element_by_id("navbar-query")
query.send_keys("sound of music ")

query.send_keys(Keys.ARROW_DOWN)
query.send_keys(Keys.RETURN)

budget = browser.find_element_by_xpath('//*[@id="titleDetails"]/div[7]').text
print(budget)

metascore = browser.find_element_by_xpath('//*[@id="title-overview-widget"]/div[3]/div[2]/div[2]/div[1]/a/div/span').text
print(metascore)

imdbscore = browser.find_element_by_xpath('//*[@id="title-overview-widget"]/div[2]/div[2]/div/div[1]/div[1]/div[1]/strong/span').text
print(imdbscore)

browser.close()

# import pandas as pd
# df = pd.read_csv('../data/alltime_movies200_clean.csv')

# df['release date'] = pd.to_datetime(df['release date'])
# titles = df['movie title']
# years = df['release date'].map(lambda x: x.year)  

# Scraping all 200 movies with Selenium

# browser = webdriver.Chrome()
# url = "http://www.google.com"
# movie_data = []

# for title, year in zip(titles, years):
#     query_term = title + ' ' + str(year) 
    
#     browser.get(url)

#     query = browser.find_element_by_id("lst-ib")
#     query.send_keys(query_term)
#     query.send_keys(" ")
#     query.send_keys("imdb")
#     query.send_keys(Keys.RETURN)
    
    
#     try:
#         element = WebDriverWait(browser, 10).until(
#             EC.presence_of_element_located((By.XPATH, '//a[contains(text(), "IMDb")]'))
#         )
#     finally:
#         element.click() 
    
#     budget, metascore, imdbscore = None, None, None
    
#     try:
#         budget = browser.find_element_by_xpath('//*[@id="titleDetails"]/div[7]').text
#     except:
#         pass
    
#     try:
#         metascore = browser.find_element_by_class_name('metacriticScore').text
#     except:
#         pass
    
#     try:
#         imdbscore = browser.find_element_by_class_name('ratingValue').text.split('/')[0]
#     except:
#         pass
    
#     movie_dict = {"title": title,
#                   "budget": budget.split()[1],
#                   "metascore": metascore,
#                   "imdbscore": imdbscore
#                  }

#     movie_data.append(movie_dict)
# browser.close()

len(movie_data)

df_imdb = pd.DataFrame(movie_data)

df_merged = pd.merge(df, df_imdb, left_on = 'movie title', right_on = 'title')

import pandas as pd
df_merged = pd.read_csv('../data/alltime_movies200_merged.csv')

# need to install sklearn if not already
#!conda install scikit-learn -y

df_merged['month'] = pd.to_datetime(df_merged['release date']).map(lambda x: x.month)
df_merged['year'] = pd.to_datetime(df_merged['release date']).map(lambda x: x.year)

selected_columns = ['actual_dtg', 'domestic total gross', 'rating', 'runtime (mins)', 'imdbscore', 'budget', 'month','year']

df = df_merged[selected_columns]

df.head()

y = df['actual_dtg']
X = df.iloc[:,2:]
X['month'] = X.month.astype("category")

X = pd.get_dummies(X)

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn import metrics

clf = LinearRegression()

scores = cross_val_score(clf, X, y,cv=5, scoring = 'neg_mean_absolute_error')

scores

# Like we expected, these are very limited information and does not make a good model. 
# We have to collect even more data

