from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import requests
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.common.exceptions import NoSuchElementException

from bs4 import BeautifulSoup
import re
from re import sub
from decimal import Decimal
import random
import pandas as pd
from urllib.request import urlopen
import dateutil.parser
import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Image

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 25)
pd.set_option('display.precision', 3)

# enables inline plots, without it plots don't show up in the notebook
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = [16, 12]


from fake_useragent import UserAgent
import os



chromedriver = "/Applications/chromedriver"
os.environ["webdriver.chrome.driver"] = chromedriver

import json


zipcodes=pd.read_csv('bayarea_zipcodes.csv')

zipcodes.ZIP
# len(zipcodes.ZIP)
# 187*0.8*500

with open('house_url.json', 'r') as f:
     house_url=json.load(f)
len(house_url)

#house_url=[]
#setting up fake user agent
ua = UserAgent()
user_agent = {'User-agent': ua.random}

#setting user agent names
opts=Options()
opts.add_argument('user-agent=user_agent')
driver = webdriver.Chrome(chromedriver, chrome_options=opts)
#only the recently sold homes
driver.get(url="https://www.zillow.com/ca/sold/")   

for zipcode_i in zipcodes.ZIP:

    sleeptime=random.uniform(5,7)
    time.sleep(sleeptime)
    #enter zipcode
    #I will only have 500 homes for each zipcode allocated around all places because
    #zillow requires you to zoom in to different map areas for more houses
    zipcode = driver.find_element_by_xpath("//input[@id='citystatezip']")
    zipcode.send_keys(Keys.BACKSPACE)
    zipcode.send_keys(Keys.BACKSPACE)
    zipcode.send_keys(Keys.BACKSPACE)
    zipcode.send_keys(Keys.BACKSPACE)
    zipcode.send_keys(Keys.BACKSPACE)
    zipcode.send_keys(str(zipcode_i))
    zipcode.send_keys(Keys.RETURN)
    sleeptime=random.uniform(4,6)
    time.sleep(sleeptime)
    
    #click on next page
    try:
        driver.find_element_by_class_name('zsg-pagination-next').click()
        sleeptime=random.uniform(3,5)
        time.sleep(sleeptime)
        #second page url
        url2=driver.current_url
        print(url2)
        ##### get the url to feed into beautifulsoup
        url_list=[]
        url1=url2[:-4]+str(1)+url2[-3:]
        url_list.append(url1)
        url_list.append(url2)
        #for i in range (page 3 to total page+1)
        for i in range(3,21):
            url_list.append(url2[:-4]+str(i)+url2[-3:])


        for j in url_list:
            #random user-agent
            ua = UserAgent()
            user_agent = {'User-agent': ua.random}
            #start using beautiful soup
            response = requests.get(j, headers=user_agent)
            sleeptime=random.uniform(3,5)
            time.sleep(sleeptime)
            print(response.status_code)
            soup=BeautifulSoup(response.text, "lxml")
            #url for individual houses
            for i in soup.find_all("a", {"class" :"zsg-photo-card-overlay-link routable hdp-link routable mask hdp-link"}):
                house_url.append('zillow.com'+i['href'])
    except NoSuchElementException:
        print('no houses sold in this area')
        sleeptime=random.uniform(2,4)
        time.sleep(sleeptime)
        continue


with open('house_url.json', 'w') as f:
     json.dump(house_url, f)

#open individual house's url using beautiful soup and extract all housing info

#setting up dataframe as a list
df_columns=['street address','city','state','zip',
            'bed','bath','sqft','type','year-built',
            'heating','cooling','parking','lot',
           'school1','school2','school3','grade1','grade2','grade3','rate1','rate2','rate3',
           'sold-price','sold-date','url']
data=[]

for j in house_url:
    #using random user-agent
    ua = UserAgent()
    user_agent = {'User-agent': ua.random}
    #feed into beautiful soup
    response = requests.get(url='https://www.'+j, headers=user_agent)
    sleeptime=random.uniform(3,5)
    time.sleep(sleeptime)
    print(response.status_code)
    page = response.text
    soup = BeautifulSoup(page,"lxml")
    
    #getting individual house characteristics
    housedata=[]
    #street address
    for i in soup.find_all("header", {"class" : "zsg-content-header addr"}):
        housedata.append(i.text.split(',')[0])
    #city, state, zip
    for i in soup.find_all("span", {"class" : "zsg-h2 addr_city"}):
        city=i.text.split(',')
        housedata.append(city[0])
        housedata.append(city[1].split(' ')[1])
        housedata.append(city[1].split(' ')[2])
    #bed, bath, sqft
    for i in soup.find_all("span", {"class" : "addr_bbs"}):
        housedata.append(i.text.split(' ')[0])
    #facts and features table: type, year built, heating, cooling, parking, lotsize
    for i in soup.find_all("div", {"class" :"hdp-fact-ataglance-value"}):
        housedata.append(i.text.strip())
    #school name
    for i in soup.find_all("div", {"class" :'nearby-schools-name'}):
        housedata.append(i.text.strip())    
    #school grades
    for i in soup.find_all("div", {"class" :'nearby-schools-grades'}):
        housedata.append(i.text.strip())
    #school rating 
    for i in soup.find_all("div", {"class" :'nearby-schools-rating'}):
        housedata.append(i.text.strip().split(' ')[0])
    
    #prices
    
    price=[]
    for i in soup.find_all("div", {"class" : "home-summary-row"}):
        #housedata.append(i.text.strip())
        price.append(i.text.strip())

    try:
        sold_price=Decimal(sub(r'[^\d.]','',re.findall('[£$€]{1}[,0-9]{1,10}',price[0])[0]))
        sold_date=dateutil.parser.parse(price[1],fuzzy=True)
    except IndexError:
        print("for sale houses")
        continue
        
    housedata.append(sold_price)
    housedata.append(sold_date)

        
    housedata.append('https://www.'+j)

    #append list of individual house characteristics into data
    data.append(housedata)

    

#converting the data list into a dataframe
housedf=pd.DataFrame(data,columns=df_columns)

housedf

#save data into a csv file
housedf.to_csv('data_95117.csv', index=False)
#studio will have 0 bedrooms

