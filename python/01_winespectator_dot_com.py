import pandas as pd
import numpy as np

import requests
from bs4 import BeautifulSoup
from selenium import webdriver

import dill
import re
import time

from winespectator_login import winespectator_login_name, winespectator_password

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import seaborn as sns

base_url = 'http://www.winespectator.com/dailypicks/category/catid/1/page/{}'

url_list = list()

for pg in range(1, 867):
    url = base_url.format(pg)
    req = requests.get(url)
    soup = BeautifulSoup(req.text, 'lxml')
    
    soup_list = soup.find_all(attrs={'class':'daily-wine-items'})
    
    if len(soup_list) > 0:
        for sl in soup_list:
            try:
                url_list.append('http://www.winespectator.com/' + sl.find('a',href=True).get('href'))
            except:
                pass

len(url_list)

with open('../priv/pkl/01_winespectator_dot_com_url_list.pkl','w') as fh:
    dill.dump(url_list, fh)

# url_list = dill.load(open('../priv/pkl/01_winespectator_dot_com_url_list.pkl','rb'))

# ! ssh -ND 8081 server1
# ! ssh -ND 8082 server2
# ! ssh -ND 8083 server3
# ! ssh -ND 8084 server4

phantom_path = '/usr/bin/phantomjs'

service_args1 = ['--proxy=127.0.0.1:8081', '--proxy-type=socks5']
service_args2 = ['--proxy=127.0.0.1:8082', '--proxy-type=socks5']
service_args3 = ['--proxy=127.0.0.1:8083', '--proxy-type=socks5']
service_args4 = ['--proxy=127.0.0.1:8084', '--proxy-type=socks5']

driver1 = webdriver.PhantomJS(phantom_path, service_args=service_args1)
driver2 = webdriver.PhantomJS(phantom_path, service_args=service_args2)
driver3 = webdriver.PhantomJS(phantom_path, service_args=service_args3)
driver4 = webdriver.PhantomJS(phantom_path, service_args=service_args4)

driver_list = [driver1, driver2, driver3, driver4]

for driver in driver_list:

    driver.get('https://www.winespectator.com/auth/login')

    userid = driver.find_element_by_name('userid')
    userid.send_keys(winespectator_login_name)

    passwd = driver.find_element_by_name('passwd')
    passwd.send_keys(winespectator_password)

    login = driver.find_element_by_id('target')
    login.click()

    time.sleep(1.5)

wine_df_list = list()

for url in enumerate(url_list):
    if (url[0] % 100) == 0:
        print url[0]
        
    driver = np.random.choice(driver_list)
    
    full_url = url[1]
    driver.get(full_url)
    time.sleep(1.5)
    
    html = driver.page_source
    
    try:
        soup = BeautifulSoup(html, 'lxml')
        wine_data = soup.find(attrs={'class':'mod-container'})
    except:
        pass
    
    # extract winery name
    try:
        winery = wine_data.find('h1').text.strip()
    except:
        winery = ''
    
    # extract wine name
    try:
        wine = wine_data.find('h4').text.strip()
    except:
        pass

    # extract year
    try:
        year = re.search(r"""((?:20|19)[0-9]{2})""", wine).group(1)
        wine = wine.replace(year, '').strip()
    except:
        year = ''
        
    # extract review
    try:
        review = wine_data.find(attrs={'id':'bt-body'}).text.strip()
    except:
        review = ''
        
    # score
    try:
        score = wine_data.find(text=re.compile('Score: [0-9]{2}'))
        score = score.replace('Score:', '').strip()
    except:
        score = ''
        
        
    for para in wine_data.find_all(attrs={'class':'paragraph'}):
        text = para.text

        if 'Release Price' in text:
            try:
                release_price = re.search(r"""Release Price \$([0-9\.]+)""", text).group(1)
            except:
                release_price = ''
        elif 'Country' in text:
            try:
                country = re.search(r"""Country (.+)""", text).group(1)
            except:
                country = ''
        elif 'Region' in text:
            try:
                region = re.search(r"""Region (.+)""", text).group(1)
            except:
                region = ''
    

    df = pd.DataFrame({'winery':winery, 'wine':wine, 'year':year, 'score':score,
                  'price':release_price, 'country':country, 'region':region,
                  'review':review, 'url':full_url}, index=pd.Index([0]))
    wine_df_list.append(df)

wine_df = pd.concat(wine_df_list, axis=0).reset_index(drop=True)

wine_df.to_pickle('../priv/pkl/01_winespectator_dot_com_data.pkl')

wine_df.review.apply(len).hist(bins=100)

