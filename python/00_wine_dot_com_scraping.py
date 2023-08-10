import pandas as pd
import numpy as np

import requests
from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities

from fake_useragent import UserAgent
import multiprocess as mp

import dill
import re
import time
import json
import os
import re
from glob import glob

page_list = [range(0,52), range(0,22), range(0,2), range(0,5), range(0,3), range(0,2)]
url_list = ['http://www.wine.com/v6/Red-Wine/wine/list.aspx?N=7155+124&pagelength=100&Nao={}',
            'http://www.wine.com/v6/White-Wine/wine/list.aspx?N=7155+125&pagelength=100&Nao={}',
            'http://www.wine.com/v6/Rose-Wine/wine/list.aspx?N=7155+126&pagelength=100&Nao={}',
            'http://www.wine.com/v6/Champagne-and-Sparkling/wine/list.aspx?N=7155+123&pagelength=100&Nao={}',
            'http://www.wine.com/v6/Dessert-Sherry-and-Port/wine/list.aspx?N=7155+128&pagelength=100&Nao={}',
            'http://www.wine.com/v6/Sake/wine/list.aspx?N=7155+134&pagelength=100&Nao={}']
color_list = ['red', 'white', 'rose', 'sparkling', 'dessert', 'sake']

wine_urls = list()

for page_range, url_base, color in zip(page_list, url_list, color_list):
    print url_base
    for pg in page_range:
        url_no = 1+100*pg

        # Get the HTML
        req = requests.get(url_base.format(url_no))
        soup = BeautifulSoup(req.text, 'lxml')

        # Get the item list 
        item_list = soup.find(attrs={'class':'productList'}).find_all(attrs={'class':'verticalListItem'})
        item_list = [x.find('a',href=True).get('href') for x in item_list]
        item_list = [(color, 'http://www.wine.com'+x) for x in item_list]
        wine_urls.extend(item_list)

wine_urls = [(x,y,z) for (x,y),z in zip(wine_urls, range(len(wine_urls)))]

with open('../priv/pkl/00_wine_dot_com_url_list.pkl','w') as fh:
    dill.dump(wine_urls, fh)

len(wine_urls)

# A function to create the Selenium web driver

def make_driver(port):
    
    service_args = ['--proxy=127.0.0.1:{}'.format(port), '--proxy-type=socks5']
    
    dcap = dict(DesiredCapabilities.PHANTOMJS)
    ua = UserAgent()
    dcap.update({'phantomjs.page.settings.userAgent':ua.random})
    
    phantom_path = '/usr/bin/phantomjs'
    
    driver = webdriver.PhantomJS(phantom_path, 
                                   desired_capabilities=dcap,
                                   service_args=service_args)
    
    # load an url to clear the initial question about location
    initial_url = 'http://www.wine.com/v6/Schug-Sonoma-Coast-Pinot-Noir-2014/wine/148901/Detail.aspx'
    driver.get(initial_url)
    time.sleep(1.5)

    try:
        elem = driver.find_element_by_xpath('//*[@id="StateSelectShopButton"]')
        elem.click()
    except:
        pass
    
    return driver

# turn all the user reviews into their own table

def get_review_table(review_list, full_url, url_no):
    
    author_list = list()
    location_list = list()
    rating_list = list()
    date_list = list()
    style_list = list()
    review_text_list = list()

    for review in review_list:

        # author
        try:
            author = review.find(attrs={'class':'reviewAuthorAlias'}).text.strip()
        except:
            author = ''
        author_list.append(author)

        # location
        try:
            location = review.find(attrs={'class':'reviewAuthorLocation'}).text.strip()
        except:
            location = ''
        location_list.append(location)

        # rating
        try:
            rating = review.find(attrs={'class':'starRatingText'}).text
        except:
            rating = ''
        rating_list.append(rating)

        # date
        try:
            date = review.find(attrs={'class':'reviewDate'}).text.strip()
        except:
            date = ''
        date_list.append(date)

        # style
        try:
            style = review.find(attrs={'class':'reviewAttributes'}).text.replace('Style','').strip()
        except:
            style = ''
        style_list.append(style)

        # review
        try:
            review_text = review.find(attrs={'class':'reviewText'}).text.strip()
        except:
            review_text = ''
        review_text_list.append(review_text)


    review_df = pd.DataFrame({'author':author_list, 'location':location_list,
                              'rating':rating_list, 'date':date_list,
                              'style':style_list, 'review':review_text_list,
                              'url':[full_url]*len(review_list),
                              'url_no':[url_no]*len(review_list)},
                              index=pd.Index(range(len(review_list))))
    return review_df

# The scraping function that returns both the wine data and the review
def scrape_data(driver, url):
    
    full_url = url[1]
    color = url[0]
    url_no = url[2]
    
    # open the full url
    driver.get(full_url)
    time.sleep(2.5)

    # try to select the image to get the larger version
    # and get the main text
    try:
        hover = driver.find_element_by_xpath('/html/body/main/section[1]/div[2]/div')
        hover = ActionChains(driver).move_to_element(hover)
        hover.perform()
        time.sleep(0.5)
    except:
        pass

    html = driver.page_source
    main_soup = BeautifulSoup(html, 'lxml')
    wine_text = main_soup.find(attrs={'class': 'productAbstract'})

    # get the html for the reviews
    try:
        elem = driver.find_element_by_xpath('/html/body/main/section[3]/ul[1]/li[3]/a')
        elem.click()
        time.sleep(0.5)
    except:
        pass

    html = driver.page_source
    review_soup = BeautifulSoup(html, 'lxml')

    try:
        review_list = review_soup.find(attrs={'class':'topReviews'}).find_all(attrs={'class':'review'})
    except:
        review_list = []

    #### WINE DATA ####
    # image url
    try:
        image_url = main_soup.find(attrs={'class':'flyOutZoomViewport'}).find('img').get('src')
    except:
        try:
            image_url = main_soup.find(attrs={'class':'hero'}).get('src')
        except:
            image_url = ''

    # wine
    try:
        wine = wine_text.find('h1').text.strip()
    except:
        wine = ''

    try:
        year = re.search(r"""((?:20|19)[0-9]{2})""", wine).group(1)
        wine = wine.replace(year, '').strip()
    except:
        year = ''

    # kind, region
    try:
        kind_loc_match = re.search(r"""(.+) from (.+)""", wine_text.find('h2').text.strip())
    except:
        kind = ''
        region = ''
    else:
        try:
            kind = kind_loc_match.group(1)
        except:
            kind = ''

        try:
            region = kind_loc_match.group(2)
        except:
            region = ''

    # review
    try:
        review = main_soup.find(attrs={'class':'tabContent aboutTheWine active'}).find(attrs={'itemprop':'description'}).text
    except:
        review = ''

    # winery
    try:
        winery = main_soup.find(attrs={'class':'tabContent theWinery'}).find('h3').text.strip()
    except:
        winery = ''

    # ratings
    try:
        ratings_list = [x.text.strip() 
                        for x in 
                        wine_text.find_all(attrs={'class': 'wineRatings'})]

        ratings_list = [re.findall(r"""((?:20|19)[0-9]{2}|[A-Z]{2}[0-9]{2})""", x) 
                        for x in ratings_list]

        recent = [float(re.search(r"""[0-9]+""", x).group(0)) for x in ratings_list[0]]
        if len(recent) >= 1:
            rating = np.array(recent).mean()
        else:
            rating = np.NaN
    except:
        rating = np.NaN


    df = pd.DataFrame({'wine':wine, 'year':year, 'kind':kind,
                       'region':region, 'review':review,
                       'winery':winery, 'rating':rating,
                       'color':color, 'url':full_url,
                       'image':image_url, 'url_no':url_no}, index=pd.Index([0]))
    
    # download the image
    if len(image_url) > 0:
        if image_url.startswith('//'):
            image_url = 'http:' + image_url
            
        filext = os.path.splitext(image_url)[-1]
        path = '../priv/images/wine_dot_com_' + str(url_no) + filext
        req = requests.get(image_url)
        
        if req.status_code == 200:
            with open(path, 'wb') as f:
                for chunk in req:
                    f.write(chunk)
            
    
    # get the review
    review_df = None
    if len(review_list) >= 1:
        try:
            review_df = get_review_table(review_list, full_url, url_no)
        except:
            pass
        
    return df, review_df

def master_scrape(args):
    
    port = args[0]
    url_list = args[1]
    
    driver = make_driver(port)
    
    data_list = list()
    review_list = list()
    
    for url in url_list:
        if url[-1] % 100 == 0:
            print url[-1]
            
            # write the data
            if len(data_list) > 0:
                data_df = pd.concat([x for x in data_list]).reset_index(drop=True)
                data_df.to_pickle('../priv/pkl/00_wine_dot_com_data_{}.pkl'.format(url[-1]-100))
            
            if len(review_list) > 0:
                review_df = pd.concat([x for x in review_list]).reset_index(drop=True)
                review_df.to_pickle('../priv/pkl/00_wine_dot_com_review_{}.pkl'.format(url[-1]-100))
            
            data_list = list()
            review_list = list()
            
        ret_data = scrape_data(driver, url)
        data_list.append(ret_data[0])
        
        if ret_data[1] is not None:
            review_list.append(ret_data[1])
            
    # save the final data dataset
    data_df = pd.concat([x for x in data_list]).reset_index(drop=True)
    data_df.to_pickle('../priv/pkl/00_wine_dot_com_data_{}.pkl'.format((url[-1]//100)*100))
            
    review_df = pd.concat([x for x in review_list]).reset_index(drop=True)
    review_df.to_pickle('../priv/pkl/00_wine_dot_com_review_{}.pkl'.format((url[-1]//100)*100))
            
    return

# Start the ssh tunnels
get_ipython().system(' ../priv/scripts/ssh_tunnels.sh')

ncomputers = 16
nthreads = 16

port_nos = np.array([8081+x for x in range(ncomputers)])

# Split the url list up for scraping
split_urls = list()

for i in range(nthreads):
    begin = i*500
    if i != (nthreads):
        end = (i+1)*500
        split_urls.append(wine_urls[begin:end])
    else:
        split_urls.append(wine_urls[begin:])

# Run the scrape
pool = mp.Pool(processes=nthreads)
results = pool.map(master_scrape, [x for x in zip(port_nos, split_urls)])
pool.close()

# The last ~500 urls were never scraped, so do these
master_scrape([port_nos[0], wine_urls[8000:]])

data_list = glob('../priv/pkl/00_wine_dot_com_data_*.pkl')
review_list = glob('../priv/pkl/00_wine_dot_com_review_*.pkl')

int_sorter = lambda x: int(re.search(r"""_([0-9]+)\.""", x).group(1))
data_list = sorted(data_list, key=int_sorter)
review_list = sorted(review_list, key=int_sorter)

images_list = glob('../priv/images/wine_dot_com_*.*')
images_list = map(int_sorter, images_list)
images_list = sorted(images_list)

def aggregate_data(file_list):
    
    # Load and combine the data for the list of files
    combined_data = list()
    
    for fil in file_list:
        df = pd.read_pickle(fil)
        combined_data.append(df)

    return combined_data

# Find missing data
def find_missing(url_nos, wine_urls=wine_urls):
    range_array = np.array(range(len(wine_urls)))
    missing_urls = np.invert(np.in1d(range_array, url_nos))
    return range_array[missing_urls]

# Combine the data and write to a file
data_df = aggregate_data(data_list)
data_df = pd.concat(data_df, axis=1)
data_df.to_pickle('../priv/pkl/00_wine_dot_com_data_combined.pkl')

data_df.head(1)

# Combine the user reviews and write to a file
review_df = aggregate_data(review_list)
review_df = pd.concat(review_df, axis=0)
review_df.to_pickle('../priv/pkl/00_wine_dot_com_review_combined.pkl')

review_df.head(1)

# How many files are missing
find_missing(data_df.url_no)

# quite a few have no user reviews--not that surprising
len(find_missing(review_df.url_no.unique()))

find_missing(np.array(images_list)) # only four are missing photographs

