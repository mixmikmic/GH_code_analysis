import pandas as pd
import numpy as np

import requests
from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from fake_useragent import UserAgent
import multiprocess as mp

import subprocess
import dill
import re
import time
import psutil
import os

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
    return driver

# A function to scrape all the contents of the review links on a given page with a list of reviews

def scrape_list(url_no, driver):
    base_url = 'http://www.winemag.com/?s=&drink_type=wine&page={}'
    url = base_url.format(url_no)
    
    scrape_dict = dict()
    
    try:
        driver.get(url)
        time.sleep(1.25)

        html = driver.page_source
        soup = BeautifulSoup(html, 'xml')
        review_list = [x.get('href') for x in soup.find_all(attrs={'class':'review-listing'}, href=True)]
        success = True
    except:
        scrape_dict[url_no] = np.NaN
        success = False
    
    if success:
        review_list = np.array(review_list)
        np.random.shuffle(review_list)
        for review in review_list:
            review_result = scrape_review(review, driver)
            time.sleep(np.random.rand()*1.5)
            scrape_dict[(url_no, review)] = review_result
    return scrape_dict

# The function to scrape a single review

def scrape_review(url, driver):     
    try:
        driver.get(url)
        time.sleep(np.random.rand()*1.5+1.5)
        html = driver.page_source

        soup = BeautifulSoup(html, 'lxml')
        success = True
        
    except:
        success = False        
        
    if success:
        # scrape the data
        value_dict = dict()

        value_dict['url'] = url[1]

        try:
            title = soup.find(attrs={'class':'article-title'}).text
            value_dict['title'] = title
        except:
            value_dict['title'] = ''

        try:
            rating = soup.find(attrs={'id':'points'}).text
            value_dict['rating'] = rating
        except:
            value_dict['rating'] = ''

        try:
            review = soup.find(attrs={'class':'description'}).text
            value_dict['review'] = review
        except:
            value_dict['review'] = ''

        try:
            primary_info = soup.find(attrs={'class':'primary-info'})
            primary_keys = [x.text.strip().lower() 
                            for x in primary_info.find_all(attrs={'class':'info-label medium-7 columns'})]
            primary_values = [x.text.strip().encode('utf-8') 
                              for x in primary_info.find_all(attrs={'class':'info medium-9 columns'})]

            try:
                price = re.search(r"""\$([0-9]+)""", primary_values[0]).group(1)
                primary_values[0] = price
            except:
                pass

            value_dict.update(dict(zip(primary_keys, primary_values)))
        except:
            pass

        try:
            secondary_info = soup.find(attrs={'class':'secondary-info'})

            secondary_keys = [x.text.strip().lower().replace(' ', '_') 
                              for x in secondary_info.find_all(attrs={'class':'info-label small-7 columns'})]
            secondary_values = [x.text.strip() 
                                for x in secondary_info.find_all(attrs={'class':'info small-9 columns'})]

            value_dict.update(dict(zip(secondary_keys, secondary_values)))
        except:
            pass
        
    if success:
        return pd.Series(value_dict)
    else:
        return url

# A master scraping function for a port and list of URLS

def master_scrape(args):
    
    port_nos = args[1]
    url_nos = args[0]
    nsplit = len(port_nos)

    for url_split,port_ in zip(np.array_split(url_nos, nsplit), 
                               np.array_split(port_nos, nsplit)):

        port = port_[0]
        np.random.shuffle(url_split)
        
        driver = make_driver(port)


        for no in url_split:
            time.sleep(np.random.rand()*5+5)
            try:
                scrape_dict = scrape_list(no, driver)
                with open('../priv/pkl/06_wine_enthusiast_dot_com_data_{}.pkl'.format(no), 'w') as fh:
                    dill.dump(scrape_dict, fh)
                print no
            except:
                print 'ERROR: ' + no
                pass
        
    return

# Start the ssh tunnels
get_ipython().system(' ../priv/scripts/ssh_tunnels.sh')

# Do all the scraping
# Note that I ended up doing this in two parts (pages 1-3000 and pages 3001-6529)
# This was so that I could terminate the cluster in the middle and create a new
# one, resulting in different IP addresses to proxy through for the scrape

nthreads = 16
ncomputers = 16

# url_nos = np.arange(1, 3001)
url_nos = np.arange(3001, 6530)

np.random.shuffle(url_nos)

port_nos = np.array([8081+x for x in range(ncomputers)])

pool = mp.Pool(processes=nthreads)
results = pool.map(master_scrape, [x for x in zip(np.array_split(url_nos, nthreads), 
                                                  np.array_split(port_nos, nthreads))])
pool.close()

get_ipython().system(' echo "pushover \'scrape finished\'" | /bin/zsh')

# try again with pages that seem to have been skipped

# Get the missing pages
from glob import glob
file_list = glob('../priv/pkl/06_wine_enthusiast_dot_com_data_*.pkl')
int_sorter = lambda x: int(re.search(r"""06_wine_enthusiast_dot_com_data_(.+).pkl""", x).group(1))
file_list = sorted(file_list, key=int_sorter)

full_list = np.arange(1,6530)
num_list = np.array([int_sorter(x) for x in file_list])

mask = np.invert(np.in1d(full_list, num_list))
url_nos = full_list[mask]
###

np.random.shuffle(url_nos)

pool = mp.Pool(processes=nthreads)
results = pool.map(master_scrape, [x for x in zip(np.array_split(url_nos, nthreads), 
                                                  np.array_split(port_nos, nthreads))])
pool.close()

get_ipython().system(' echo "pushover \'scrape finished\'" | /bin/zsh')

