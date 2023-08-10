import pandas as pd
import numpy as np

import requests
import json
from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities

from fake_useragent import UserAgent
import multiprocess as mp

from glob import glob

import dill
import re
import time

# Start the ssh tunnels
get_ipython().system(' ../priv/scripts/ssh_tunnels.sh')

ncomputers = 16
nthreads = 16

port_nos = np.array([8081+x for x in range(ncomputers)])

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

# create the url list
base_url = 'http://www.snooth.com/wines/#action=search&hide_state=1&country=US&color[0]={}&entity=wine&store_front=0&search_page={}'

# red wines
scrape_urls = [base_url.format(0, pg) for pg in range(1,1001)]

# white wines
scrape_urls.extend([base_url.format(1, pg) for pg in range(1,1001)])

# rose
scrape_urls.extend([base_url.format(2, pg) for pg in range(1,468)])

scrape_urls = [x for x in enumerate(scrape_urls)]

color_dict = {'0':'red', '1':'white', '2':'rose'}

def master_scrape_urls(args):
    port = args[0]
    scrape_list = args[1]
    
    driver = make_driver(port)

    url_list = list()

    for u in scrape_list:
        url_no = u[0]
        url = u[1]
        
        if url_no % 100 == 0:
            print url_no
            
            if len(url_list) > 0:
                with open('../priv/pkl/02_snooth_dot_com_url_{}.pkl'.format(url_no-100), 'w') as fh:
                    dill.dump(url_list, fh)
            url_list = list()
            
            
        color_no = re.search(r"""color\[0\]=([0-2])&""", url).group(1)
        color = color_dict[color_no]
    
        driver.get(url)
        time.sleep(1.1)
        html = driver.page_source
        soup = BeautifulSoup(html, 'lxml')

        the_list = [x.find('a', href=True).get('href') 
                         for x in soup.find_all(attrs={'class':'wine-name'}) 
                         if x.find('a', href=True) is not None]
        the_list = [(color, x, url_no) for x in the_list]
        url_list.extend(the_list)
        
    with open('../priv/pkl/02_snooth_dot_com_url_{}.pkl'.format((url_no//100)*100), 'w') as fh:
        dill.dump(url_list, fh)
        
    return

len(scrape_urls)

# Split the url list up for scraping
split_urls = list()

for i in range(12):
    begin = i*200
    if i != (12-1):
        end = (i+1)*200
        split_urls.append(scrape_urls[begin:end])
    else:
        split_urls.append(scrape_urls[begin:])

pool = mp.Pool(processes=12)
results = pool.map(master_scrape_urls, [x for x in zip(port_nos[:12], split_urls)])
pool.close()

url_data_list = glob('../pkl/02_snooth_dot_com_url_*.pkl')
int_sorter = lambda x: int(re.search(r"""_([0-9]+)\.pkl""", x).group(1))
url_data_list = sorted(url_data_list, key=int_sorter)

combined_urls = list()
for fil in url_data_list:
    with open(fil, 'r') as fh:
        combined_urls.extend(dill.load(fh))

combined_urls[0]

combined_urls = [(x[1][0],x[1][1],x[0]) for x in enumerate(combined_urls)]

combined_urls[-1]

len(combined_urls)

with open('../priv/pkl/02_snooth_dot_com_url_list.pkl','w') as fh:
    dill.dump(combined_urls, fh)

# reload or load if the variable doesn't exist
with open('../priv/pkl/02_snooth_dot_com_url_list.pkl','r') as fh:
    combined_urls = dill.load(fh)

def scrape_data(args):
    
    port = args[0]
    url_list = args[1]

    driver = make_driver(port)
    
    req = requests.session()
    req_proxy = {'http': "socks5://127.0.0.1:{}".format(port)}
    
    wine_df_list = list()
    
    for url in url_list:

        color = url[0]
        full_url = url[1]
        url_no = url[2]    

        if (url_no % 100) == 0:
            print url_no

            if len(wine_df_list) > 0:
                wine_df = pd.concat(wine_df_list, axis=0).reset_index(drop=True)
                wine_df.to_pickle('../priv/pkl/02_snooth_dot_com_data_{}.pkl'.format(url_no-100))

            wine_df_list = list()
            
            
        driver.get(full_url)
        time.sleep(1.2)
        html = driver.page_source
        soup = BeautifulSoup(html, 'lxml')

        wine_block1 = soup.find(attrs={'class':'wpp2014-wine_block-info'})
        wine_block2 = soup.find(attrs={'class':'wpp2014-reg_rat-region_vintage'})

        # wine name and year
        try:
            wine = wine_block1.find(attrs={'id':'wine-name'}).text.strip()
        except:
            wine = ''
            year = ''

        try:
            year = re.search(r"""((?:20|19)[0-9]{2})""", wine).group(1)
            wine = wine.replace(year, '').strip()
        except:
            year = ''

        # review
        try:
            review = wine_block1.find(attrs={'class':'winemakers-notes'}).text.replace("Winemaker's Notes:", '').strip()
        except:
            review = review.strip()

        # prices
        try:
            price_list = wine_block1.find(attrs={'class':'wpp2014-wine_block-sample_prices'}).find_all(attrs={'itemprop':'price'})
        except:
            price_list = np.NaN
        else:
            price_list = np.array([float(x.text) for x in price_list]).mean()



        # region, winery, varietal
        region = ''
        winery = ''
        varietal = ''

        try:
            data_list = [re.split(r"""\s+""", x.text.replace(u'\xbb', '').strip()) 
                         for x in wine_block2.find_all('div')]
        except:
            pass

        try:
            for l in data_list:
                if 'region' in l[0].lower():
                    region = ' '.join(l[1:])
                if 'winery' in l[0].lower():
                    winery = ' '.join(l[1:])
                if 'varietal' in l[0].lower():
                    varietal = ' '.join(l[1:])
        except:
            pass

        # get the image
        try:
            image_url = soup.find(attrs={'id':'wine-image-top'}).get('src')

            if len(image_url) > 0:
                filext = os.path.splitext(image_url)[-1]
                path = '../priv/images/snooth_dot_com_' + str(url_no) + filext
                img = req.get(image_url, proxies=req_proxy)
                time.sleep(1.2)
                
                # print image_url, url_no, path

                if img.status_code == 200:
                    with open(path, 'wb') as f:
                        for chunk in img:
                            f.write(chunk)

        except:
            image_url = ''


        df = pd.DataFrame({'wine':wine, 'year':year, 
                           'review':review,
                           'region':region, 'winery':winery, 'varietal':varietal,
                           'price':price_list,
                           'color':color, 'url':full_url,
                           'image_url':image_url, 'url_no':url_no}, index=pd.Index([0]))

        wine_df_list.append(df)
        
        
    wine_df = pd.concat(wine_df_list, axis=0).reset_index(drop=True)
    wine_df.to_pickle('../priv/pkl/02_snooth_dot_com_data_{}.pkl'.format(str((url_no//100)*100)))
    
    return            

# Load the completed data

file_list = glob('../priv/pkl/02_snooth_dot_com_data_*.pkl')
int_sorter = lambda x: int(re.search(r"""_([0-9]+)\.pkl""", x).group(1))
file_list = sorted(file_list, key=int_sorter)

file_nums = np.array(map(int_sorter, file_list))

all_nums = np.arange(0, 49311, 100)
redo_nums = all_nums[np.invert(np.in1d(all_nums, file_nums))]

redo_lims = np.array(list(zip(redo_nums, redo_nums+100)))
if redo_lims[0,-1] == 49300:
    redo_lims[-1,-1] = 49500

bool_selector = [(((x[-1] >= redo_lims[:,0])&(x[-1] < redo_lims[:,1]))==True).any() for x in combined_urls]

selected_nums = [x for (x,y) in zip(combined_urls, bool_selector) if y]

len(selected_nums)

redo_nums

split_urls = list()

nthreads = 5

for i in range(nthreads):
    begin = i*100
    if i != (nthreads - 1):
        end = (i+1)*100
        split_urls.append(selected_nums[begin:end])
    else:
        split_urls.append(selected_nums[begin:])

len(split_urls)

split_urls[4][-1]

# Do the scrape
pool = mp.Pool(processes=nthreads)
results = pool.map(scrape_data, [x for x in zip(port_nos, split_urls)])
pool.close()

get_ipython().system(' echo "pushover \'scrape finished\'" | /bin/zsh')

