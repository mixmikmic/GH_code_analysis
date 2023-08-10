import requests
from bs4 import BeautifulSoup
from xml.etree import ElementTree
import pandas as pd
import time
import numpy as np

# a = soup.find_all("span", {"class": 'lot-size' })
# [el.getText() for el in a]

def find_attr(sp, *attrs):
    obj = sp
    for attr in attrs:
        if obj.find(attr):
            obj = obj.find(attr)
        else:
            return np.nan
    return obj.getText()

listings2 = []



for i in range(1,21):
    #print i
    page_url = "http://www.zillow.com/homes/for_sale/Boston-MA/44269_rid/any_days/42.4379,-70.629044,42.191135,-71.310883_rect/10_zm/%d_p/" % i
    page = requests.get(page_url).text
    page_soup = BeautifulSoup(page, 'html.parser')
    
    item = page_soup.find("div", {"id": 'search-results' }).find_all("article", {"class": "property-listing"})
    if not len(item) == 26:
        print i, len(item)
    else:
        listings2 = listings2 + item
    time.sleep(2)

rec_arr = []
price_arr = []



zid = 'X1-ZWz1f9f43srdhn_5b1bv'

def request_record(listing):
    
    el = listing.find("figure").find("a")["href"][13:-1].split('/')[0]
    addr_zip = ('+'.join(el.split('-')[:3]) , el.split('-')[-1])  
    
    url = 'http://www.zillow.com/webservice/GetDeepSearchResults.htm?zws-id='+ zid             +'&address='+ addr_zip[0]             +'&citystatezip=' + addr_zip[1]

    req = requests.get(url)
    
    soup = BeautifulSoup(req.text).find('response')
    
    if soup:         
        record = {}
        record['lat'] = find_attr(soup, 'address', 'latitude') 
        record['lng'] = find_attr(soup, 'address', 'longitude') 
        record['yr'] = find_attr(soup, 'yearbuilt') 
        record['sz'] = find_attr(soup, 'lotsizesqft') 
        record['fsz'] = find_attr(soup, 'finishedsqft') 
        record['rm'] = find_attr(soup, 'totalrooms') 
        record['val'] = find_attr(soup, 'zestimate', 'amount') 
        record['use'] = find_attr(soup, 'usecode') 
        record['url'] = url
        return record
    else:
        return {'lat': "",
                'lng': "",
                'yr': "",
                'sz': "",
                'fsz': "",
                'rm': "",
                'val': "",
                'use': "",
                'url': url
               }
    
def find_price(listing):  
    price = listing.find("dt", {"class": 'price-large' })
    return price.getText() if price else np.nan

rec_arr2 = map(request_record, listings)
# price_arr = map(find_price, listings)

df = pd.DataFrame.from_records(rec_arr)
df['price'] = price_arr

df2 = pd.DataFrame.from_records(rec_arr2)
df2['price'] = price_arr

df2

df

df3 = df.fillna(df2)
len(df3.dropna(subset=['val','price'], how='all'))

len(df3.drop_duplicates(subset=['url']))

df3.drop_duplicates(subset=['url']).to_csv('realestate2.csv')



item[0].find("figure").find("a")["href"]

len(listings)

page_url = "hhttp://www.zillow.com/homes/for_sale/Boston-MA/44269_rid/any_days/globalrelevanceex_sort/%d_p/" % 1
page = requests.get(page_url).text
page_soup = BeautifulSoup(page, 'html.parser')

a = page_soup.find_all("a", {"class": 'hdp-link routable' })
items = [el['href'][13:-1].split('/') for el in a]
addr_zips = [('+'.join(el[0].split('-')[:3]) , el[0].split('-')[-1]) for el in items]

dt = page_soup.find_all("dt", {"class": 'price-large' })
prices = [el.getText() for el in a]

assert len(addr_zips) == len(prices)

rec_arr = []
for el in addr_zips:
    rec_arr = rec_arr + [request_record(el)]
    #time.sleep(1000)
prices_all = prices_all + prices
rec_arr_all = rec_arr_all + rec_arr

df = pd.DataFrame.from_records(rec_arr_all)
df['price'] = prices_all

