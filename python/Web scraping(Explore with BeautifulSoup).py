from urllib2 import Request, urlopen, HTTPError
from urlparse import urlunparse, urlparse
import json 
import pandas as pd
from matplotlib import pyplot as plt
import requests

from bs4 import BeautifulSoup
import urllib

r = urllib.urlopen('https://itunes.apple.com/us/genre/ios-books/id6018?mt=8').read()
soup = BeautifulSoup(r)
print type(soup)

all_categories = soup.find_all("div", class_="nav")
category_url = all_categories[0].find_all(class_ = "top-level-genre")
categories_url = pd.DataFrame()
for itm in category_url:
    category = itm.get_text()
    url = itm.attrs['href']
    d = {'category':[category], 'url':[url]}
    df = pd.DataFrame(d)
    categories_url = categories_url.append(df, ignore_index = True)
print categories_url

categories_url['url'][0]

def extract_apps(url):
    r = urllib.urlopen(url).read()
    soup = BeautifulSoup(r)
    apps = soup.find_all("div", class_="column")
    apps_link = apps[0].find_all('a')
    column_first = pd.DataFrame()
    for itm in apps_link:
        app_name = itm.get_text()
        url = itm.attrs['href']
        d = {'category':[app_name], 'url':[url]}
        df = pd.DataFrame(d)
        column_first = column_first.append(df, ignore_index = True)
    apps_link2 = apps[1].find_all('a')
    column_second = pd.DataFrame()
    for itm in apps_link2:
        app_name = itm.get_text()
        url = itm.attrs['href']
        d = {'category':[app_name], 'url':[url]}
        df = pd.DataFrame(d)
        column_second = column_second.append(df, ignore_index = True)
    apps_link3 = apps[2].find_all('a')
    column_last = pd.DataFrame()
    for itm in apps_link3:
        app_name = itm.get_text()
        url = itm.attrs['href']
        d = {'category':[app_name], 'url':[url]}
        df = pd.DataFrame(d)
        column_last = column_last.append(df, ignore_index = True)
    Final_app_link = pd.DataFrame()
    Final_app_link = Final_app_link.append(column_first, ignore_index = True)
    Final_app_link = Final_app_link.append(column_second, ignore_index = True)
    Final_app_link = Final_app_link.append(column_last, ignore_index = True)
    return Final_app_link

app_url = pd.DataFrame()
for itm in categories_url['url']:
    apps = extract_apps(itm)
    app_url = app_url.append(apps, ignore_index = True)

app_url['url'][0]

def get_content(url):
    r = urllib.urlopen(url).read()
    soup = BeautifulSoup(r)
    des = soup.find_all('div', id = "content")
    apps = soup.find_all("div", class_="lockup product application")
    rate = soup.find_all("div", class_="extra-list customer-ratings")
    dic = []
    global app_name, descript, link, price, category, current_rate, current_count, total_count, total_rate, seller,mul_dev,mul_lang,new_ver_des
    for itm in des:
        try:
            descript = itm.find_all('div',{'class':"product-review"})[0].get_text().strip().split('\n')[2].encode('utf-8')
        except:
            descript = ''
        try:
            new_ver_des = itm.find_all('div',{'class':"product-review"})[1].get_text().strip().split('\n')[2].encode('utf-8')
        except:
            new_ver_des = ''
        try:
            app_name = itm.find_all('div',{'class':"left" })[0].get_text().split('\n')[1]
        except:
            app_name = ''
    for itm in apps:
        category = itm.find_all('span',{'itemprop':"applicationCategory" })[0].get_text()
        price = itm.find_all('div',{'class':"price" })[0].get_text() 
        link = itm.a["href"]
        seller = itm.find_all("span", itemprop="name")[0].get_text()
        try:
            device = itm.find_all("span", itemprop="operatingSystem")[0].get_text()
            if 'and' in device.lower():
                mul_dev = 'Y'
            else:
                mul_dev = "N"
        except:
            mul_dev = "N"
        try:
            lang = itm.find_all("li",class_ = "language")[0].get_text().split(',')
            if len(lang) >1:
                mul_lang = "Y"
            else:
                mul_lang = "N"
        except:
            mul_lang = "N"
    for itm in rate:
        try:
            current_rate = itm.find_all('span',{'itemprop':"ratingValue"})[0].get_text()
        except:
            current_rate = ''
        try:
            current_count = itm.find_all('span',{'itemprop':"reviewCount"})[0].get_text()
        except:
            current_count = ''
        try:
            total_count = itm.find_all('span',{'class':"rating-count"})[1].get_text()
        except:
            try:
                total_count = itm.find_all('span',{'class':"rating-count"})[0].get_text()
            except:
                total_count = ''
        try:
            total_rate = itm.find_all('div', class_="rating",itemprop = False)[0]['aria-label'].split(',')[0]
        except:
            total_rate = ''
    for i in range(3):
        try:
            globals()['user_{0}'.format(i)] = soup.find_all("div", class_="customer-reviews")[0].find_all("span", class_='user-info')[i].get_text().strip( ).split('  ')[-1]
        except:
            globals()['user_{0}'.format(i)] = ''
        try:
            globals()['star_{0}'.format(i)] = soup.find_all("div", class_="customer-reviews")[0].find_all("div", class_="rating")[i]['aria-label']
        except:
            globals()['star_{0}'.format(i)] = ''
        try:
            globals()['comm_{0}'.format(i)] = soup.find_all("div", class_="customer-reviews")[0].find_all("p", class_="content")[i].get_text()
        except:
            globals()['comm_{0}'.format(i)] = ''
       
    dic.append({'app':app_name,'link':link, 'price':price,'category':category,'current rating':current_rate, 
                'current reviews':current_count,'overall rating':total_rate,'overall reviews':total_count,
                'description':descript,'seller':seller,'multiple languages':mul_lang,
                'multiple devices':mul_dev,'new version description':new_ver_des,'user 1':user_0,
                'rate 1':star_0,'comment 1':comm_0,'user 2':user_1,'rate 2':star_1,'comment 2':comm_1,
                'user 3':user_2,'rate 3':star_2,'comment 3':comm_2})
    dic = pd.DataFrame(dic)
    return dic

full_content = pd.DataFrame()
for itm in app_url['url']:
    content = get_content(itm)
    full_content = full_content.append(content, ignore_index = True)

full_content

full_content.to_csv('app.csv',encoding='utf-8',index=True)





