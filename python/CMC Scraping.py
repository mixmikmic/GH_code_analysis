import pandas as pd
import numpy as np

import os
#import time
from datetime import datetime

#import settings_skyze

from pprint import pprint
import re
import csv


import datetime

# import webscraping libraries
import urllib.request as urllibReq
from bs4 import BeautifulSoup

#markets = ["AseanCoin","InsureX","Mothership","People_Coin","Birds","Blakestar","Cream","GeyserCoin","0x"]
# markets = ["ATMCoin","Dochain","Nebulas-Token","XTD-Coin","Frazcoin","TrueFlip","VeChain","Hshare",
#             "MyBit-Token","Krypstal","Growers-International","YOYOW","Smoke","Bolenum","0x",
#             "Health-Care-Chain","The-ChampCoin","Excelcoin","Timereum","Sojourn","Etheriya",
#             "CoinonatX","InvestFeed","Monster-Byte","Dent","BitAsean","Digital-Developers-Fund",
#             "AdShares","BlockCAT","DeepOnion","InfChain","AppleCoin","Shadow-Token","Rupaya",
#             "Dentacoin","Nexxus","Stakecoin","Blocktix","NEVERDIE","First-Bitcoin-Capital",
#             "Rustbits","Mao-Zedong","IOU1","Centra","Bytom","Wink","CoinDash","Minex","Etherx",
#             "Stox","HBCoin","Fuda-Energy","FiboCoins","FundYourselfNow","district0x","Compcoin",
#             "OpenAnx","KekCoin","BlakeStar","Cream","Birds","Aseancoin","Mothership","GeyserCoin",
#             "InsureX","PeopleCoin","EmberCoin","CampusCoin","Primalbase"]
markets = ["Bitcoin"]
markets.sort()


### Let's get the markets from CMC

import requests
import json

# Get the URL JSON
# request=requests.get(url='https://api.coinmarketcap.com/v1/ticker/')
request=requests.get(url='https://api.coinmarketcap.com/v1/ticker/?limit=3')

# convert to DataFrame
df = pd.DataFrame(request.json())
#print(df)

# get the list of markets
print()
print("=== List of Markets =====")
markets = df["id"]
print(df["id"])

markets = ["bitcoin"]
for market in markets:
    print (market)
    # . https://coinmarketcap.com/assets/0x/historical-data/?start=20100101&end=20170826
    todayDate = datetime.date.today().strftime('%Y%m%d')
    startDate = "20170727"
    marketStr = re.sub(" ", "-", market)
    scrape_page = "https://coinmarketcap.com/assets/%(mkt)s/historical-data/?start=%(startDate)s&end=%(endDate)s" % {"mkt":marketStr, "startDate":startDate, "endDate":todayDate}
        
    # query the website and return the html to the variable ‘page’
    page = urllibReq.urlopen(scrape_page)

    print("from: %(start)s . to:  %(end)s" % {"start":startDate,"end":todayDate})
    print(scrape_page)
    
    # parse the html using beautiful soap and store in variable `soup`
    soup = BeautifulSoup(page, 'html.parser')
    # soup = BeautifulSoup(page, ‘lxml’)

    # Take out the <div> of name and get its value
    name_box = soup.find('h1', attrs={'class': 'name'})

    result = []
    allrows = soup.findAll('tr')
    for row in allrows:
        result.append([])
        allcols = row.findAll('td')
        for col in allcols:
          thestrings = col.findAll(text=True) 
          a = re.sub('[,]', '', thestrings[0])
          result[-1].append(a)

    result.pop(0)
   

df = pd.DataFrame(result)
df[0] = pd.to_datetime(df[0])
print(df.head(5))
df.to_csv("/Users/michaelnew/Dropbox/Trading/Data/test1.csv",header=False, date_format='%Y-%m-%d %H:%M:%S')



