from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import requests
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait


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

from fake_useragent import UserAgent
import os

import ast
import json


with open('schoolinfo.json', 'r') as f:
     schoolinfo=json.load(f)

len(schoolinfo)

pages=['01','02','03','04','05','06','07','08','09']
pages.extend([str(i) for i in list(range(10,20001))])
coordinate_url=[]
url='https://www.greatschools.org/gsr/api/schools/11864?state=CA&extras=boundaries'
for i in pages:
    coordinate_url.append(url.split('schools/')[0]+'schools/'+str(i)+'?'+url.split('schools/')[1].split('?')[1])

#schoolinfo=[]
for i in coordinate_url[19731:]:
    #random user-agent
    ua = UserAgent()
    user_agent = {'User-agent': ua.random}
    #start using beautiful soup
    response = requests.get(i, headers=user_agent)
    sleeptime=random.uniform(3,5)
    time.sleep(sleeptime)
    print(response.status_code)
    soup=BeautifulSoup(response.text, "lxml")
    if 'High School' in soup.p.text:
        continue
    elif 'Middle School' in soup.p.text:
        continue
    elif soup.p.text=='{}':
        continue
    else:
        try:
            schoolinfo.append(ast.literal_eval(soup.p.text))
        except ValueError:
            continue
            
            
          

with open('schoolinfo.json', 'w') as f:
     json.dump(schoolinfo, f)

len(schoolinfo)

schoolinfo[-1], len(schoolinfo)

with open('schoolinfo.json', 'r') as f:
     schoolinfo1=json.load(f)

schoolinfo=[]
for i in schoolinfo1:
    schoolreduce={}
    schoolreduce['gradelevel']=i['gradeLevels']
    schoolreduce['name']=i['name'].lower()
    schoolreduce['rating']=i['rating']
    schoolreduce['city']=i['address']['city'].lower()
    schoolreduce['type']=i['schoolType'].lower()
    try:
        schoolreduce['boundary']=i['boundaries']['p']['coordinates'][0][0]
    except KeyError:
        try:
            schoolreduce['boundary']=i['boundaries']['m']['coordinates'][0][0]
        except KeyError:
            continue
    schoolinfo.append(schoolreduce)

with open('schoolinfo_final.json', 'w') as f:
     json.dump(schoolinfo, f)

