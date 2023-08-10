import requests
from bs4 import BeautifulSoup
import urllib
import time
import pandas as pd

s = requests.Session()
r = s.get('http://forlap.ristekdikti.go.id/perguruantinggi')

jar = r.cookies

fgt = None
phs = None
fd = None

jar.set('FORLAPDIKTI',fd,domain='forlap.ristekdikti.go.id',path='/perguruantinggi')
jar.set('PHPSESSID',phs,domain='forlap.ristekdikti.go.id',path='/perguruantinggi')
jar.set('FGTServer',fgt,domain='forlap.ristekdikti.go.id',path='/perguruantinggi')

r = requests.get('http://forlap.ristekdikti.go.id/perguruantinggi/search',cookies=jar)

soup = BeautifulSoup(r.content)

#Initiatilize empty list
arr = []

for tt in soup.find_all("tr",class_="ttop"):
    d = {}
    d['id'] = tt.contents[3].text.strip()
    d['name'] = tt.contents[5].text
    d['url'] = tt.contents[5].a.get('href')
    d['prov'] = tt.contents[7].text.strip()
    d['cat'] = tt.contents[9].text.strip()
    d['mhs'] = int(tt.contents[15].text.strip().replace(".",""))
    d['dosen'] = int(tt.contents[13].text.strip().replace(".",""))
    arr.append(d)

START_PAGE = 20
END_PAGE = 4980
BASE_URL = "http://forlap.ristekdikti.go.id/perguruantinggi/search/"

for pagenum in range(START_PAGE,END_PAGE+20,20):
    URL = BASE_URL+str(pagenum)
    print("Current page: "+URL)
    r = requests.get(URL,cookies=jar)
    soup = BeautifulSoup(r.content)
    for tt in soup.find_all("tr",class_="ttop"):
        d = {}
        d['id'] = tt.contents[3].text.strip()
        d['name'] = tt.contents[5].text
        d['url'] = tt.contents[5].a.get('href')
        d['prov'] = tt.contents[7].text.strip()
        d['cat'] = tt.contents[9].text.strip()
        d['mhs'] = int(tt.contents[15].text.strip().replace(".",""))
        d['dosen'] = int(tt.contents[13].text.strip().replace(".",""))
        arr.append(d)

len(arr)

df = pd.DataFrame(arr)

df.head(10)

df.to_csv("forlapbase.csv")

