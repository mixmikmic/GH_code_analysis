import requests	
from bs4 import BeautifulSoup
import re
import pandas as pd
import os
from sqlalchemy import create_engine
import config
from unidecode import unidecode
import datetime

fs_urls = pd.read_csv('data/usfs_sites.csv')

fs_urls

cg_name = fs_urls.iloc[1]['facilityname']

site_url = "http://" + config.LAMP_IP + "/" + fs_urls.iloc[1]['url']
cg_req = requests.get(site_url)
cg_soup = BeautifulSoup(cg_req.text, 'lxml')

site_url = fs_urls.iloc[1]['url']
cg_data = open("webfiles/" + site_url,'r').read()
cg_soup = BeautifulSoup(cg_data, 'lxml')

try :
    for strong_tag in cg_soup.find_all('strong'):
        if ('Area Status' in unidecode(strong_tag.text)):
            status = unidecode(strong_tag.next_sibling).strip()
except Exception:
    print('couldnt get area status')

status

try :
    lat = cg_soup.find_all('div', text=re.compile('Latitude'))
    div = [row.next_sibling.next_sibling for row in lat]
    latitude  = div[0].text.strip()

except Exception:
    print('couldnt get location info')

latitude

def get_location(soup, search_text):
    try :
        divs = soup.find_all('div', text=re.compile(search_text))
        loc_div = [row.next_sibling.next_sibling for row in divs]
        return loc_div[0].text.strip()
    except Exception as ex:
        print("get_location: couldnt extract " + search_text)
        return ""

cg_lat = get_location(cg_soup, 'Latitude')
cg_long = get_location(cg_soup, 'Longitude')
cg_elev = get_location(cg_soup, 'Elevation')

print(str(cg_lat) + "," + str(cg_long) + " elevation:" + str(cg_elev))

try :
    tables = cg_soup.find_all('div', {'class': 'tablecolor'})
except Exception:
    print('couldnt get tables')

try :
    print(len(tables))
    rows = tables[0].find_all('tr')
    for row in rows:
        if row.th.text == 'Reservations:':
            reservations = unidecode(row.td.text).strip()
        if row.th.text == 'Open Season:':
            openseason = unidecode(row.td.text).strip()
        if row.th.text == 'Current Conditions:':
            conditions = unidecode(row.td.text).strip()
        if row.th.text == 'Water:':
            water = unidecode(row.td.text).strip()
        if row.th.text == 'Restroom:':
            restroom = unidecode(row.td.text).strip()
except Exception as ex:
    print('couldnt get basic campground info')
    print(ex)
    
df_info = pd.DataFrame({
        'FacilityName':[cg_name],
        'Reservations':[reservations],
        'OpenSeason':[openseason],
        'CurrentConditions':[conditions],
        'Water':[water],
        'Restroom':[restroom]
    })

df_info





