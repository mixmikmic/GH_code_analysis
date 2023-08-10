import requests
from bs4 import BeautifulSoup
import pandas as pd
from selenium import webdriver
import time
import pickle

browser = webdriver.PhantomJS()

resort_urls = {'Loveland': 'colorado/loveland',
               'Arapahoe Basin': 'colorado/arapahoe-basin-ski-area',
               'Copper': 'colorado/copper-mountain-resort',
               'Eldora': 'colorado/eldora-mountain-resort',
               'Alpine Meadows': 'california/squaw-valley-usa',
               'Vail': 'colorado/vail',
               'Monarch': 'colorado/monarch-mountain',
               'Crested Butte': 'colorado/crested-butte-mountain-resort',
               'Taos': 'new-mexico/taos-ski-valley',
               'Diamond Peak': 'nevada/diamond-peak',
               'Winter Park': 'colorado/winter-park-resort',
               'Beaver Creek': 'colorado/beaver-creek'}

URL = 'http://www.onthesnow.com/'+resort_urls['Winter Park']+'/ski-resort.html'
browser.get(URL)
# time.sleep(3)

soup = BeautifulSoup(browser.page_source,'html.parser')
temp_rows = soup.select('p.bluetxt.temp') 
snowfall_rows = soup.select('p.bluetxt.sfa')
sdepth_rows = soup.select('p.bluetxt.sd') 

temp = int(''.join([x for x in [cell.text for cell in temp_rows][0] if x.isnumeric()]))
snowfall = int(''.join([x for x in [cell.text for cell in snowfall_rows][0] if x.isnumeric()]))
depth = int(''.join([x if x.isnumeric() else '0' for x in [cell.text for cell in sdepth_rows][0]]))

temp, snowfall, depth

def get_stats(resort):
    URL = 'http://www.onthesnow.com/'+resort_urls[resort]+'/ski-resort.html'
    browser.get(URL)
    soup = BeautifulSoup(browser.page_source,'html.parser')
    temp_rows = soup.select('p.bluetxt.temp') 
    snowfall_rows = soup.select('p.bluetxt.sfa')
    sdepth_rows = soup.select('p.bluetxt.sd') 
    temp = int(''.join([x for x in [cell.text for cell in temp_rows][0] if x.isnumeric()]))
    snowfall = int(''.join([x for x in [cell.text for cell in snowfall_rows][0] if x.isnumeric()]))
    depth = int(''.join([x if x.isnumeric() else '0' for x in [cell.text for cell in sdepth_rows][0]]))
    return temp, snowfall, depth

get_stats('Winter Park')

for resort in resort_urls:
    print(get_stats(resort))

stuff = soup.select('table.ovv_info tbody tr td')

rows = [cell.text for cell in stuff]
rows

_, elevs, colors, lifts, price = rows

bottom = int(''.join([x for x in elevs.split()[0] if x.isnumeric()]))
top = int(''.join([x for x in elevs.split()[2] if x.isnumeric()]))
bottom, top

greens = int(''.join([x for x in colors.split()[0] if x.isnumeric()]))
blues = int(''.join([x for x in colors.split()[1] if x.isnumeric()]))
blacks = int(''.join([x for x in colors.split()[2] if x.isnumeric()]))
bbs = int(''.join([x for x in colors.split()[3] if x.isnumeric()]))
greens, blues, blacks, bbs

lifts = int(lifts)
lifts

price.split()

price_split = float([x for x in price.split() if x.startswith('US')][1][3:])
price_split

def get_elevs_colors_lifts_price(resort):
    URL = 'http://www.onthesnow.com/'+resort_urls[resort]+'/ski-resort.html'
    browser.get(URL)
    soup = BeautifulSoup(browser.page_source,'html.parser')
    stuff = soup.select('table.ovv_info tbody tr td')
    rows = [cell.text for cell in stuff]
    _, elevs, colors, lifts, price = rows
    bottom = int(''.join([x for x in elevs.split()[0] if x.isnumeric()]))
    top = int(''.join([x for x in elevs.split()[2] if x.isnumeric()]))
    greens = int(''.join([x for x in colors.split()[0] if x.isnumeric()]))
    blues = int(''.join([x for x in colors.split()[1] if x.isnumeric()]))
    blacks = int(''.join([x for x in colors.split()[2] if x.isnumeric()]))
    bbs = int(''.join([x for x in colors.split()[3] if x.isnumeric()]))
    lifts = int(lifts)
    price = int(''.join([x if x.isnumeric() else '0' for x in price.split()[0]]))
    return [bottom, top, greens, blues, blacks, bbs, lifts, price]

elevs_colors_lifts_price = {}
for resort in resort_urls:
    elevs_colors_lifts_price[resort] = get_elevs_colors_lifts_price(resort)
    print(elevs_colors_lifts_price[resort])



