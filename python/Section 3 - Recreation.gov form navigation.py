from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from pyvirtualdisplay import Display
from bs4 import BeautifulSoup
import re
import pandas as pd
import datetime
from datetime import timedelta, date
import sys
import requests

local_ip = '172.17.0.3'
print('starting display')
display = Display(visible=0, size=(1024, 768))
display.start()

print('setting up web browser')
browser = webdriver.Firefox()

def check_recreationgov(facility_url, start_date, stay_length) :
    
    browser.set_window_size(1366, 768)
    
    #browser = mechanize.Browser()
    #user_agent_str = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.134 Safari/537.36"
    #browser.addheaders = [('User-agent', user_agent_str)]	
    end_date = "{:%m/%d/%Y}".format(datetime.datetime.strptime(start_date, "%m/%d/%Y") + timedelta(stay_length))

    print('getting url')
    try:
        browser.get(facility_url)
        browser.set_script_timeout(30)
        browser.set_page_load_timeout(30) # seconds

        #print(browser.page_source)
        #browser.open(facility_url)
    except Exception as ex:
        print("Unable to open url: " + facility_url)
        print(ex)
        return pd.DataFrame()
    
    print('browsing form')
    form = browser.find_element_by_name('unifSearchForm')
    arrival = form.find_element_by_name('arrivalDate')
    departure = form.find_element_by_name('departureDate')

    arrival.send_keys(start_date)
    departure.send_keys(end_date)
    browser.find_element_by_name("submit").click()

    element = WebDriverWait(browser, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "matchSummary")))

    soup = BeautifulSoup(browser.page_source, 'lxml')
    divs = soup.findAll('div', attrs={"class" : 'matchSummary'})
    query_result = divs[0].text
    results = pd.DataFrame()

    results['last_updated'] = ["{:%m/%d/%Y}".format(datetime.date.today())]
    results['start_date'] = start_date
    results['end_date'] = end_date

    availability_info = query_result.split(' ')
    sites_available = availability_info[0]
    #total_sites = availability_info[5]

    results['sites_available'] = [sites_available]
    #results['total_sites'] = [total_sites]
    


    return results


test = check_recreationgov('http://172.17.0.3/reservations.html', '05/23/2016', 3)

test

browser.close()
display.stop()



