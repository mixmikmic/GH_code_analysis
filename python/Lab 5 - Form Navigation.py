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
import config

local_ip = config.LAMP_IP
print('starting display')
display = Display(visible=0, size=(1024, 768))
display.start()

print('setting up web browser')
browser = webdriver.Firefox()

browser.set_window_size(1366, 768)

facility_url = "http://" + config.LAMP_IP + "/reservations.html"

try:
    browser.get(facility_url)
    browser.set_script_timeout(30)
    browser.set_page_load_timeout(30) # seconds
    
except Exception as ex :
    print("Unable to open url: " + facility_url)
    print(ex)

form = browser.find_element_by_name('unifSearchForm')
arrival = form.find_element_by_name('arrivalDate')
departure = form.find_element_by_name('departureDate')

start_date = '06/1/2016'
stay_length = 2
end_date = "{:%m/%d/%Y}".format(datetime.datetime.strptime(start_date, "%m/%d/%Y") + timedelta(stay_length))
print(start_date + " - " + end_date)

arrival.send_keys(start_date)
departure.send_keys(end_date)
browser.find_element_by_name("submit").click()

element = WebDriverWait(browser, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "matchSummary")))
soup = BeautifulSoup(browser.page_source, 'lxml')

divs = soup.findAll('div', attrs={"class" : 'matchSummary'})
query_result = divs[0].text
query_result



