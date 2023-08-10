from bs4 import BeautifulSoup
import urllib

# set the url we want to visit
url = "http://www.opentable.com/washington-dc-restaurant-listings"

# visit that url, and grab the html of said page
html = urllib.urlopen(url).read()

len(html)

html[0:1000]

# This is the raw HTML of the page.

# we need to convert this into a soup object
soup = BeautifulSoup(html, 'html.parser', from_encoding="utf-8")

# print the restaurant names
print soup.find_all('span', {'class': 'rest-row-name-text'})[0:20]

r_names = []
# for each element you find, print out the restaurant name
for entry in soup.find_all('span', {'class': 'rest-row-name-text'}):
    r_names.append(entry.renderContents())

r_names[0:20]

# first, see if you can identify the location for all elements -- print it out
print soup.find_all('span', {'class': 'rest-row-meta--location rest-row-meta-text'})[0:5]

r_loc = []
for entry in soup.find_all('span', {'class': 'rest-row-meta--location rest-row-meta-text'}):
    r_loc.append(entry.renderContents())
    
r_loc[0:10]

# print out all prices
print soup.find_all('div', {'class': 'rest-row-pricing'})[0:5]

r_dollars = []
# get EACH number of dollar signs per restaurant
# this one is trickier to eliminate the html. Hint: try a nested find
for entry in soup.find_all('div', {'class': 'rest-row-pricing'}):
    r_dollars.append(entry.find('i').renderContents())
    
r_dollars[0:10]

r_dollar_count = []

for entry in soup.find_all('div', {'class': 'rest-row-pricing'}):
    price = entry.find('i').renderContents()
    r_dollar_count.append(price.count('$'))
    
r_dollar_count[0:10]

# print out all objects that contain the number of times the restaurant was booked
print soup.find_all('span', {'class': 'tadpole'})[0:20]

# let's first try printing out all 'span' class objects
for entry in soup.find_all('span')[0:30]:
    print entry

# Can't find the booking count in the object. This requires javascript.

# import
from selenium import webdriver

# create a driver called driver
driver = webdriver.Chrome(executable_path="../chromedriver/chromedriver")

# close it
driver.close()

# let's boot it up, and visit a URL of our choice
driver = webdriver.Chrome(executable_path="../chromedriver/chromedriver")
driver.get("http://www.python.org")

# visit our OpenTable page
driver = webdriver.Chrome(executable_path="../chromedriver/chromedriver")
driver.get("http://www.opentable.com/washington-dc-restaurant-listings")
# # always good to check we've got the page we think we do
# assert "OpenTable" in driver.title

# import sleep
from time import sleep

# visit our relevant page
driver = webdriver.Chrome(executable_path="../chromedriver/chromedriver")
driver.get("http://www.opentable.com/washington-dc-restaurant-listings")
# wait one second
sleep(1)
#grab the page source
html = driver.page_source

# BeautifulSoup it!
html = BeautifulSoup(html, 'lxml')

# Now, let's return to our earlier problem: how do we locate bookings on the page?

# print out the number bookings for all restaurants
print html.find_all('div', {'class':'booking'})[0:10]

r_bookings = []
for booking in html.find_all('div', {'class':'booking'}):
    r_bookings.append(booking.text)
    
r_bookings[0:15]

# We've succeeded!

# But we can clean this up a little bit. 
# We're going to use regular expressions (regex) to grab only the 
# digits that are available in each of the text.

# The best way to get good at regex is to, well, just keep trying and testing: http://pythex.org/

# import regex
import re

# Given we haven't covered regex, I'll show you how to use the search function to match any given digit.

r_bookings_num = []
# for each entry, grab the text
for booking in html.find_all('div', {'class':'booking'}):
    # match all digits
    match = re.search(r'\d+', booking.text)
    # append if found
    if match:
        r_bookings_num.append(int(match.group()))
    # otherwise 0
    else:
        r_bookings_num.append(0)
        
r_bookings_num[0:15]

# print out all entries
entries = html.find_all('div', {'class':'result content-section-list-row cf with-times'})

# I did this previously. I know for a fact that not every element has a 
# number of recent bookings. That's probably exactly why OpenTable houses 
# this in JavaScript: they want to continously update the number of bookings 
# with the most relevant number of values.

# what happens when a booking is not available?
# print out some booking entries, using the identification code we wrote above
for entry in html.find_all('div', {'class':'result content-section-list-row cf with-times'})[0:50]:
    print entry.find('div', {'class':'booking'})

# if we find the element we want, we print it. Otherwise, we print 'ZERO'
entries = []
for entry in html.find_all('div', {'class':'result content-section-list-row cf with-times'}):
    try:
        entries.append(entry.find('div', {'class':'booking'}).text)
    except:
        entries.append('ZERO')
        
print entries.count('ZERO')

# I'm going to create my empty df first
import pandas as pd
dc_eats = pd.DataFrame(columns=["name","location","price","bookings"])

# loop through each entry
for entry in html.find_all('div', {'class':'result content-section-list-row cf with-times'}):
    # grab the name
    name = entry.find('span', {'class': 'rest-row-name-text'}).text
    # grab the location
    location = entry.find('span', {'class': 'rest-row-meta--location rest-row-meta-text'}).renderContents()
    # grab the price
    price = entry.find('div', {'class': 'rest-row-pricing'}).find('i').renderContents().count('$')
    # try to find the number of bookings
    try:
        temp = entry.find('div', {'class':'booking'}).text
        match = re.search(r'\d+', temp)
        if match:
            bookings = match.group()
    except:
        bookings = 'NA'
    # add to df
    dc_eats.loc[len(dc_eats)]=[name, location, price, bookings]

# check out our work
dc_eats.head()

# we can send keys as well
# import
from selenium.webdriver.common.keys import Keys

# open the driver
driver = webdriver.Chrome(executable_path="../chromedriver/chromedriver")
# visit Python
driver.get("http://www.python.org")
# verify we're in the right place
assert "Python" in driver.title

# find the search position
elem = driver.find_element_by_name("q")
# clear it
elem.clear()
# type in pycon
elem.send_keys("pycon")

# send those keys
elem.send_keys(Keys.RETURN)
# no results
assert "No results found." not in driver.page_source

# close
driver.close()

# all at once:
driver = webdriver.Chrome(executable_path="../chromedriver/chromedriver")
driver.get("http://www.python.org")
assert "Python" in driver.title
elem = driver.find_element_by_name("q")
elem.clear()
elem.send_keys("pycon")
elem.send_keys(Keys.RETURN)
assert "No results found." not in driver.page_source
driver.close()

