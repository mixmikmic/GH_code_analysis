# import required modules
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import time
import re
import sys

# make a GET request
req = requests.get('http://www.ilga.gov/senate/default.asp')
# read the content of the server’s response
src = req.text

# parse the response into an HTML tree
soup = BeautifulSoup(src, 'lxml')
# take a look
print(soup.prettify()[:1000])

# find all elements in a certain tag
# these two lines of code are equivilant

# soup.find_all("a")

# soup.find_all("a")
# soup("a")

# Get only the 'a' tags in 'sidemenu' class
soup("a", class_="sidemenu")

# get elements with "a.sidemenu" CSS Selector.
soup.select("a.sidemenu")

# YOUR CODE HERE

# this is a list
soup.select("a.sidemenu")

# we first want to get an individual tag object
first_link = soup.select("a.sidemenu")[0]

# check out its class
type(first_link)

print(first_link.text)

print(first_link['href'])

# YOUR CODE HERE

# make a GET request
req = requests.get('http://www.ilga.gov/senate/default.asp?GA=98')
# read the content of the server’s response
src = req.text
# soup it
soup = BeautifulSoup(src, "lxml")

soup

# get all tr elements
rows = soup.find_all("tr")
len(rows)

rows

# returns every ‘tr tr tr’ css selector in the page
rows = soup.select('tr tr tr')
print(rows[2].prettify())

# select only those 'td' tags with class 'detail'
row = rows[2]
detailCells = row.select('td.detail')
len(detailCells)

detailCells[1].find("a").get("href")

# Keep only the text in each of those cells
rowData = [cell.text for cell in detailCells]

rowData

# check em out
print(rowData[0]) # Name
print(rowData[3]) # district
print(rowData[4]) # party

# make a GET request
req = requests.get('http://www.ilga.gov/senate/default.asp?GA=98')

# read the content of the server’s response
src = req.text

# soup it
soup = BeautifulSoup(src, "lxml")

# Create empty list to store our data
members = []

# returns every ‘tr tr tr’ css selector in the page
rows = soup.select('tr tr tr')

# loop through all rows
for row in rows:
    # select only those 'td' tags with class 'detail'
    detailCells = row.select('td.detail')
    
    # get rid of junk rows
    if len(detailCells) is not 5: 
        continue
        
    # Keep only the text in each of those cells
    rowData = [cell.text for cell in detailCells]
    
    # Collect information
    name = rowData[0]
    district = int(rowData[3])
    party = rowData[4]
    
    # Store in a tuple
    tup = (name,district,party)
    
    # Append to list
    members.append(tup)

len(members)

for m in members:
    print(m)

def get_members(url):

    # make a GET request
    req = requests.get(url)

    # read the content of the server’s response
    src = req.text

    # soup it
    soup = BeautifulSoup(src, "lxml")

    # Create empty list to store our data
    members = []

    # returns every ‘tr tr tr’ css selector in the page
    rows = soup.select('tr tr tr')

    # loop through all rows
    for row in rows:
        # select only those 'td' tags with class 'detail'
        detailCells = row.select('td.detail')

        # get rid of junk rows
        if len(detailCells) is not 5: 
            continue

        url_extension = detailCells[1].find("a").get("href")

        # Keep only the text in each of those cells
        rowData = [cell.text for cell in detailCells]

        # Collect information
        name = rowData[0]
        district = int(rowData[3])
        party = rowData[4]

        # YOUR CODE HERE.

        # stuff here
        # http://www.ilga.gov/senate/SenatorBills.asp?MemberID=1911&GA=98&Primary=True

        base = "http://www.ilga.gov/senate/"
        full_path = base + url_extension 

        # Store in a tuple
        tup = (name, district, party, full_path)

        # Append to list
        members.append(tup)

# Uncomment to test 

members[:5]

# YOUR FUNCTION HERE

def get_members(url):
    # make a GET request
    req = requests.get(url)

    # read the content of the server’s response
    src = req.text

    # soup it
    soup = BeautifulSoup(src, "lxml")

    # Create empty list to store our data
    members = []

    # returns every ‘tr tr tr’ css selector in the page
    rows = soup.select('tr tr tr')

    # loop through all rows
    for row in rows:
        # select only those 'td' tags with class 'detail'
        detailCells = row.select('td.detail')

        # get rid of junk rows
        if len(detailCells) is not 5: 
            continue

        url_extension = detailCells[1].find("a").get("href")

        # Keep only the text in each of those cells
        rowData = [cell.text for cell in detailCells]

        # Collect information
        name = rowData[0]
        district = int(rowData[3])
        party = rowData[4]

        # YOUR CODE HERE.

        # stuff here
        # http://www.ilga.gov/senate/SenatorBills.asp?MemberID=1911&GA=98&Primary=True

        base = "http://www.ilga.gov/senate/"
        full_path = base + url_extension 

        # Store in a tuple
        tup = (name, district, party, full_path)

        # Append to list
        members.append(tup)
        
    return members

# Uncomment to test you3 code!

senateMembers = get_members('http://www.ilga.gov/senate/default.asp?GA=98')
len(senateMembers)

senateMembers[4]

# COMPLETE THIS FUNCTION
def get_bills(url):
    src = requests.get(url).text
    soup = BeautifulSoup(src, "lxml")
    rows = soup.select('tr tr tr')

    bills = []
    for row in rows:
        
        detailCells = row.findAll('td')
        rowData = [cell.text for cell in detailCells]
        
        if len(rowData) == 6:
        
            bill_id = rowData[0]
            description = rowData[2]
            chamber = rowData[3]
            last_action = rowData[4]
            last_action_date = rowData[5]

            tup = (bill_id, description, chamber, last_action, last_action_date)
            bills.append(tup)
            
    bills = bills[1:]

    return(bills)

# uncomment to test your code:
test_url = senateMembers[0][3]
get_bills(test_url)[0:5]

# YOUR CODE HERE

from random import randint
import time

senators_bills = {}

for senator in senateMembers:
    senators_bills[senator[0]] = get_bills(senator[3])
    
    time.sleep(randint(1,3))

senators_bills.keys()

senators_bills["William Delgado"]

from random import randint
for i in range(10):
    print("hello")
    time.sleep(randint(1,3))



