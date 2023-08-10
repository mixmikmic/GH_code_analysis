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

soup.find_all("a")

soup.find_all("a")
soup("a")

# Get only the 'a' tags in 'sidemenu' class
soup("a", class_="sidemenu")

# get elements with "a.sidemenu" CSS Selector.
soup.select("a.sidemenu")

# SOLUTION
soup.select("a.mainmenu")

# this is a list
soup.select("a.sidemenu")

# we first want to get an individual tag object
first_link = soup.select("a.sidemenu")[0]

# check out its class
type(first_link)

print(first_link.text)

print(first_link['href'])

# SOLUTION
[link['href'] for link in soup.select("a.mainmenu")]

# make a GET request
req = requests.get('http://www.ilga.gov/senate/default.asp?GA=98')
# read the content of the server’s response
src = req.text
# soup it
soup = BeautifulSoup(src, "lxml")

# get all tr elements
rows = soup.find_all("tr")
len(rows)

# returns every ‘tr tr tr’ css selector in the page
rows = soup.select('tr tr tr')
print(rows[2].prettify())

# select only those 'td' tags with class 'detail'
row = rows[2]
detailCells = row.select('td.detail')
detailCells

# Keep only the text in each of those cells
rowData = [cell.text for cell in detailCells]

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

members[1]

# SOLUTION

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
    
    # add href
    href = row.select('a')[1]['href']
    
    # add full path
    full_path = "http://www.ilga.gov/senate/" + href + "&Primary=True"
    
    # Store in a tuple
    tup = (name,district,party, full_path)
    
    # Append to list
    members.append(tup)

members[:5]

# SOLUTION
def get_members(url):
    src = requests.get(url).text
    soup = BeautifulSoup(src, "lxml")
    rows = soup.select('tr')
    members = []
    for row in rows:
        detailCells = row.select('td.detail')
        if len(detailCells) is not 5:
            continue
        rowData = [cell.text for cell in detailCells]
        name = rowData[0]
        district = int(rowData[3])
        party = rowData[4]
        href = row.select('a')[1]['href']
        full_path = "http://www.ilga.gov/senate/" + href + "&Primary=True"
        tup = (name,district,party,full_path)
        members.append(tup)
    return(members)

# Test you code!
senateMembers = get_members('http://www.ilga.gov/senate/default.asp?GA=98')
len(senateMembers)

# SOLUTION
def get_bills(url):
    src = requests.get(url).text
    soup = BeautifulSoup(src, "lxml")
    rows = soup.select('tr tr tr')
    bills = []
    rowData = []
    for row in rows:
        detailCells = row.select('td.billlist')
        if len(detailCells) is not 5:
            continue
        rowData = [cell.text for cell in row]
        bill_id = rowData[0]
        description = rowData[2]
        champber = rowData[3]
        last_action = rowData[4]
        last_action_date = rowData[5] 
        tup = (bill_id,description,champber,last_action,last_action_date)
        bills.append(tup)
    return(bills)

# uncomment to test your code:
test_url = senateMembers[0][3]
get_bills(test_url)[0:5]

# SOLUTION
bills_dict = {}
for member in senateMembers[:5]:
    bills_dict[member[1]] = get_bills(member[3])
    time.sleep(0.5)

bills_dict[52]



