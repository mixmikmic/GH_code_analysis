# import required modules

import requests
from bs4 import BeautifulSoup

# make a GET request
req = requests.get('')

# read the content of the server’s response
src = req.text

# parse the response into an HTML tree
soup = BeautifulSoup(src, 'lxml')

# take a look
print(soup.prettify()[:1000])

# find all elements in a certain tag

soup.find_all("a")

# Get only the 'a' tags in 'sidemenu' class

soup.find_all("a", class_="sidemenu")

# get elements with "a.sidemenu" CSS Selector

soup.select("a.sidemenu")



