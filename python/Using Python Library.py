# import library
import requests

# now fetch google page with it and store to resp object
resp = requests.get('https://www.google.com')

# this can be written as well
resp = requests.api.get('https://www.google.com')

# show what methods/attributes
dir(resp)

# show status code
resp.status_code

resp.headers

resp.text



# import can be done this way
from bs4 import BeautifulSoup

# now lets parse obtained html file
soup = BeautifulSoup(resp.text, 'lxml')

# find first a tag
soup.find('a')

# find all span
soup.find_all('span')

for each in resp.history:
    print(each.url)





class SomeClass:
    def __init__(self):
        pass

