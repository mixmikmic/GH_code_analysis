import IPython
import re
from sys import version 
from urllib import urlopen
from bs4 import BeautifulSoup 
print ' Reproducibility conditions for this notebook '.center(85,'-')
print 'Python version:     ' + version
print 'RegEx version:      ' + re.__version__
print 'IPython version:    ' + IPython.__version__
print '-'*85

import re
p = re.compile("[A-Za-z0-9\._+-]+@[A-Za-z]+\.(com|org|edu|net)")
print p

string = 'purple alice-b@mymail.com monkey dishwasher'
match = re.search(p, string)
if match:
    print match.group() 

html_doc = """
<html><head><title>The Dormouse's story</title></head>
<body>
<p class="title"><b>The Dormouse's story</b></p>

<p class="story">Once upon a time there were three little sisters; and their names were
<a href="http://example.com/elsie" class="sister" id="link1">Elsie</a>,
<a href="http://example.com/lacie" class="sister" id="link2">Lacie</a> and
<a href="http://example.com/tillie" class="sister" id="link3">Tillie</a>;
and they lived at the bottom of a well.</p>

<p class="story">...</p>
"""

from bs4 import BeautifulSoup
soup = BeautifulSoup(html_doc, 'html.parser')

print soup.title
print soup.title.name
print soup.title.string

print soup.find(id="link3")

soup.find_all('a')

# One common task is extracting all the URLs found within a page’s <a> tags:

for link in soup.find_all('a'):
    print(link.get('href'))

# Another common task is extracting all the text from a page:

print(soup.get_text())

print(soup.prettify(formatter="minimal"))

from urllib import urlopen
from bs4 import BeautifulSoup 
import re

html = urlopen("https://en.wikipedia.org/wiki/Black–Scholes_model")
bsObj = BeautifulSoup(html)
for link in bsObj.find("div", {"id":"bodyContent"}).findAll("a", 
                        href=re.compile("^(/wiki/)((?!:).)*$")):
    if 'href' in link.attrs:
        print(link.attrs['href'])

#TODO... 

