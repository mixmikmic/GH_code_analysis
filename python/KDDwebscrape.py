from bs4 import BeautifulSoup
import requests
import re
import unicodedata
import string

url = "http://dl.acm.org/citation.cfm?id=2783258&preflayout=flat"

r  = requests.get(url)

data = r.text

soup = BeautifulSoup(data)

#print(soup.prettify()); # remove colon to see the format of the html we are parsing

# builds a list of journal and keynote speaker abstracts; remove the colon below to see the list output

abstracts=[]
for l in soup.find_all('p'):
    abstracts.append(filter(lambda x: x in string.printable,unicodedata.normalize('NFKD', l.get_text()).encode('ascii','ignore').decode('unicode_escape').encode('ascii','ignore'))) 
abstracts;

# similar to code above, builds a list of all authors and speakers
authors=[]
for l in soup.find_all('a', href=re.compile('author_page.cfm')):
    authors.append(filter(lambda x: x in string.printable,unicodedata.normalize('NFKD', l.get_text()).encode('ascii','ignore').decode('unicode_escape').encode('ascii','ignore'))) 

authors = [x for x in authors if x != "View colleagues"] # removes leftover value
#set(authors)
authors; # remove the colon to see the output

# similar to code above, builds a list of titles
titles = []
for l in soup.find_all('a', href=re.compile('citation.cfm')):
    titles.append(filter(lambda x: x in string.printable,unicodedata.normalize('NFKD', l.get_text()).encode('ascii','ignore').decode('unicode_escape').encode('ascii','ignore')))     
titles; # remove colon to see list output

