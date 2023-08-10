import requests
from bs4 import BeautifulSoup

### Scrap the HTML and turn into a beautiful soup object

#Create a variable with the url
url = 'http://chrisralbon.com'

# Use requests to get the contents
r = requests.get(url)

# Get the text of the contents
html_content = r.text

# contvert the html content into a beautiful soup object
soup = BeautifulSoup(html_content, 'lxml')

# View the title tag of the soup object
soup.title

# View the string within the title tag
soup.title.string

# View the paragraph tap of the soup
soup.p

soup.title.parent.name

soup.a

soup.find_all('a')[0:5]

soup.p.string

soup.find_all('h2')[0:5]

soup.find_all('a')[0:5]

