from bs4 import BeautifulSoup
import requests

source = """
<!DOCTYPE html>  
<html>  
  <head>
    <title>Scraping</title>
  </head>
  <body class="col-sm-12">
    <h1>section1</h1>
    <p>paragraph1</p>
    <p>paragraph2</p>
    <div class="col-sm-2">
      <h2>section2</h2>
      <p>paragraph3</p>
      <p>unclosed
    </div>
  </body>
</html>  
"""

soup = BeautifulSoup(source, "html.parser")

print 'Head:'
print '', soup.find_all("head")
# [<head>\n<title>Scraping</title>\n</head>]

print '\nType of head:'
print '', map(type, soup.find_all("head"))
# [<class 'bs4.element.Tag'>]

print '\nTitle tag:'
print '', soup.find("title")
# <title>Scraping</title>

print '\nTitle text:'
print '', soup.find("title").text
# Scraping

divs = soup.find_all("div", attrs={"class": "col-sm-2"})
print '\nDiv with class=col-sm-2:'
print '', divs
# [<div class="col-sm-2">....</div>]

print '\nClass of first div:'
print '', divs[0].attrs['class']
# [u'col-sm-2']

print '\nAll paragraphs:'
print '', soup.find_all("p")
# [<p>paragraph1</p>, 
#  <p>paragraph2</p>, 
#  <p>paragraph3</p>, 
#  <p>unclosed\n    </p>]

url = 'https://www.theguardian.com/technology/2017/jan/31/amazon-expedia-microsoft-support-washington-action-against-donald-trump-travel-ban'
req = requests.get(url)
source = req.text
soup = BeautifulSoup(source, 'html.parser')

links = soup.find_all('a')
print links

links = soup.find_all('a', attrs={
    'data-component': 'auto-linked-tag'
})

for link in links: 
    print link['href'], link.text

url = 'https://www.theguardian.com/uk/technology'
req = requests.get(url)
source = req.text
soup = BeautifulSoup(source, 'html.parser')

articles = soup.find_all('a', attrs={
    'class': 'js-headline-text'
})

for article in articles: 
    print article['href'][:70], article.text[:20]



