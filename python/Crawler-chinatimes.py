for page in range(1, 10):
    print("http://www.chinatimes.com/realtimenews?page={}".format(page))

import urllib.request

domain = "http://www.chinatimes.com"
url = "http://www.chinatimes.com/realtimenews?page=1"

o = urllib.request.urlopen(url)
h = o.read(1024)
html = b""
while h:
    h = o.read(4096)
    html += h
print(html.decode('utf-8'))

from bs4 import BeautifulSoup
soup = BeautifulSoup(html, 'html.parser')

for a in soup.find_all('a'):
    if 'class' in a.attrs:
        if a.attrs['class'] == ['']:
            print("{}{}".format(domain, a.attrs['href']))

