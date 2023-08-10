import requests

dir(requests)



resp = requests.get('https://projecteuler.net/')

resp

resp.status_code

resp.headers

resp.url

resp.encoding

resp.raw

resp.text



from bs4 import BeautifulSoup

euler_dom = BeautifulSoup(resp.text, 'html.parser') # lxml

euler_dom

nav = euler_dom.find(id='nav')

nav

nav.ul

nav.ul.li

nav.ul.li.next_sibling

nav.ul.li.next_element

nav.find_all('a')

nav.find_all('a', string='Archives')

archieves = nav.find('a', string='Archives')

archieves

archieves.attrs['href']

from urllib.parse import urljoin

a_url = urljoin(resp.url, archieves.attrs['href'])

a_url

a_resp= requests.get(a_url)

a_resp.text

a_soup = BeautifulSoup(a_resp.text, 'html.parser')

a_soup.find('table', id='problems_table')

for tr in a_soup.find('table', id='problems_table').find_all('tr'):
    tds = tr.find_all('td')
#     print(tds)
    if len(tds) > 1:
        print(tds[1].string)





resp = requests.post('http://www.nepalstock.com/company', 
                     data={'stock-symbol': 'AHPL'})

resp

resp.text

soup = BeautifulSoup(resp.text, 'html.parser')

soup



soup.find_all('table')

my_table = soup.find_all('table')[0]

my_table

for tr in my_table.find_all('tr'):
    # print(dir(tr))
    print(tr.find_all('td')[0].text)



with requests.sessions as session:
    pass

