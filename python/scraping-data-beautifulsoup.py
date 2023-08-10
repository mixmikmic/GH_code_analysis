from bs4 import BeautifulSoup
import requests

BASE_URL = 'http://en.wikipedia.org'
#Wikipedia will reject requests unless we add
# a user-agent attribute to our http header
HEADERS = {'User-Agent': 'Mozilla/5.0'}

def get_Nobel_soup():
    '''Return a parsed tag tree of our Nobel prize page'''
    response = requests.get(
        BASE_URL + '/wiki/List_of_Nobel_Laureates', headers=HEADERS)
    #return the response parsed by BeautifulSoup
    return BeautifulSoup(response.content, 'lxml') #lxml is one of the parser options

soup = get_Nobel_soup()
soup.find('table', {'class': 'wikitable sortable'})
# this works, but fine is not very robust.  If we change the order
# of the two classes we specified, it won't work if it doesn't match
# the order that the two classes were defined in in the HTML
soup.find('table', {'class': 'sortable wikitable'})

# So instead of using BeautifulSoup's selectors (which are fragile)
# we recommend using lxml's methods instead:
soup.select('table.sortable.wikitable') #lxml uses CSS style selectors ('.' is class, '#' is id, etc.)
# This works no matter the order of the classes and returns an *array* of all the matches

table = soup.select_one('table.sortable.wikitable') #selects just the first one
#print(table)
table.select('th')
# these lxml selectors also support regex and other approaches.

def get_column_titles(table):
    '''Get the Nobel categories from the table header'''
    cols = []
    for th in table.select_one('tr').select('th')[1:]: #loop through table head, ignoring leftmost year column
        link = th.select_one('a')
        if link:
            cols.append({'name': link.text,
                        'href': link.attrs['href']})
        else:
            cols.append({'name':th.text, 'href': None})
    return cols
            
print( get_column_titles(table) )

def get_Nobel_winners(table):
    cols = get_column_titles(table)
    winners = []
    for row in table.select('tr')[1:-1]:
        try:
            year = int(row.select_one('td').text) #Gets first <td>
        except ValueError:
            year = None
        for i, td in enumerate(row.select('td')[1:]):
            for winner in td.select('a'):
                href = winner.attrs['href']
                if not href.startswith('#endnote'):
                    winners.append({
                        'year':year,
                        'category':cols[i]['name'],
                        'name':winner.text,
                        'link':winner.attrs['href']
                    })
    return winners

winners = get_Nobel_winners(table)
print(winners)[:2]

import requests
import requests_cache

requests_cache.install_cache('nobel_pages', backend='sqlite', expire_after=7200)
#use requests as usual...

def get_winner_nationality(w):
    '''scrape bio data from the winner's wikipedia page'''
    response = requests.get('http://en.wikipedia.org' + w['link'], headers=HEADERS)
    soup = BeautifulSoup(response.content, 'lxml')
    person_data = {'name': w['name']}
    attr_rows = soup.select('table.infobox tr') #remember, this is CSS-style selectors
    for tr in attr_rows:
        try:
            attribute = tr.select_one('th').text
            if attribute == 'Nationality':
                person_data[attribute] = tr.select_one('td').text
        except AttributeError:
            pass
    return person_data

# test the get_winner_nationality
wdata = []
# test first 50 winners
for w in winners[:50]:
    wdata.append(get_winner_nationality(w))
missing_nationality = []
for w in wdata:
    # if missing 'Nationality' add to list
    if not w.get('Nationality'):
        missing_nationality.append(w)
print(missing_nationality)



