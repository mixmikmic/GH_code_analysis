import bs4
import requests

url = "http://www.citador.pt/frases/citacoes/a/gustavo-santos/{}"

quotes_dump = list()
for counter, i in enumerate(range(1, 200, 10)):
    # get content from generated Url
    print("Request url \"{}\"".format(url.format(i)))
    response = requests.get(url.format(i))
    if not str(response.status_code).startswith("2"):
        continue
    
    # get tag and body content
    soup = bs4.BeautifulSoup(response.text)
    divs = soup.findAll('div', {'class': "panel panel-default"})
    for div in divs:
        tag = div.find('div', {'class': 'panel-heading'}).text.strip()
        body = div.find('div', {'class': 'panel-body'}).findAll('div')[0].text.strip()
        quotes_dump.append((counter, tag.lower(), body))

trigger_words = dict((tag[1], {'quotes': [], 'last_used_at': None}) for tag in quotes_dump)
for counter, tag, body in quotes_dump:
    updated_quotes = trigger_words[tag]['quotes']
    updated_quotes.append(counter)
    updated_quotes = list(set(updated_quotes))
    trigger_words[tag]['quotes'] = updated_quotes

quotes_dump




