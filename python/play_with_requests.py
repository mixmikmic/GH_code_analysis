import requests

r = requests.get('http://www.google.com')

r.status_code

r.text

r = requests.get('http://api.citygridmedia.com/content/places/v2/search/where?what=restaurant&where=chicago,IL&tag=11279&placement=sec-5&publisher=test&format=json')

[(location['name'], location['rating']) for location in r.json()['results']['locations']]

r = requests.get('http://api.citygridmedia.com/content/places/v2/search/where?what=restaurant&where=chicago,IL&tag=11279&placement=sec-5&publisher=test')

r.text

import bs4

soup = bs4.BeautifulSoup(r.text, 'xml')

soup.findAll('location')[0].rating.text

soup.findAll('location')[0]

requests.post('https://hooks.slack.com/services/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', json={'text': 'Hello World!', "icon_emoji": ":nerd_face:", "username": "fun_with_apis part deux"} )



