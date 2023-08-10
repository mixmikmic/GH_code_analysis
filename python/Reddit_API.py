import requests

data = requests.get(r'http://www.reddit.com/user/narek/comments/.json',headers = {'User-agent': 'Educational bot'})

data = data.json()

from pprint import pprint
pprint(data)

for child in data['data']['children']:
    print child['data']['id'], " ", child['data']['author'],child['data']['body']
    print

