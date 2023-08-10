import requests

r = requests.get('https://www.timeout.com/hong-kong/restaurants/hong-kong-restaurant-reviews')

r.text

from IPython.display import HTML

# dangerous: following code will ruin your notebook (style)
# Use iframe instead.
#HTML(r.text)

open('timeout.com.html', 'w').write(r.text)

HTML('''
<iframe src="timeout.com.html" width=100% height=500px>
''')

from bs4 import BeautifulSoup

mypage = BeautifulSoup(r.text)

len(mypage.find_all('div', attrs={'class': 'row'}))

items = []
for article in mypage.find_all('article', attrs={'class': 'feature-item'}):
    item = {}
    feature = article.find('div', attrs={'class': 'feature-item__content'})
    row = feature.find('div', attrs={'class': 'row'})
    item['title'] = row.find('h3').text.strip()
    item['rating'] = row.find('div', attrs={'class': 'rating'}).text.strip()[:1]
    item['flags'] = article.find('span', attrs={'class': 'icon_pin'}).text.strip()
    items.append(item)
    #print(feature)
    #break
#items

import pandas as pd

pd.DataFrame(items)





