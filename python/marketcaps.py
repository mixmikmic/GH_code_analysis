from bs4 import BeautifulSoup
import requests
import matplotlib.pyplot as plt
import numpy as np
import datetime
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

url = "https://coinmarketcap.com"
content = requests.get(url).content
soup = BeautifulSoup(page.content, 'html.parser')

table = soup.find('table', {'id': 'currencies'})

data = {}
for row in table.findChildren('tr'):
    ticker = row.findNext('span', {'class': 'currency-symbol'}).text.strip()
    marketcap = row.findNext('td', {'class': 'market-cap'}).text.strip()
    marketcap = float(marketcap)
    data[ticker] = marketcap
    #print "%s %.0f" % (ticker, marketcap)

top20 = sorted(((value,key) for (key,value) in data.items()),reverse=True)[:20]
t = []
v = []
for i in top20:
    t.append(i[1])
    v.append(i[0]/1e9) # in billions

now = datetime.datetime.now()
today = now.strftime("%Y-%m-%d")
fig, ax = plt.subplots(figsize=(12,10))
ax.barh(range(len(t)),v, color='g', height=0.9);
ax.invert_yaxis()
ax.set_yticks(range(len(t)))
ax.set_yticklabels(t)
xs = np.arange(0,135,10)
ax.set_xticks(xs)
ax.set_xlabel('USD Billions')
plt.title('Market cap comparison for top 20 crypto assets %s' % today, size = 14);
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='on', labelbottom='on')
fig.savefig('market_caps_crypto.png', dpi = 200)

import re
t = []
v = []
for line in file('sp500_top_20.txt'):
    d = re.split('\s\s+', line.strip())
    t.append(d[1])
    v.append(float(d[2]))

now = datetime.datetime.now()
today = now.strftime("%Y-%m-%d")
fig, ax = plt.subplots(figsize=(12,10))
ax.barh(range(len(t)),v, color='g', height=0.9);
ax.invert_yaxis()
ax.set_yticks(range(len(t)))
ax.set_yticklabels(t)
xs = np.arange(0,1001,100)
ax.set_xticks(xs)
ax.set_xlabel('USD Billions')
plt.title('Market cap comparison for top 20 S&P500 %s' % today, size = 14);
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='on', labelbottom='on')
fig.savefig('market_caps_sp500.png', dpi = 200)

