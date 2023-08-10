import numpy as np
import pandas as pd

np.random.seed(42)

a = np.random.randn(3, 4)
a[2][2] = np.nan
print(a)
np.savetxt('np.csv', a, fmt='%.2f', delimiter=',', header=" #1, #2,  #3,  #4")
df = pd.DataFrame(a)
print(df)
df.to_csv('pd.csv', float_format='%.2f', na_rep="NAN!")

import numpy as np
import pandas as pd
from tempfile import NamedTemporaryFile
from os.path import getsize

np.random.seed(42)
a = np.random.randn(365, 4)

tmpf = NamedTemporaryFile()
np.savetxt(tmpf, a, delimiter=',')
print("Size CSV file", getsize(tmpf.name))

tmpf = NamedTemporaryFile()
np.save(tmpf, a)
tmpf.seek(0)
loaded = np.load(tmpf)
print("Shape", loaded.shape)
print("Size .npy file", getsize(tmpf.name))

df = pd.DataFrame(a)
df.to_pickle(tmpf.name)
print("Size pickled dataframe", getsize(tmpf.name))
print("DF from pickle\n", pd.read_pickle(tmpf.name))

import numpy as np
import tables
from tempfile import NamedTemporaryFile
from os.path import getsize

np.random.seed(42)
a = np.random.randn(365, 4)

tmpf = NamedTemporaryFile()
h5file = tables.open_file(tmpf.name, mode='w', title="NumPy Array")
root = h5file.root
h5file.create_array(root, "array", a)
h5file.close()

h5file = tables.open_file(tmpf.name, "r")
print(getsize(tmpf.name))

for node in h5file.root:
   b = node.read()
   print(type(b), b.shape)

h5file.close()

import numpy as np
import pandas as pd
from tempfile import NamedTemporaryFile

np.random.seed(42)
a = np.random.randn(365, 4)

tmpf = NamedTemporaryFile()
store = pd.io.pytables.HDFStore(tmpf.name)
print(store)

df = pd.DataFrame(a)
store['df'] = df
print(store)

print("Get", store.get('df').shape)
print("Lookup", store['df'].shape)
print( "Dotted", store.df.shape)

del store['df']
print("After del\n", store)

print("Before close", store.is_open)
store.close()
print("After close", store.is_open)

df.to_hdf('test.h5', 'data', format='table')
print(pd.read_hdf('test.h5', 'data', where=['index>363']))

import numpy as np
import pandas as pd
from tempfile import NamedTemporaryFile

np.random.seed(42)
a = np.random.randn(365, 4)

tmpf = NamedTemporaryFile(suffix='.xlsx')
df = pd.DataFrame(a)
print(tmpf.name)
df.to_excel(tmpf.name, sheet_name='Random Data')
print("Means\n", pd.read_excel(tmpf.name, 'Random Data').mean())

import json

json_str = '{"country":"Netherlands","dma_code":"0","timezone":"Europe\/Amsterdam","area_code":"0","ip":"46.19.37.108","asn":"AS196752","continent_code":"EU","isp":"Tilaa V.O.F.","longitude":5.75,"latitude":52.5,"country_code":"NL","country_code3":"NLD"}'

data = json.loads(json_str)
print("Country", data["country"])
data["country"] = "Brazil"
print(json.dumps(data))

import pandas as pd

json_str = '{"country":"Netherlands","dma_code":"0","timezone":"Europe\/Amsterdam","area_code":"0","ip":"46.19.37.108","asn":"AS196752","continent_code":"EU","isp":"Tilaa V.O.F.","longitude":5.75,"latitude":52.5,"country_code":"NL","country_code3":"NLD"}'

data = pd.read_json(json_str, typ='series')
print("Series\n", data)

data["country"] = "Brazil"
print("New Series\n", data.to_json())

import feedparser as fp

rss = fp.parse("http://www.packtpub.com/rss.xml")

print("# Entries", len(rss.entries))

for i, entry in enumerate(rss.entries):
   if "Java" in entry.summary:
      print(i, entry.title)
      print(entry.summary)

from bs4 import BeautifulSoup
import re

soup = BeautifulSoup(open('loremIpsum.html'),"lxml")

print("First div\n", soup.div)
print("First div class", soup.div['class'])

print("First dfn text", soup.dl.dt.dfn.text)

for link in soup.find_all('a'):
   print("Link text", link.string, "URL", link.get('href'))

# Omitting find_all
for i, div in enumerate(soup('div')):
   print(i, div.contents)


#Div with id=official
official_div = soup.find_all("div", id="official")
print("Official Version", official_div[0].contents[2].strip())

print("# elements with class", len(soup.find_all(class_=True)))

tile_class = soup.find_all("div", class_="tile")
print("# Tile classes", len(tile_class))

print("# Divs with class containing tile", len(soup.find_all("div", class_=re.compile("tile"))))

print("Using CSS selector\n", soup.select('div.notile'))
print("Selecting ordered list list items\n", soup.select("ol > li")[:2])
print("Second list item in ordered list", soup.select("ol > li:nth-of-type(2)"))

print("Searching for text string", soup.find_all(text=re.compile("2014")))



