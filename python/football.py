import io
import zipfile

import networkx
import requests

url = "http://www-personal.umich.edu/~mejn/netdata/football.zip"
response = requests.get(url)
content = io.BytesIO(response.content)
with zipfile.ZipFile(content) as zf:  # zipfile object
    txt = zf.read('football.txt').decode()  # read info file
    gml = zf.read('football.gml').decode()  # read gml data
    # throw away bogus first line with # from mejn files
    gml = gml.split('\n')[1:]

import networkx as nx

G = nx.parse_gml(gml) # parse gml data

print(txt)
# print degree for each team - number of games
for n,d in G.degree():
    print('%s %d' % (n, d))

