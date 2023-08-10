import math

math.sqrt(2)

import numpy as np

np.sqrt(2)

v = np.array([1, 2])
v

M = np.array([[1, -4], [-3, 8]])
M

# Python only 3.5+
M @ v

M.dot(v)

import glob

glob.glob("*")

glob.glob("*.md")

file = open("README.md")
text = file.read()
file.close()
print(text)

# or to make sure it will get closed
with open("README.md") as file:
    text = file.read()
    print(text)

with open("README_short.md", 'w') as file_to_write:
    file_to_write.write(text[:300])

import requests

r = requests.get("http://p.migdal.pl/")

r.status_code

r.text[:1000]

# https://developers.google.com/maps/documentation/geocoding/start
r = requests.get("https://maps.googleapis.com/maps/api/geocode/json",
                 params={"address": "ul. Miecznikowa 1, Warszawa"})

r.json()

get_ipython().system('pip install tqdm')

from tqdm import tqdm
from time import sleep

sleep(2)

for i in tqdm(range(10)):
    sleep(0.5)



