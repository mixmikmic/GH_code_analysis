# Python 2/3 comaptibility
from __future__ import print_function

import ipywidgets

url = "https://news.ycombinator.com/robots.txt"
iframe = '<iframe src=' + url + ' width=700 height=200></iframe>'
ipywidgets.HTML(iframe)

import pandas as pd

tables = pd.read_html("https://news.ycombinator.com")

len(tables)

c

tables[2].head()

tables[2].head()

## Your Code here.  We will be around to help



