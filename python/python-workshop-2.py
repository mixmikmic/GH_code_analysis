import requests
import pandas as pd
import lxml.html as lx

url = "https://en.wikipedia.org/wiki/California_State_Legislature,_2017%E2%80%9318_session"
#url = "https://www.google.com/dgagegeh"

response = requests.get(url)
response.raise_for_status()

html = lx.fromstring(response.text)
html.make_links_absolute(url)

tables = html.cssselect("table")
table = tables[5]

links = table.cssselect("tr td:nth-of-type(3) a")

senator_links = [link.get("href") for link in links]



import pandas as pd
import time

#url_senator = senator_links[0]

def scrape_bio(url_senator):
    print(url_senator)
    response = requests.get(url_senator)
    response.raise_for_status()

    table = pd.read_html(response.text, attrs={"class": "infobox vcard"})[0]

    name = table.iloc[0, 0]
    name

    has_born = table.iloc[:, 0].str.contains("Born", na = False)
    #print(has_born)
    born = table.iloc[:, 1].loc[has_born].values[0]
    
    time.sleep(0.5)
    
    return {"name": name, "born": born}

bios = [scrape_bio(u) for u in senator_links]

pd.DataFrame(bios)



senators = pd.read_csv("senators.csv")
senators.head()

senators.born[0]

senators["age"] = senators.born.str.rsplit(")").str.get(-2).str.rsplit().str.get(-1)
senators["age"] = pd.to_numeric(senators["age"])

import bokeh.charts as bkh
bkh.output_notebook()

plt = bkh.Histogram(senators, "age")
bkh.show(plt)

