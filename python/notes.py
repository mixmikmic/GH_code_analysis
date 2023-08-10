import lxml.html as lx
import numpy as np
import pandas as pd
import requests
import time

url = "https://en.wikipedia.org/wiki/California_State_Legislature,_2017%E2%80%9318_session"

response = requests.get(url)
# .raise_for_status() throws an error if the status code is not OK.
response.raise_for_status()

html = lx.fromstring(response.text)
# .make_links_absolute() converts all URLs in the page to absolute URLs.
html.make_links_absolute(response.url)

# The senators are listed in the 5th table on the page.
tab = html.cssselect("table")[5]
# The links are in the 3rd column of the table.    
links = tab.cssselect("tr td:nth-of-type(3) a")
links = [link.get("href") for link in links]

links

def scrape_bio(url):
    print(url)
    
    response = requests.get(url)
    response.raise_for_status()

    # read_html() converts all tables in a page to data frames.
    tables = pd.read_html(response.text, attrs = {"class": "infobox"})
    bio = tables[0]

    name = bio.iloc[0, 0]
    
    has_senate = bio.iloc[:, 0].str.contains("Senate", na = False)
    term = bio.loc[has_senate].iloc[0, 0]
    
    has_born = bio.iloc[:, 0].str.startswith("Born", na = False)
    born = bio.loc[has_born].iloc[0, 1]
    
    has_party = bio.iloc[:, 0].str.startswith("Political party", na = False)
    party = bio.loc[has_party].iloc[0, 1]
    
    # Wikipedia blocks IPs that request pages too quickly, so slow down.
    time.sleep(0.5)
    
    return {"name": name, "term": term, "born": born, "party": party}

scrape_bio(links[0])

senators = pd.DataFrame([scrape_bio(u) for u in links])
senators.head()

senators.to_csv("senators.csv", index = False)

# .str reminds Pandas that this is a string column.
# .rsplit() splits a string into several pieces starting from the right side.
# .get() gets an element from a list inside a data frame.
senators["age"] = senators.born.str.rsplit(")").str.get(-2).str.rsplit().str.get(-1)
senators["age"] = pd.to_numeric(senators.age)
senators.head()

# output_notebook() sets up Bokeh to display in the Jupyter notebook.
bkh.output_notebook()

plt = bkh.Histogram(senators, values = "age")
bkh.show(plt)

