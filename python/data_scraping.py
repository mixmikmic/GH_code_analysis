from bs4 import BeautifulSoup
import requests as rq
from time import sleep
import pandas as pd
import re

# ---------- grab all case urls from respective year directory pages ----------

# base url directory page for each year
base_url = "https://supreme.justia.com/cases/federal/us/year/%s.html"

# base url text page for each case
case_url = "https://supreme.justia.com"
id_url = []

# iterate through years 1946 to 2015
years = range(1946,2016)
for year in years:
    soup = BeautifulSoup(rq.get(base_url % year).text, "lxml")
    results = soup.findAll("div", attrs={"class":"result"})
    
    # collect all case urls on each year page
    for result in results:
        id_url.append(case_url + result.a["href"])
    
    # prevent connection error
    sleep(0.1)

# ---------- visit each case page, scrape syllabus, store data ----------

# initially split page into metadata and text (irregular formatting, some null)
metadata,syllabus,citations,urls=[],[],[],[]

# iterate through unique ids collected above
for url in id_url:
    # go to section of the DOM with text
    soup = BeautifulSoup(rq.get(url).text, "lxml")
    
    # check if syllabus exists
    header = soup.find("ul", attrs={"class":"centered-list clear"})
    exists = False
    
    if header is not None:
        if header.text.lower().find("syllabus") > -1:
            exists = True

        # if syllabus exists, collect text
        if exists:    
            # save name of case
            name = soup.find("h1", attrs={"class":"title"}).text

            page_text = soup.find("span", attrs={"class":"headertext"}) 
            if page_text is None:
                page_text = soup.find("div", attrs={"id":"opinion"})
            
            # collect syllabus text
            syllabus_list = ""
            for index in range(0,len(page_text.findAll("p"))):

                # don't append blank lines or returns
                if page_text.findAll("p")[index] != "":
                    syllabus_list += ((page_text.findAll("p")[index].text) + " ")

            metadata.append(name)
            syllabus.append(syllabus_list)
            citations.append(url.split("/")[-3] + " U.S. " + url.split("/")[-2])
            urls.append(url)
    else:
        continue

# ---------- create dataframe ----------
rawdict = {}
rawdict["full_cite"] = metadata
rawdict["us_cite"] = citations
rawdict["text"] = syllabus
rawdict["url"] = urls
dfclean = pd.DataFrame(rawdict)

# ---------- clean up full_cite column -----------
years,names, = [],[]
for x in range(0,len(dfclean)):
    years.append(re.findall("\s\((.*)",dfclean.full_cite[x])[0][:-1])
    names.append(re.findall(".+?(?=\s\d)",dfclean.full_cite[x])[0])
    
dfclean["year"] = years
dfclean["case"] = names

dfclean.to_csv("final_justia_data_merge.csv", sep=',', encoding='utf-8',index=False)

print "Number of cases scaped from Justia:", len(dfclean)



