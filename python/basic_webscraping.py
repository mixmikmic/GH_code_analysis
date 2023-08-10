URL = "http://www.soc.cornell.edu/people/faculty/"

import requests
from bs4 import BeautifulSoup as BS

html = requests.get(URL)

html.content

soup = BS(html.content, "html.parser")



def getSoup(url):
    html = requests.get(url)
    soup = BS(html.content, "html.parser")
    return soup

links = soup.findAll('a', href=True) #Finds all 'a' tags with an href object (i.e. all hyperlinks)

links

#Let's take a look at one of these items 
links[20]

type(links[20])

dir(links[20])

x = links[20]

x.contents

x['href']

profiles = []
for l in links:
    if "/people/faculty/" in l['href']:
        profiles.append(l['href'])

profiles

##We can remove the incorrect links by applying a conditional filter to profiles
profiles = [x for x in profiles if x.endswith('faculty/') == False]

profiles

#Note that there are many duplicates in the list...
print(len(profiles))
print(len(set(profiles)))

profiles = list(set(profiles))

from time import sleep
profile_contents = {}
for p in profiles:
    print("Getting information from: ", p)
    sleep(1) #Sleeping for a time interval so we're not querying too frequently
    soup = getSoup(p)
    name = p.split('/')[-2]
    profile_contents[name] = soup

print(profile_contents.keys())

#If we want to get the information for a particular professor we can look up their dictionary entry
macy = profile_contents['macy']
macy

macy.find('div', {'class': 'entry-content'})

content = macy.find('div', {'class': 'entry-content'})
content.text

content_refined = content.findAll('h4')

content_refined[0]

titles = content_refined[0].text

titles.split('PhD')

title_and_education = titles.split('PhD')

title = title_and_education[0]
education = title_and_education[1]
education = 'PhD'+education

title

education

def getFacultyInfo(soup):
    info = soup.find('div', {'class': 'entry-content'})
    return info

def getTitleAndEducation(info):
    info_refined = info.findAll('h4')
    titles = info_refined[0].text
    title_and_education = titles.split('PhD')
    title = title_and_education[0]
    education = 'PhD'+title_and_education[1]
    return title, education

macy = getFacultyInfo(profile_contents['macy'])
macy_te = getTitleAndEducation(macy)
print(macy_te[0], macy_te[1])

heckathorn = getFacultyInfo(profile_contents['heckathorn'])
heckathorn_te = getTitleAndEducation(heckathorn)
print(heckathorn_te[0], heckathorn_te[1])

garip = getFacultyInfo(profile_contents['garip'])
garip_te = getTitleAndEducation(garip)
print(garip_te[0], garip_te[1])

garip

import string

def getTitleAndEducation2(info):
    info_refined = info.findAll('h4')
    titles = info_refined[0].text
    titles = ''.join(x for x in titles if x not in string.punctuation)
    title_and_education = titles.split('PhD')
    title = title_and_education[0].rstrip()
    education = 'PhD'+title_and_education[1]
    education = education.split('Curriculum')[0].rstrip() #removing additional info and whitespace
    return title, education

getTitleAndEducation2(garip)

for prof in profile_contents:
    print("Getting info for: ", prof)
    try:
        info = getFacultyInfo(profile_contents[prof])
        te = getTitleAndEducation(info)
        print(prof, te[0], te[1], '\n')
    except:
        print("ERROR: Failed to get info from", prof)
    sleep(1)



def getFacultyName(soup):
    name_info = soup.findAll('h1', {'class':'entry-title'})
    name = name_info[0].text
    return name

for prof in profile_contents:
    name = getFacultyName(profile_contents[prof])
    print(name)

faculty_info = {}
for prof in profile_contents:
    print("Getting info for: ", prof)
    try:
        name = getFacultyName(profile_contents[prof])
        info = getFacultyInfo(profile_contents[prof])
        te = getTitleAndEducation2(info)
        print(te)
        faculty_info[name] = {'title': te[0], 'education':te[1]}
    except:
        print("ERROR: Failed to get info from", prof)
    

faculty_info



import pandas as pd
df = pd.DataFrame.from_dict(faculty_info, orient='index')

df

df.to_csv('../data/facultyinfo.csv',encoding='utf-8')









