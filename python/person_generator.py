import random
import requests
import re as re
from bs4 import BeautifulSoup
import pickle
import csv

# random integers
random.randint(1,30)

def names_from_search(soup):
    name_regex = r'>(\w*)<'
    try:
        text = soup.find_all(class_="p1")
        text = str(text)
        names = re.findall(name_regex, text)
        return names
    except:
        return None

#boy names
url = ['https://www.babble.com/pregnancy/1000-most-popular-boy-names/']
for i in range(len(url)):
    response = requests.get(url[i])
    page = response.text
    soup = BeautifulSoup(page)
    boy_names = names_from_search(soup)

print len(boy_names)

# girl names
url = ['https://www.babble.com/pregnancy/1000-most-popular-girl-names/']
for i in range(len(url)):
    response = requests.get(url[i])
    page = response.text
    soup = BeautifulSoup(page)
    girl_names = names_from_search(soup)
    
print len(girl_names)

# looks like we have some extra entries, possibly blanks
new_girl_names = []
for i in girl_names:
    if len(i) > 2:
        new_girl_names.append(i)
print len(new_girl_names)

# now we add male and female names together in a single list
first_names = new_girl_names + boy_names
print len(first_names)
first_names

csvfile = 'firstnames.csv'

with open(csvfile, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    for val in first_names:
        writer.writerow([val])

def surnames_from_search(soup):
    surname_regex = r'>([A-Z][a-z]*)<'
    try:
        text = soup.find_all("td")
        text = str(text)
        names = re.findall(surname_regex, text)
        return names
    except:
        return None
    
url = ['http://surnames.behindthename.com/top/lists/united-states/1990']
for i in range(len(url)):
    response = requests.get(url[i])
    page = response.text
    soup = BeautifulSoup(page)
    surnames = surnames_from_search(soup)

print len(surnames)    
surnames

csvfile = 'surnames.csv'

with open(csvfile, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    for val in surnames:
        writer.writerow([val])

f = open('us_postal_codes.csv')
csv_f = csv.reader(f)

zip_codes = []

for row in csv_f:
    zip_codes.append(row[0])

def zipcode():
    return zip_codes[random.randint(0,len(zip_codes))]

zipcode()

#### first names

f = open('firstnames.csv')
csv_f = csv.reader(f)

first_names = []
for row in csv_f:
    first_names.append(row[0])
    
print len(first_names)

#### surnames

f = open('surnames.csv')
csv_f = csv.reader(f)

surnames = []

for row in csv_f:
    surnames.append(row[0])
    
print len(surnames)

list_of_domains_1 = ['@aol.com', '@twc.com', '@gmail.com', '@yahoo.com', '@verizon.net', '@att.com', '@mail.com',
                    '@email.cz', '@hotmail.com', '@outlook.com', '@mail.ru']
                  
list_of_domains_2 = ['@whitehouse.gov', '@horizon.net', '@aresmacrotech.com', '@mct.com', '@wuxing.com', '@neonet.com', '@ucas.gov', 
                    '@hentai.jp', '@evo.ru', '@gazprom.ru', '@gizoogle.com',  '@shell.com', '@exxonmobil.com', 
                    '@omega.com', '@shinto.jp', '@baidu.com', '@cas.gov', '@kkk.org', '@breitbart.com', '@koch.net',
                    '@gs.com', '@mailchimp.com', '@email.cz', '@fox.com', '@pornhub.com', '@ravensfans.com',
                    '@ohiostate.edu', '@lsu.edu', '@alabama.edu', '@spinradindustries.com', '@larpersunited.org', 
                    '@theonion.com', '@alpha.com','@doe.gov', '@tupaclives.net', '@nltk.com', '@redtube.com',
                    '@harvard.edu', '@yale.edu', '@fordham.edu', '@penn.edu']

def pick_domain():
    if random.randint(0,100) >= 90:
        return list_of_domains_2[random.randint(0,len(list_of_domains_2))]
    else:
        return list_of_domains_1[random.randint(0,len(list_of_domains_1))]
    
def personal_info():
    person = str(first_names[random.randint(0,len(surnames))]+ ' ' + surnames[random.randint(0,len(surnames))])
    number = str(random.randint(0,1000))
    domain = pick_domain()
    return [person, person.replace(' ','.').lower()+number+domain, zipcode()]

personal_info()

personal_info()



