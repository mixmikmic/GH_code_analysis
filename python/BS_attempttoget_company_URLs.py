import requests
import bs4
from bs4 import BeautifulSoup
import pandas as pd
import time

import re
import numpy as np

URL = "https://fortune.com/fortune500/list/"
#conducting a request of the stated URL above:
page = requests.get(URL)
#specifying a desired format of “page” using the html parser - this allows python to read the various components of the page, rather than treating it as one long string.
soup = BeautifulSoup(page.text, "html.parser")
#printing soup in a more structured tree format that makes for easier reading
print(soup.prettify())

co_names = soup.find_all("span", attrs= {'class':'column small-5 company-title'})

co_names_str = [tag.get_text() for tag in co_names]
co_names_str

fortune_500_list = pd.read_csv('../Data/fortune500_cos_20170331.csv')

fortune_500_list.head()

company_list = fortune_500_list[['Company Name']]

company_list = company_list.assign(indeed_url= pd.Series(np.nan))
company_list.head()

#url to scrape with company names: https://www.indeed.com/jobs?q=caterpillar&l=
#replace "caterpillar" with whatever company I want to search, use "+" for spaces, "%27" for an apostrophe, "%2C" for comma

def search_text_parse (strng):
    return strng.replace(' ', '+').replace(",","%2C").replace("'", "%27").replace("&","%26")

search_text_parse("testing the function of this thing's fun")

def cmp_name(strng):
    return strng.replace(' ', '-').replace(",","").replace(".", "")

cmp_name('testing this thing, with. a text')

import time
import random

i = 0
#for loop to come
for i in range(len(company_list)):
    
    #if np.isnan(company_list['indeed_url'][i]):
    if not isinstance(company_list.loc[i,'indeed_url'], str):
        
        co_name = company_list['Company Name'][i]
        co_name_parsed = search_text_parse(co_name)

        text = requests.get('https://www.indeed.com/jobs?q=' + co_name_parsed + '&l=').text
        text = BeautifulSoup(text, 'html.parser')

        #<a data-tn-element="companyName">
        company_names = text.find_all('span', {'class': 'company'})
        print(co_name)

        for spans in company_names:
            tag = spans.find('a')
            if tag == None:
                next
            else:
                name = tag.get_text()
                if None != re.match('^'+co_name[0:3], name.strip(), flags = re.I):
                    if tag['href'][1:4] == 'cmp':
                        company_list.loc[i,'indeed_url'] = tag['href']
                        print('did it: ', company_list['indeed_url'][i] )
                        break
                    else: pass

        time.sleep(random.random())
    else: 
        pass
    
company_list



null_list = company_list.loc[company_list.loc[:,'indeed_url'].isnull()]
len(null_list)

null_list.loc[:,'indeed_url'] = null_list.loc[:,'Company Name'].apply(cmp_name)
null_list = null_list.loc[:,['Company Name','indeed_url']]
null_list.loc[:,'indeed_url'] = ['/cmp/' + x for x in null_list.loc[:,'indeed_url']]
null_list


test = pd.merge(test, null_list, how='left', on= 'Company Name', left_on=None, right_on=None, sort=True,
         suffixes=('', '_y'), copy=True, indicator=False)
test.loc[test.loc[:,'indeed_url'].isnull(),'indeed_url'] = test.loc[test.loc[:,'indeed_url'].isnull(), 'indeed_url_y']
test = test.iloc[:,[0,1]]
company_list = test
company_list.isnull().any()

company_list.to_csv('../Data/fort500_cmp_url.csv')

