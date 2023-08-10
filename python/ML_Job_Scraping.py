# Import the required libraries
import bs4
import urllib2
import pandas as pd
import math
import time
from pandas import DataFrame, Series
import matplotlib
get_ipython().magic('matplotlib inline')

# Base URL 
base_url = 'http://www.naukri.com/machine-learning-jobs-'
source = urllib2.urlopen(base_url).read()

soup = bs4.BeautifulSoup(source, "lxml")

all_links = [link.get('href') for link in soup.findAll('a') if 'job-listings' in  str(link.get('href'))]
print "Sample job description link:",all_links[0]

jd_url = all_links[2]
jd_source = urllib2.urlopen(jd_url).read()
jd_soup = bs4.BeautifulSoup(jd_source,"lxml")

# Job Location
location = jd_soup.find("div",{"class":"loc"}).getText().strip()
print location

# Job Description
jd_text = jd_soup.find("ul",{"itemprop":"description"}).getText().strip()
print jd_text

# Experience Level
experience = jd_soup.find("span",{"itemprop":"experienceRequirements"}).getText().strip()
print experience

# Role Level Information
labels = ['Salary', 'Industry', 'Functional Area', 'Role Category', 'Design Role']
role_info = [content.getText().split(':')[-1].strip() for content in jd_soup.find("div",{"class":"jDisc mt20"}).contents 
 if len(str(content).replace(' ',''))!=0]

role_info_dict = {label: role_info for label, role_info in zip(labels, role_info)}
print role_info_dict

# Skills required
key_skills = '|'.join(jd_soup.find("div",{"class":"ksTags"}).getText().split('  '))[1:]
print key_skills

# Education Level
edu_info = [content.getText().split(':') for content in jd_soup.find("div",{"itemprop":"educationRequirements"}).contents 
 if len(str(content).replace(' ',''))!=0]

edu_info_dict = {label.strip(): edu_info.strip() for label, edu_info in edu_info}

# Sometimes the education information for one of the degrees can be missing
edu_labels = ['UG', 'PG', 'Doctorate']
for l in edu_labels:
    if l not in edu_info_dict.keys():
        edu_info_dict[l] = ''
print edu_info_dict

# Company Info
company_name = jd_soup.find("div",{"itemprop":"hiringOrganization"}).contents[1].p.getText()
print company_name

naukri_df = pd.DataFrame()
column_names = ['Location', 'Link', 'Job Description', 'Experience','Salary', 'Industry', 'Functional Area', 'Role Category', 
                'Design Role', 'Skills','Company Name', 
                'UG','PG','Doctorate']

from collections import OrderedDict
df_dict = OrderedDict({'Location':location, 'Link':all_links[0],'Job Description':jd_text,'Experience':experience,
                       'Skills':key_skills,'Company Name':company_name})
df_dict.update(role_info_dict)
df_dict.update(edu_info_dict)
df_dict

naukri_df = naukri_df.append(df_dict,ignore_index=True)
naukri_df

# Reordering the columns to a preferred order as specified
naukri_df = naukri_df.reindex(columns=column_names)
naukri_df





print soup.find("div", { "class" : "count" }).h1.contents[1].getText()

num_jobs = int(soup.find("div", { "class" : "count" }).h1.contents[1].getText().split(' ')[-1])
print num_jobs

num_pages = int(math.ceil(num_jobs/50.0))
print "URL of the last page to be scraped:", base_url + str(num_pages)

# Together into one function
import bs4
import urllib2
import pandas as pd
import math
import time
from pandas import DataFrame
from collections import OrderedDict
import cPickle

# Base URL 
base_url = 'http://www.naukri.com/machine-learning-jobs-'
source = urllib2.urlopen(base_url).read()

soup = bs4.BeautifulSoup(source, "lxml")
num_jobs = int(soup.find("div", { "class" : "count" }).h1.contents[1].getText().split(' ')[-1])
num_pages = int(math.ceil(num_jobs/50.0))

# Together into one function
labels = ['Salary', 'Industry', 'Functional Area', 'Role Category', 'Design Role']
edu_labels = ['UG', 'PG', 'Doctorate']
naukri_df = pd.DataFrame()
           
for page in range(1,num_pages+1):
    page_url = base_url+str(page)
    source = urllib2.urlopen(page_url).read()
    soup = bs4.BeautifulSoup(source,"lxml")
    all_links = [link.get('href') for link in soup.findAll('a') if 'job-listings' in  str(link.get('href'))]
    for url in all_links:
        jd_source = urllib2.urlopen(url).read()
        jd_soup = bs4.BeautifulSoup(jd_source,"lxml")
        try:
            jd_text = jd_soup.find("ul",{"itemprop":"description"}).getText().strip()
            location = jd_soup.find("div",{"class":"loc"}).getText().strip()
            experience = jd_soup.find("span",{"itemprop":"experienceRequirements"}).getText().strip()
            
            role_info = [content.getText().split(':')[-1].strip() for content in jd_soup.find("div",{"class":"jDisc mt20"}).contents if len(str(content).replace(' ',''))!=0]
            role_info_dict = {label: role_info for label, role_info in zip(labels, role_info)}
            
            key_skills = '|'.join(jd_soup.find("div",{"class":"ksTags"}).getText().split('  '))[1:]

            edu_info = [content.getText().split(':') for content in jd_soup.find("div",{"itemprop":"educationRequirements"}).contents if len(str(content).replace(' ',''))!=0]
            edu_info_dict = {label.strip(): edu_info.strip() for label, edu_info in edu_info}
            for l in edu_labels:
                if l not in edu_info_dict.keys():
                    edu_info_dict[l] = ''

            company_name = jd_soup.find("div",{"itemprop":"hiringOrganization"}).contents[1].p.getText().strip()
        
        except AttributeError:
            continue
        df_dict = OrderedDict({'Location':location, 'Link':url,'Job Description':jd_text,'Experience':experience,'Skills':key_skills,'Company Name':company_name})
        df_dict.update(role_info_dict)
        df_dict.update(edu_info_dict)
        naukri_df = naukri_df.append(df_dict,ignore_index=True)
        time.sleep(1)
    print page
    

import cPickle
column_names = ['Location', 'Link', 'Job Description', 'Experience','Salary', 'Industry', 'Functional Area', 'Role Category', 
                'Design Role', 'Skills', 'Company Name',
                'UG','PG','Doctorate']

naukri_df = naukri_df.reindex(columns=column_names)        
with open('naukri_dataframe.pkl', 'wb') as f:
    cPickle.dump(naukri_df, f)            

with open('naukri_dataframe.pkl', 'r') as f:
    naukri_df = cPickle.load(f) 

naukri_df.shape

naukri_df.head()

s

