import pandas as pd
import bs4 as bs
import urllib2
import re
import numpy as np
import uuid
import pandas as pd
from os import walk
import json
import sys
import time
reload(sys)
sys.setdefaultencoding('utf-8')

path = '/home/ahmed/Dropbox/DFCI/14_zoo/scrap-science/pubmed/raw_result'
dfs = []
for (_, _, filenames) in walk(path):
    for filename in filenames:
        fullPath = path+'/'+filename
        print fullPath
        dfs.append( pd.read_csv(fullPath, index_col=False,header=0) )
df = pd.concat(dfs)
# reset index
df = df.reset_index(drop=True)
print df.shape
# all titles to lower, strip, remove periods and remove double spaces
df['Title'] = df.Title.str.lower()
df['Title'] = df.Title.str.strip()
df['Title'] = df.Title.str.replace('.','')
df['Title'] = df.Title.str.replace('  ',' ')
# remove duplicates by title
df = df.drop_duplicates(subset='Title', keep="first")
# reset index
df = df.reset_index(drop=True)
print df.shape
df.head()

# add these columns
df['abstract']=""
df['email']=""
df['keywords']=""
df['fullURL']=""
df['source']=['pubmed' for x in range(df.shape[0])]
df['year']=np.zeros(df.shape[0],dtype=np.int)
df['key']=""

#
df['use']=np.zeros(df.shape[0],dtype=np.int)
#archive
print df.shape
df.head(n=4)

years = range(2014,3000)
yearsToInclude = [ str(x) for x in years]

for i in range(995 , df.shape[0] ):   
    
    # get url
    url = 'https://www.ncbi.nlm.nih.gov/' + df.iloc[i].URL
    try:
        # fetch url
        response = urllib2.urlopen(url)
        # convert to bs
        soup = bs.BeautifulSoup(response,"html")
        
        # 0 set fullURL
        df = df.set_value(i, "fullURL", url )

        # 1 get abstract
        abstract = soup.findAll("div",{"class":"abstr"})
        if len(abstract) == 1:
            df = df.set_value(i, "abstract", str(abstract[0]).replace('"',"'") )

        # 2 get email
        afflist = soup.findAll("div", { "class" : "afflist" })
        if len(afflist) == 1:
            email = re.search(r'[\w\.-]+@[\w\.-]+', str(afflist[0]) )
            if email:
                email = email.group(0)
                if email[-1] == '.':
                    email = email[:-1]
                df = df.set_value(i, "email", email)


        # 3 get keywords
        keywords = soup.findAll("div", { "class" : "keywords" })
        if len(keywords) == 1:
            df = df.set_value(i, "keywords", str(keywords[0].p.text).replace('"',"'") )
            
        print i
        
        # 4 set use
        if any(word in df.iloc[i].ShortDetails  for word in yearsToInclude):
            df = df.set_value(i, "use", 1)
            
        # 5 set year
        df = df.set_value(i, "year", int( df.iloc[i].ShortDetails[-4:] ))
        
        # 6 set key
        df = df.set_value(i, "key", str(uuid.uuid4()) )
            
        time.sleep(2)
        df.to_csv("pubmed_temp.csv")
        
    except urllib2.HTTPError:
        print str(i) + " no http"
        pass

   

# temp
df = pd.DataFrame.from_csv('/home/ahmed/Dropbox/DFCI/14_zoo/scrap-science/pubmed/pubmed_temp.csv')
print df.shape
df = df [df.use == 1]
print df.shape
df = df [df.year >= 2014]
print df.shape
df.to_csv("pubmed_dirty.csv")

df = pd.DataFrame.from_csv('/home/ahmed/Dropbox/DFCI/14_zoo/scrap-science/pubmed/out/pubmed_result.csv')
print df.shape
df.head()

counter = 0
for i in range(df.shape[0]):# 
    if pd.isnull(df.iloc[i].email):
        counter+=1
print counter



