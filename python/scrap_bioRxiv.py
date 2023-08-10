from IPython.utils import io
import sys
import pandas as pd
import bs4 as bs
import urllib2
import re
import numpy as np
import uuid
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.layout import LAParams
from pdfminer.image import ImageWriter
from pdfminer.converter import  TextConverter
from pdfminer.pdfpage import PDFPage
reload(sys)
sys.setdefaultencoding('utf-8')
import time
from selenium import webdriver
import glob
import os

url_list=[
    'http://www.biorxiv.org/search/deep%252Blearning%252Bmedical%252Bimaging?page=',
    'http://www.biorxiv.org/search/deep%252Blearning%252Bradiology?page=',
    'http://www.biorxiv.org/search/deep%252Blearning%252Bpathology?page=',
    'http://www.biorxiv.org/search/deep%252Blearning%252Bgenomics?page=',
    'http://www.biorxiv.org/search/convolutional%252Bneural%252Bnetwork?page=',
    'http://www.biorxiv.org/search/autoencoder?page=',
    'http://www.biorxiv.org/search/deep%252Blearning?page=',
    'http://www.biorxiv.org/search/cnn?page='
]

result_list =[
    319,
    52,
    238,
    730,
    342,
    68,
    1192,
    136
]

# go through above list and get all individual urls to papers
all_urls = []
for url_1,max_result in iter(zip(url_list,result_list)):
    page_number_list = np.arange(0,np.ceil(max_result/10.0))
    print page_number_list
    for i in page_number_list:
        url = url_1 + str(int(i))
        print url
        try:
            # fetch url
            response = urllib2.urlopen(url)
            # convert to bs
            soup = bs.BeautifulSoup(response,"html")
            urls = soup.findAll("a", { "class" : "highwire-cite-linked-title" })
            all_urls.extend ( ['http://www.biorxiv.org' + str(urls[x]['href']) for x in range(len(urls)) ] )
            time.sleep(2)
            np.save("biorxiv_temp_urls.npy", all_urls)
        except urllib2.HTTPError:
            print str(i) + " no http"
            pass

            
            

urls = np.load('/home/ahmed/Dropbox/DFCI/14_zoo/scrap-science/biorxiv/biorxiv_temp_urls.npy')
print urls.shape
urls = list(set(urls))
print len(urls)

urls[0]

# now go through all these urls and scrap

class redirect_output(object):
    """context manager for reditrecting stdout/err to files"""


    def __init__(self, stdout='', stderr=''):
        self.stdout = stdout
        self.stderr = stderr

    def __enter__(self):
        self.sys_stdout = sys.stdout
        self.sys_stderr = sys.stderr

        if self.stdout:
            sys.stdout = open(self.stdout, 'w')
        if self.stderr:
            if self.stderr == self.stdout:
                sys.stderr = sys.stdout
            else:
                sys.stderr = open(self.stderr, 'w')

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self.sys_stdout
        sys.stderr = self.sys_stderr

# search result page
allTitles = []
allAuthors = []
allFullURLs = []
# abstract page
allEmails = [] 
allYears = []
allUses = []
allAbstracts = []
allComments = [] # combine with year into shortDetails
# extra one
allPdfURLs = []
#added at pandas level
# source = arvix
# key = uuid

total = 0
print total
fileToSave = "temp_biorxiv_13.csv"
# loop through the main search result pages

for idx,url in enumerate(urls[total:]):

    try:
        print url, idx
        # fetch url
        response = urllib2.urlopen(url)
        # convert to bs
        soup = bs.BeautifulSoup(response,"html")

        # 1 get title
        titles = soup.findAll("h1", { "class" : "highwire-cite-title" })
        allTitles.append( titles[0].text )

        # 2 get authors
        authors = soup.findAll("span", { "class" : "highwire-citation-authors" })
        allAuthors.append( authors[0].text )

        # 3 get all fullURLs
        allFullURLs.append ( url )

        # 4 get emails through pdf
        pdfUrl = url + '.full.pdf'
        print pdfUrl
        allPdfURLs.append(pdfUrl)
        allEmails.append( download_pdf(pdfUrl, 
                                       url, 'pdf'
                                       , '/home/ahmed/Dropbox/DFCI/08_radiomics.io/chromedriver') )

        # 5 get year
        years = soup.findAll("li", { "class" : "published" })
        tempList1 = [ int( str(years[x].text).strip()
                          [str(years[x].text).strip().find('201')
                           :
                           (str(years[x].text).strip().find('201')+4)] )
                          for x in range(len(years)) ]
        allYears.append(tempList1[0])

        # 6 get use based on year
        if tempList1[0] >= 2014:
            allUses.append(1)
        else:
            allUses.append( 0 )  

        # 7 get abstracts
        abstracts = soup.find_all("p" , {'id' : 'p-2' })
        if len(abstracts) == 0:
            allAbstracts.append("")
        else:
            allAbstracts.append( str(abstracts[0].text).strip() )  

        # 8 get comments
        # used when javascript rpoduces the tag and bs4 cant find it
        driver = webdriver.Chrome('/home/ahmed/Dropbox/DFCI/08_radiomics.io/chromedriver')
        driver.get(url)
        siko = driver.find_element_by_class_name("pub_jnl")

        if 'article is a preprint and has not been peer-reviewed' in siko.text.strip():
            allComments.append("Retrieved from bioRxiv")
        else:
            allComments.append( str(siko.text.strip()) [:str(siko.text.strip()).find(' doi:')] )

        driver.close()


        ######
        df = pd.DataFrame()
        # those that match pubmed
        df['Title'] = allTitles
        df['Description'] = allAuthors
        df['ShortDetails'] = [allComments[x] + ". " + str(allYears[x]) for x in range(len(allYears))  ]
        df['abstract'] =   allAbstracts
        df['email'] =   allEmails
        df['fullURL'] = allFullURLs                                                                           
        df['source'] = ['biorxiv' for x in range(len(allYears))]                                                                            
        df['year'] = allYears
        df['key'] = [str(uuid.uuid4()) for x in range(len(allYears))]
        df['use'] = allUses
        # extra one
        df['pdfURL']= allPdfURLs

        df.to_csv( fileToSave)

    except urllib2.HTTPError:
        print str(i) + " no http"
        pass


    print "done"


df = pd.DataFrame()
# those that match pubmed
df['Title'] = allTitles
df['Description'] = allAuthors
df['ShortDetails'] = [allComments[x] + ". " + str(allYears[x]) for x in range(len(allYears))  ]
df['abstract'] =   allAbstracts
df['email'] =   allEmails
df['fullURL'] = allFullURLs                                                                           
df['source'] = ['biorxiv' for x in range(len(allYears))]                                                                            
df['year'] = allYears
df['key'] = [str(uuid.uuid4()) for x in range(len(allYears))]
df['use'] = allUses
# extra one
df['pdfURL']= allPdfURLs
    
    
df.head(n=10)

lista = [allTitles,allAuthors,allFullURLs,allComments,allYears,allAbstracts,allEmails,allUses,allPdfURLs]
for lis in iter(lista):
    print len(lis)

df.to_csv( "test.csv")
# then need to add
# var Data1 = ....
# export default Data1





