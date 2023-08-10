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
from selenium import webdriver
import time
import glob
import os

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

# to download a copy 
# https://stackoverflow.com/questions/24844729/download-pdf-using-urllib
    

# url of pdf like https://arxiv.org/pdf/1709.00042.pdf
# outputs a list of emails
def getEmailList(code):

    with redirect_output("txt/" + code + ".txt"):
        get_ipython().run_line_magic('run', 'pdf2txt.py pdf/temp.pdf')
    
    with open("txt/" + code + ".txt", 'r') as myfile:
        data=myfile.read()
        emails = re.findall(r'[\w\.-]+@[\w\.-]+', data )
        if emails:
            return emails
        else:
            return []

def download_pdf(lnk, name, download_folder, path_to_chrome_driver):

    options = webdriver.ChromeOptions()
    profile = {
               "plugins.plugins_list": [{"enabled": False,
                                         "name": "Chrome PDF Viewer"}],
               "download.default_directory": download_folder,
               "download.extensions_to_open": ""
                }

    options.add_experimental_option("prefs", profile)
    print("Downloading file from link: {}".format(lnk))
    driver = webdriver.Chrome(path_to_chrome_driver,chrome_options = options)
    driver.get(lnk)

    filename = lnk.split("/")[4].split(".cfm")[0]
    print("File: {}".format(filename))
    print("Status: Download Complete.")
    print("Folder: {}".format(download_folder))
    time.sleep(10)
    
     # get that pdf
    list_of_files = glob.glob('pdf/*')
    latest_file = max(list_of_files, key=os.path.getctime)
    latest_full_path = '/home/ahmed/Dropbox/DFCI/08_radiomics.io/science/ieee/' + latest_file 
    # change its name
    os.rename( latest_full_path, 
              '/home/ahmed/Dropbox/DFCI/08_radiomics.io/science/ieee/pdf/' + name.replace("/","-") + '.pdf')

    driver.close()

df = pd.DataFrame.from_csv('ieee.csv', index_col=0)

print df.shape
df.head()

#  save as much pdfs as possible
for i in range(505,df.shape[0]):
    #
    url =  'http://sci-hub.bz/' + df.iloc[i]['DOI']
    print url
    # downloaf pdf in pdfs folder
    download_pdf(url, df.iloc[i]['DOI'] , 'pdf', '/home/ahmed/Dropbox/DFCI/08_radiomics.io/chromedriver')

    print i

# now most odfs have beenn saved
# now build csv and convert pdf to txt, to get email

def getEmail(pdf):
    file_path =  'pdf/' + pdf
    txt_file = "txt/" + pdf + ".txt"
        
    try:
        with redirect_output(txt_file):
            get_ipython().run_line_magic('run', 'pdf2txt.py $file_path')

        with open(txt_file, 'r') as myfile:
            data=myfile.read()
            emails = re.findall(r'[\w\.-]+@[\w\.-]+', data )
            if emails:
                return emails
            else:
                return []
    except:
        return []

# search result page
allTitles = []
allAuthors = []
allFullURLs = []
# abstract page
allEmails = [] 
allYears = []
allAbstracts = []
allComments = [] # combine with year into shortDetails

#added at pandas level
# source = ieee
# key = uuid
# use = all 1


for i in range(df.shape[0]): # 
    #
    allTitles.append(df.iloc[i]['Document Title'])
    # 
    allAuthors.append(df.iloc[i]['Authors'].replace(";",", "))
    #
    allFullURLs.append(df.iloc[i]['PDF Link'])
    # emails
    # check if pdf exists:
    fname =  df.iloc[i]['DOI'].replace("/","-") + '.pdf'
    print fname
    if os.path.isfile('pdf/' + fname):
        allEmails.append(  getEmail ( fname ) )
    else:
        allEmails.append([])
    #
    allYears.append(int(df.iloc[i]['Year']))
    # 
    allAbstracts.append(df.iloc[i]['Abstract'])
    #
    allComments.append(df.iloc[i]['Publication Title'])
    print i
    

lista = [allTitles,allAuthors,allFullURLs,allComments,allYears,allAbstracts,allEmails]
for lis in iter(lista):
    print len(lis)

df = pd.DataFrame()
# those that match pubmed
df['Title'] = allTitles
df['Description'] = allAuthors
df['ShortDetails'] = [allComments[x] + ". " + str(allYears[x]) for x in range(len(allYears))  ]
df['abstract'] =   allAbstracts
df['email'] =   allEmails
df['fullURL'] = allFullURLs                                                                           
df['source'] = ['ieee' for x in range(len(allYears))]                                                                            
df['year'] = allYears
df['key'] = [str(uuid.uuid4()) for x in range(len(allYears))]
df['use'] = [1 for x in range(len(allYears))]
df['doi'] = df1['DOI']
    
    
df.head(n=5)

df.to_json( 'ieee_final.json' ,"records")
# then need to add
# var Data1 = ....
# export default Data1

### load json 
df = pd.read_json('/home/ahmed/Dropbox/DFCI/14_zoo/scrap-science/ieee/out/ieee_final_beautiful.json')
df.to_csv("ieee.csv")
print df.shape
df.head()

# not used
# get email func from pubmed
# here it is different as get downloaded filename is not working
def download_pdf(lnk, name, download_folder, path_to_chrome_driver):

    options = webdriver.ChromeOptions()
    profile = {
               "plugins.plugins_list": [{"enabled": False,
                                         "name": "Chrome PDF Viewer"}],
               "download.default_directory": download_folder,
               "download.extensions_to_open": ""
                }

    options.add_experimental_option("prefs", profile)
    print("Downloading file from link: {}".format(lnk))
    driver = webdriver.Chrome(path_to_chrome_driver,chrome_options = options)
    
    full_path = '/home/ahmed/Dropbox/DFCI/14_zoo/scrap-science/ieee/pdf1/'
    # before
    before = os.listdir(full_path)
    # download
    driver.get(lnk)
    # after
    after = os.listdir(full_path)  
    # look for change
    change = set(after) - set(before)
    
    # if no change
    while len(change) == 0:
        time.sleep(1)
        
    # if change but part
    while len(change) == 1 and '.part' in max( glob.glob('pdf1/*') , key=os.path.getctime):
        time.sleep(1) 
        
    # read file
    list_of_files = glob.glob('pdf1/*')
    latest_file = max(list_of_files, key=os.path.getctime)
    print "latest_file" , latest_file
    latest_full_path = '/home/ahmed/Dropbox/DFCI/14_zoo/scrap-science/ieee/' + latest_file 
    print latest_full_path
    # change its name
    new_name = '/home/ahmed/Dropbox/DFCI/14_zoo/scrap-science/ieee/pdf1/' + name.replace("/","-") + '.pdf'
    os.rename( latest_full_path, new_name )

    driver.close()
    
    try:
        with redirect_output("txt1/" + name.replace("/","-") + ".txt"):
            get_ipython().run_line_magic('run', 'pdf2txt.py $new_name')

        with open("txt1/" + name.replace("/","-") + ".txt", 'r') as myfile:
            data=myfile.read()
            emails = re.findall(r'[\w\.-]+@[\w\.-]+', data )
            if emails:
                return emails
            else:
                return []
    except:
        pass
            

def download_pdf(lnk, name, download_folder, path_to_chrome_driver):

    options = webdriver.ChromeOptions()
    profile = {
               "plugins.plugins_list": [{"enabled": False,
                                         "name": "Chrome PDF Viewer"}],
               "download.default_directory": download_folder,
               "download.extensions_to_open": ""
                }

    options.add_experimental_option("prefs", profile)
    print("Downloading file from link: {}".format(lnk))
    driver = webdriver.Chrome(path_to_chrome_driver,chrome_options = options)
    driver.get(lnk)

    filename = lnk.split("/")[4].split(".cfm")[0]
    print("File: {}".format(filename))
    print("Status: Download Complete.")
    print("Folder: {}".format(download_folder))
    time.sleep(15)
    
     # get that pdf
    list_of_files = glob.glob('pdf1/*')
    latest_file = max(list_of_files, key=os.path.getctime)
    latest_full_path = '/home/ahmed/Dropbox/DFCI/14_zoo/scrap-science/ieee/' + latest_file 
    # change its name
    os.rename( latest_full_path, 
              '/home/ahmed/Dropbox/DFCI/14_zoo/scrap-science/ieee/pdf1/' + name.replace("/","-") + '.pdf')

    driver.close()

# loop through, if email no there, get it.
# this added 99 pdf to pdf folder 417-->516
# downloaf pdf in pdfs folder
              
for i in range(497,df.shape[0]):# 
    if df.iloc[i].email == []:
        url =  'http://sci-hub.bz/' + df.iloc[i]['doi']
        print url , i
        # downloaf pdf in pdfs folder
        download_pdf(url, df.iloc[i]['doi'] , 'pdf1', '/home/ahmed/Dropbox/DFCI/08_radiomics.io/chromedriver')
    
#         url =  'http://sci-hub.bz/' + df.iloc[i]['doi']
#         print url, i
#         email = download_pdf(url,df.iloc[i]['doi'] , 'pdf1', '/home/ahmed/Dropbox/DFCI/08_radiomics.io/chromedriver')
#         df = df.set_value(i, "email", email )

# fill in emails
# before : 330 without emails, txt folder has 481 (some empty)
# after : 274 without emails, txt folder has 516 (some empty)
for i in range(df.shape[0]):# 
    if df.iloc[i].email == []:
        # emails
        # check if pdf exists:
        fname =  df.iloc[i]['doi'].replace("/","-") + '.pdf'
        print fname
        if os.path.isfile('pdf/' + fname):
            df.set_value(i, "email", getEmail ( fname ) )

df.to_csv('ieee_result.csv')

counter = 0
for i in range(df.shape[0]):# 
    if df.iloc[i].email == []:
        counter+=1
print counter

df.shape[0]-274



