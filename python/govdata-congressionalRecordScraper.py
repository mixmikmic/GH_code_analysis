from bs4 import BeautifulSoup

#import requests
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities

dcap = dict(DesiredCapabilities.PHANTOMJS)
dcap["phantomjs.page.settings.userAgent"] =         ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.98 Safari/537.36")
dcap[u'acceptSslCerts'] = True
    
driver = webdriver.PhantomJS(desired_capabilities=dcap, 
                             service_args=['--ssl-protocol=any', '--ignore-ssl-errors=true'],
                             service_log_path="/Users/kyledunn/phantom.log")

driver.set_window_size(1440, 1024)

def loadComplete(driver):
    try:
        return 'true' == driver.execute_script('retFalse()')
    except WebDriverException:
        pass

import pandas as pd

import glob

def getHref(obj):
    try:
        url = obj['href']
        return url
    except:
        pass

def getLinkTable(year, month, day, hOrS, code=3):
    theUrl = """https://www.gpo.gov/fdsys/browse/collection.action?collectionCode=CREC&""" +              """browsePath={y}%2F{m:02d}%2F{m:02d}-{d:02d}%5C%2F{n}%2F{hors}&""" +              """isCollapsed=false&leafLevelBrowse=false&isDocumentResults=true"""
              
    theUrl = theUrl.format(y=year, m=month, d=day, hors=hOrS, n=code)
    
    #print theUrl
    
    #try:
    driver.get(theUrl)
    #except URLError as e:
    #    print year, month, day, hOrS, e.message
    #    return []

    pageString = driver.page_source
    
    #print pageString
    
    soup = BeautifulSoup(pageString, "html.parser")
    
    return soup.find('table', {'class': 'browse-node-table'})

# Scrape the CREC site for all <a> tags - i.e. links to the congressional record documents (HTML + PDF)
def getLinks(year, month, day, hOrS):
    # URLs change over time slightly
    code = 2
    expandedTable = getLinkTable(year, month, day, hOrS, code)
    
    while expandedTable is None and code < 6:
        code = code + 1
        expandedTable = getLinkTable(year, month, day, hOrS, code)
        
    if expandedTable is None:
        #print year, month, day, hOrS, "is invalid"
        return []
    
    aTags = [a for a in expandedTable.find_all('a') if a]

    allLinks = map(getHref, aTags)

    return allLinks
    

get_ipython().run_cell_magic('time', '', '\n# Took about 1.6 hours for 9 months on a 40Mbit connection\n\ndocs = { "HOUSE": [], "SENATE": [] }\nfor y in range(2016, 2017):\n    for m in range(4, 13):\n        for d in range(1,32):\n            for hOrS in ["HOUSE", "SENATE"]:\n                try:\n                    docs[hOrS] = docs[hOrS] + getLinks(y, m, d, hOrS)\n                except TypeError:\n                    pass')

#with open("/Users/kyledunn/Desktop/congressionalRecord/links-house-10-Dec-2016.txt", "w") as theFile:
#    theFile.write("\n".join(docs["HOUSE"]))

#with open("/Users/kyledunn/Desktop/congressionalRecord/links-senate-10-Dec-2016.txt", "w") as theFile:
#    theFile.write("\n".join(docs["SENATE"]))

housePlaintext = [d for d in docs["HOUSE"] if "htm" in d]
senatePlaintext = [d for d in docs["SENATE"] if "htm" in d]

#with open("/Users/kyledunn/Desktop/congressionalRecord/textlinks-house-6-April-2016.txt", "w") as theFile:
#    theFile.write("\n".join(housePlaintext))

#with open("/Users/kyledunn/Desktop/congressionalRecord/textlinks-senate-6-April-2016.txt", "w") as theFile:
#    theFile.write("\n".join(senatePlaintext))

#senatePlaintext = []
#with open("/Users/kyledunn/Desktop/congressionalRecord/textlinks-senate-6-April-2016.txt", "r") as theFile:
#    senatePlaintext = theFile.readlines()
    
#housePlaintext = []
#with open("/Users/kyledunn/Desktop/congressionalRecord/textlinks-house-6-April-2016.txt", "r") as theFile:
#    housePlaintext = theFile.readlines()

from multiprocessing.dummy import current_process
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool 

import os
from time import sleep

def downloadAndSave(url, hOrS):
    name = url.split("/")[-1].strip()
    filename = "/Users/kyledunn/Desktop/congressionalRecord/{0}/{1}".format(hOrS, name)
    
    # If the the file exists and isn't empty - move on
    #if os.path.isfile(filename) and os.path.getsize(filename) > 0:
    #    return
    
    r = requests.get(url)
    
    html = r.text

    try:
        with open(filename, "w") as theFile:
            theFile.write(html)
    except UnicodeEncodeError:
        print filename
        with open(filename, "w") as theFile:
            theFile.write(html.encode('utf-8'))
        
    
    sleep(0.2)
    return

get_ipython().run_cell_magic('time', '', '\nnumThreads = 8\npool = ThreadPool(numThreads)\n\n# 116 Senate sessions, 87 House took ~21 minutes\n\nresults = pool.map(lambda p: downloadAndSave(p, "HOUSE"), housePlaintext)\nresults = pool.map(lambda p: downloadAndSave(p, "SENATE"), senatePlaintext)\npool.close() \npool.join()')

senateSessions = set(["-".join(f.split("/")[-1].split("-")[1:4]) for f in senatePlaintext])
print len(senateSessions)
houseSessions = set(["-".join(f.split("/")[-1].split("-")[1:4]) for f in housePlaintext])
print len(houseSessions)

def parsePage(theText):
    
    soup = BeautifulSoup(theText, "html.parser")
    
    return "\n".join(soup.find('pre').text.splitlines()[4:])

import os

def mergeFiles(date, hOrS):
    mergedFile = '/Users/kyledunn/Desktop/congressionalRecord/{0}/Merged/{1}.txt'.format(hOrS, date)
    
    # If the the file exists and isn't empty - move on
    #if os.path.isfile(mergedFile) and os.path.getsize(mergedFile) > 0:
    #    return    
    
    path = '/Users/kyledunn/Desktop/congressionalRecord/{0}/*{1}*.htm'.format(hOrS, date)
    files = glob.glob(path)
     
    sessionText = ""
    # Remove file extension from sorting to get proper order
    for f in sorted(files, key = lambda x: x.rsplit('.', 1)[0]): 
        with open(f, "r") as theFile:
            try:
                sessionText = sessionText + parsePage(theFile.read())
            except AttributeError:
                print f, "failed to parse"
                
    
    try:
        with open(mergedFile, "w") as theFile:
            theFile.write(sessionText)
    except UnicodeEncodeError:
        with open(mergedFile, "w") as theFile:
            theFile.write(sessionText.encode('utf-8'))
    
    #print sessionText

get_ipython().run_cell_magic('time', '', '\nnumThreads = 8\npool = ThreadPool(numThreads)\n\n# Took about 2.5 minutes for ~200 sessions\n\nhResults = pool.map(lambda d: mergeFiles(d, "HOUSE"), houseSessions)\nsResults = pool.map(lambda d: mergeFiles(d, "SENATE"), senateSessions)\n\npool.close() \npool.join()')

senateSessions

redoH = [
    "CREC-2015-05-14-pt1-PgH2999-3.htm",
    "CREC-2007-12-17-pt2-PgH15741.htm",
    "CREC-2014-12-11-pt2-PgH9307-2.htm",
    "CREC-2000-03-23-pt1-PgH1330-2.htm",
    "CREC-2011-12-06-pt1-PgH8153.htm"
]

redoS = [
    "CREC-2001-05-09-pt1-PgS4773-2.htm",
    "CREC-2008-09-22-pt1-PgS9197-2.htm",
    "CREC-2001-07-24-pt1-PgS8121.htm",
    "CREC-2007-05-24-pt1-PgS6837.htm",
    "CREC-2008-02-29-pt1-PgS1393-7.htm",
    "CREC-1996-03-20-pt1-PgS2341-6.htm",
    "CREC-2002-02-26-pt1-PgS1141-2.htm" 
]

hRedoUrls = [u for u in housePlaintext if u.split("/")[-1][:-1] in redoH]
sRedoUrls = [u for u in senatePlaintext if u.split("/")[-1][:-1] in redoS]

#print len(hRedoUrls), len(sRedoUrls)

numThreads = 8

def makeBrowser(id):
    dcap = dict(DesiredCapabilities.PHANTOMJS)
    dcap["phantomjs.page.settings.userAgent"] =             ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10.9; rv:32.0) Gecko/20100101 Firefox/32.0")

    driver = webdriver.PhantomJS(desired_capabilities=dcap, 
                                 service_args=['--ignore-ssl-errors=true', '--ssl-protocol=tlsv1'],
                                 service_log_path="/Users/kyledunn/phantom-{0}.log".format(id))

    driver.set_window_size(1440, 1024)
    
    return driver

pool = ThreadPool(numThreads)

#browsers = [makeBrowser(i) for i in range(numThreads)]
threadIds = list(set(pool.map(lambda p : current_process().ident, range(4*numThreads))))
browserRefs = dict(zip(threadIds, browsers))

results = pool.map(lambda p: downloadAndSave(browserRefs, p, "HOUSE"), hRedoUrls)
results = pool.map(lambda p: downloadAndSave(browserRefs, p, "SENATE"), sRedoUrls)

pool.close() 
pool.join()

hRedoSessions = ["2011-12-06", "2000-03-23", "2014-12-11", "2015-05-14", "2007-12-17"]
sRedoSessions = ["20080229"]

numThreads = 1
pool = ThreadPool(numThreads)

#hResults = pool.map(lambda d: mergeFiles(d, "HOUSE"), hRedoSessions)
sResults = pool.map(lambda d: mergeFiles(d, "SENATE"), sRedoSessions)

pool.close() 
pool.join()



