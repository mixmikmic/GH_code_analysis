import requests
import lxml.html as lh

url = 'http://names.mongabay.com/data/1000.html'
page = requests.get(url)
doc = lh.fromstring(page.content)
print page.content[:500]

# the tag tr (table row) is used in many places, 
# among them the table of interest to us.
# we can identify those rows by the fact that 
# the table contains 11 columns.
tr_elements = doc.xpath('//tr')
for T in tr_elements[:20]:
    for e in T.iterchildren():
        print e.text_content(),
    print

type(tr_elements[0])

element=T[7]
# element.  # uncomment this line and hit <tab> after the dot to see the methods and properties of an HTML element

for i in range(len(tr_elements)):
    if len(tr_elements[i])==11:
        print i, tr_elements[i].text_content()
        break

col=[]  # collect column names into col
T=tr_elements[0]
print type(T)
i=0
print len(T)
for t in T.iterchildren():
    i+=1
    name=t.text_content()
    print '%d:"%s"'%(i,name)
    col.append((name,[]))

print 'the columns are:',col

for j in range(7,len(tr_elements)):
    T=tr_elements[j]
    if len(T)!=11:
        break
    i=0
    for t in T.iterchildren():
        data=t.text_content()
        if i>0:
            try:
                data=float(data)
            except:
                print data,'cannot be converted to float, row,col=',j,i
                data=None
        col[i][1].append(data)
        i+=1

[len(C) for (title,C) in col]

min_len=min([len(C) for (title,C) in col])
min_len

#To determine what was the orignal order for renaming
[n for n,l in col]

Dict={title:column for (title,column) in col}
import pandas as pd
df=pd.DataFrame(Dict)
df = df[[n for n,l in col]]
df.head()

#!pip install pattern

import pattern.web
num=5;
url = 'http://rss.nytimes.com/services/xml/rss/nyt/World.xml'
results = pattern.web.Newsfeed().search(url, count=num)

print 'The current top headers from the NY times are:'
for i in range(num):
    print "%d\t%s"%(i,results[i].title)

print '\n\nURL: %s \n\n Header\n%s \n\nFull Article\n %s \n\n' % (results[0].url, results[0].title, results[0].description)

print '%s \n\n %s \n\n %s \n\n' % (results[0].url, results[0].title, pattern.web.plaintext(results[0].description))

import codecs

outputFile = codecs.open('tutorialOutput.txt', encoding='utf-8', mode='w')

def scrape(url):
    page = requests.get(url)
    doc = lh.fromstring(page.content)
    text = doc.xpath('//p[@itemprop="articleBody"]')
    finalText = str()
    for par in text:
        finalText += par.text_content()
    return finalText+'\n\n\n'

for result in results:
    outputText = scrape(result.url)
    outputFile.write(outputText)

outputFile.close()

get_ipython().system('head -4 tutorialOutput.txt')

url = 'http://164.100.47.132/LssNew/psearch/Result13.aspx?dbsl='

for i in xrange(5175,5973):
    newUrl = url + str(i)
    print 'Scraping: %s' % newUrl

from scrapy.contrib.spiders import CrawlSpider, Rule
from scrapy.contrib.linkextractors.sgml import SgmlLinkExtractor
from scrapy.selector import HtmlXPathSelector
from scrapy.item import Item
from BeautifulSoup import BeautifulSoup
import re
import codecs

class MySpider(CrawlSpider):
    name = 'statespider' #name is a name
    start_urls = ['http://www.state.gov/r/pa/prs/dpb/2010/index.htm',
    ] #defines the URL that the spider should start on. adjust the year.

        #defines the rules for the spider
    rules = (Rule(SgmlLinkExtractor(allow=('/2010/'), restrict_xpaths=('//*[@id="local-nav"]'),)), #allows only links within the navigation panel that have /year/ in them.

    Rule(SgmlLinkExtractor(restrict_xpaths=('//*[@id="dpb-calendar"]',), deny=('/video/')), callback='parse_item'), #follows links within the caldendar on the index page for the individuals years, while denying any links with /video/ in them

    )

    def parse_item(self, response):
        self.log('Hi, this is an item page! %s' % response.url) #prints the response.url out in the terminal to help with debugging
        
        #Insert code to scrape page content

        #opens the file defined above and writes 'texts' using utf-8
        with codecs.open(filename, 'w', encoding='utf-8') as output:
            output.write(texts)



