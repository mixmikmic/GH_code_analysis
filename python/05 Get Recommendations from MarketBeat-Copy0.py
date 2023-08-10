# import urllib2 as ul
import requests
import csv
# from lxml import html
from BeautifulSoup import BeautifulSoup as bs
import scipy.io as sio
from numpy import shape

MARKETBEAT_NASDAQ_URL = 'http://www.marketbeat.com/stocks/NASDAQ/{0}'
MARKETBEAT_NASDAQ_MOST_RECENT_URL = 'http://www.marketbeat.com/stocks/NASDAQ/{0}/?MostRecent=1'

def get_ranking_csvrows_from_url(url,headers=None):
    try:
        page = requests.get(url)
    except Exception as inst:
        raise inst
    soup = bs(page.text)

    table = soup.find(lambda tag: tag.name == 'table' and
                                  tag.has_key('id') and
                                  tag['id'] == 'ratingstable' and
                                  tag.has_key('class') and 
                                  tag['class'] == "tablesorter")
    if table == None:
        raise Exception('No table in page')
        
    cur_headers = table.findAll('th') #,{'class':'header'})
    if cur_headers == None: 
        raise Exception('Headers not found')

    cur_headers = [ h.text.strip() for h in cur_headers ]
    cur_headers.insert(0,u'Ticker')

    if headers != None and headers != cur_headers:
        raise Exception('Wrong headers found', str(cur_headers))
    
    #csvout.writerow([u'Ticker'] + [ h.text for h in headers]) # concat 'Ticker' as the first column
        
#     %debug
    data = table.findAll('td') #,{'class':'yfnc_tabledata1'})
#     print shape(data)
    
    it = iter([ d.text.strip() for d in data ]) # create an iterator over the textual data
    csvrows = zip([tick]*(1+len(data)/5),it,it,it,it,it,it) # each call to it returns the next data entry, so this zip will create a 6-tuple array
#     print shape(csvrows)

    # dirty trick (only return 2 parameters if we don't know what we're looking for):
    if headers == None:
        return csvrows, cur_headers
    else:
        return csvrows

tick = 'INTC'
csvrows,headers = get_ranking_csvrows_from_url(MARKETBEAT_NASDAQ_URL.format(tick))

headers

MARKETBEAT_NASDAQ_URL.format(tick)

# THIS SHOULD THROW AN EXCEPTION
tick = 'AAME'
try:
    csvrows = get_ranking_csvrows_from_url(MARKETBEAT_NASDAQ_URL.format(tick),headers)
except Exception as e:
    print 'SUCCESS'
    print e
else:
    print 'FAIL'

print "\t".join(headers)
print "\t".join(csvrows[0])

print shape(csvrows)
print shape(headers)
print "\t".join(csvrows[0])
print "\t".join(csvrows[1])

# the last cell is unneeded html code, we'll get rid of it later

with open('NASDAQ.txt','r') as f:
    lines = f.readlines()

tickers = [l.strip() for l in lines if l.strip() != '']
# tickers

out = open('data/marketbeat_nasdaq.csv','a')
csvout = csv.writer(out)

headers = None
for i,tick in enumerate(tickers[1:]):
    tick = tick.split('\t')[0]
    print i,tick,',',
    try:
        if headers == None:
            csvrows,headers = get_ranking_csvrows_from_url(MARKETBEAT_NASDAQ_URL.format(tick))
        else:
            csvrows = get_ranking_csvrows_from_url(MARKETBEAT_NASDAQ_URL.format(tick),headers)
        csvrows = [ r[:-1] for r in csvrows ] # get rid of the last element in every tuple
    except Exception as inst:
        print inst
        continue

    csvout.writerows(csvrows)

out.flush()
out.close()
# csvout.close()        
# close(out)

with open('stocks.csv','r') as infile:
    csvin = csv.reader(infile)
    data = map(tuple,csvin)

data[len(data)-1]

sio.savemat('stocks.mat',{'data':data},do_compression=True)

