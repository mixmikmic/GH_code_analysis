from IPython.core.display import Image

Image('51job.com/51job.com.png')

import requests

# search for frontend developing positions in Beijing and Hong Kong
url = 'https://search.51job.com/list/330000%252C010000,000000,0000,32,9,99,%25E5%2589%258D%25E7%25AB%25AF%25E5%25BC%2580%25E5%258F%2591,2,1.html?lang=c&stype=&postchannel=0000&workyear=99&cotype=99&degreefrom=99&jobterm=99&companysize=99&providesalary=99&lonlat=0%2C0&radius=-1&ord_field=0&confirmdate=9&fromType=&dibiaoid=0&address=&line=&specialarea=00&from=&welfare='

r = requests.get(url)

from pyquery import PyQuery

pq = PyQuery(r.text)

# Search for the company names
pq('span.t2 a').text()

r.encoding = 'gbk'

pq = PyQuery(r.text)
pq('span.t2 a').text()

import datetime
print('Last updated:', datetime.datetime.now())



