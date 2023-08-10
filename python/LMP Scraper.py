import datetime
import urllib

from urlparse import urlparse
from threading import Thread
import httplib, sys
from Queue import Queue

# Generate date range that matches date signature in URL of .csv files
endDate = datetime.datetime.strptime('20151118', '%Y%m%d')
startDate = datetime.datetime.strptime('20130101', '%Y%m%d')

dayDelta = (endDate-startDate).days
dateList = [ startDate + datetime.timedelta(days=n) for n in range(0, dayDelta) ]
dateList = [datetime.datetime.strftime(d, '%Y%m%d') for d in dateList]

# Generate URL list (for checking, will not be used directly)
urlList = ['https://www.misoenergy.org/Library/Repository/Market%20Reports/' + d + '_rt_lmp_final.csv' for d in dateList]

concurrent = 10

def doWork():
    while True:
        d = q.get()
        url = 'https://www.misoenergy.org/Library/Repository/Market%20Reports/' + d + '_rt_lmp_final.csv'
        csvFile = urllib.URLopener()
        csvFile.retrieve(url, '../miso/' + d + '_rt_lmp_final.csv')

get_ipython().run_cell_magic('time', '', '\nq = Queue(concurrent * 2)\n\nfor i in range(concurrent):\n    t = Thread(target=doWork)\n    t.daemon = True\n    t.start()\n    \ntry:\n    for date in dateList:\n        q.put(date)\n    q.join()\nexcept KeyboardInterrupt:\n    sys.exit(1)')



