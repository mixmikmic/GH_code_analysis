import sys
sys.path.append('../src/')

from ALMAQueryCal import *

q = queryCal()

fileCal = "callist_20170329.list"
listCal = q.readCal(fileCal, fluxrange=[0.5, 10000000])

print "Number of selected sources: ", len(listCal)

listCal

data    = q.queryAlma(listCal, public = True, savedb=True, dbname='calibrators_gt_0.5Jy.db')

report = q.selectDeepfield_fromsql("calibrators_gt_0.5Jy.db", maxFreqRes=10000000, array='12m',         excludeCycle0=True, selectPol=False, minTimeBand={3:3600., 6:3600., 7:3600.}, verbose=True, silent=True)

q.writeReport(report, "report.txt", silent=True)



