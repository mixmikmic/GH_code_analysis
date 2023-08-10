get_ipython().magic('matplotlib notebook')
import matplotlib.pyplot as pl 
import pytimber

db=pytimber.LoggingDB()

db.search('%SPS%BCT%TOTAL_INTEN%')

data=db.get('SPS.BCTDC.51895:TOTAL_INTENSITY','2016-08-02 12:00:00','2016-08-02 12:01:00')

pl.clf()
timestamps,intensities=data['SPS.BCTDC.51895:TOTAL_INTENSITY']
for ts,d in zip(timestamps,intensities):
  pl.plot(d,label=pytimber.dumpdate(ts,fmt='%H:%M:%S'))

pl.title(pytimber.dumpdate(ts,fmt='%Y-%m-%d'))
pl.legend()



