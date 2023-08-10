import pytimber
import time

db = pytimber.LoggingDB()

db.search("%BEAM_INTENSITY%")

db.tree.LHC.Beam_Quality.Beam_1.get_vars()

db.get("HX:FILLN",time.time())

now=time.time()
"The unixtime `%.3f` correspond to `%s` local time."%(now,pytimber.dumpdate(now))

db.get("HX:FILLN",time.time())

db.get("HX:FILLN",'2016-08-03 16:30:00.000')

db.get("HX:FILLN",'2016-08-03 16:30:00.000',unixtime=False)

db.get("HX:FILLN",'2016-08-02 16:30:00.000','next')

db.get("HX:FILLN",'2016-08-02 16:30:00.000','2016-08-03 16:30:00.000')

db.get("LHC.BCTDC.A6R4.B1:BEAM_INTENSITY",now)

db.get("LHC.BCTDC.A6R4.B%:BEAM_INTENSITY",now)

db.get(["LHC.BCTDC.A6R4.B1:BEAM_INTENSITY","LHC.BCTDC.A6R4.B2:BEAM_INTENSITY"],now)

# prepare for plotting
get_ipython().magic('matplotlib notebook')
import matplotlib.pyplot as pl

ts=pytimber.parsedate("2016-07-01 03:10:15.000")
ib1="LHC.BCTFR.A6R4.B1:BUNCH_INTENSITY"
ib2="LHC.BCTFR.A6R4.B2:BUNCH_INTENSITY"
data=db.get([ib1,ib2],ts,'next')
timestamps,valuesb1=data[ib1]
timestamps,valuesb2=data[ib2]
pl.figure()
pl.plot(valuesb1[0])
pl.plot(valuesb2[0])
pl.xlabel("slot number")
pl.ylabel("protons per bunch")

ts=pytimber.parsedate("2016-07-01 03:10:15.000")
ib1="LHC.BCTFR.A6R4.B1:BUNCH_INTENSITY"
ib2="LHC.BCTFR.A6R4.B2:BUNCH_INTENSITY"
data=db.get([ib1,ib2],ts,ts+60)
timestamps,valuesb1=data[ib1]
timestamps,valuesb2=data[ib2]

pl.figure()
pl.imshow(valuesb1,aspect='auto',origin='bottom')
pl.ylabel('seconds'); pl.xlabel("slot number")
pl.colorbar(); pl.clim(9e10,12e10)

pl.figure()
pl.imshow(valuesb2,aspect='auto',origin='bottom')
pl.ylabel('seconds'); pl.xlabel("slot number")
pl.colorbar(); pl.clim(9e10,12e10)

