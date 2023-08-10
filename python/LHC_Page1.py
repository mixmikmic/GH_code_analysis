get_ipython().magic('matplotlib notebook')
import matplotlib.pyplot as plt
import time
import pytimber

db = pytimber.LoggingDB()

now=time.time()
now_minus_a_day = now - 3600*24
ib1="LHC.BCTDC.A6R4.B1:BEAM_INTENSITY"
ib2="LHC.BCTDC.A6R4.B2:BEAM_INTENSITY"
nrg="LHC.BOFSU:OFSU_ENERGY"
data=db.get([ib1,ib2,nrg],now_minus_a_day,now)

plt.figure(figsize=(12,6))

tt,vv=data[ib1]
plt.plot(tt,vv,'-b',label='Beam1')
tt,vv=data[ib2]
plt.plot(tt,vv,'-r',label='Beam2')
plt.ylabel('Protons')
plt.twinx()
tt,vv=data[nrg]
plt.plot(tt,vv,'-g',label='Energy')
plt.ylabel('Energy [GeV]')
plt.title(time.asctime(time.localtime(now)))
pytimber.set_xaxis_date()

print "Experiments' instantaneous luminosity variable names"
db.search("%LUMI_INST")

print "Exploration of the variables' tree"
db.tree.LHC.Beam_Instrumentation.Beam_Position.DOROS_BPMs.IP1.LHC_BPM_1L1_B1_DOROS_ACQUISITION_STATUS

