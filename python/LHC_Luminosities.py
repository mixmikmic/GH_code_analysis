get_ipython().magic('matplotlib notebook')
import matplotlib.pyplot as plt
import time
import pytimber

db = pytimber.LoggingDB()

now=time.time()
now_minus_a_day = now - 3600*24
alice='ALICE:LUMI_TOT_INST'
atlas='ATLAS:LUMI_TOT_INST'
cms='CMS:LUMI_TOT_INST'
lhcb='LHCB:LUMI_TOT_INST'
data=db.get([alice,atlas,cms,lhcb],now_minus_a_day,now)

#Create figure
plt.figure(figsize=(12,6))

#Plot Alice
tt,vv=data[alice]
plt.plot(tt,1000*vv,'-g',label=r'Alice $\times$ 1000')

#Plot Atlas
tt,vv=data[atlas]
plt.plot(tt,vv,'-b',label='Atlas')

#Plot CMS
tt,vv=data[cms]
plt.plot(tt,vv,'-r',label='CMS')

#Plot LHCb
tt,vv=data[lhcb]
plt.plot(tt,10*vv,'-k',label=r'LHCb $\times$ 10')

#Set axis and legend
plt.ylabel(r'Luminosity [$10^{30} \rm cm^{-2}  s^{-1}$]')
plt.legend()
plt.title(time.asctime(time.localtime(now)))
pytimber.set_xaxis_date()

