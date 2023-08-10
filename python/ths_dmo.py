import hypy as hp
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

t,s = hp.ldf('ths_ds1.txt')

hp.plot(t,s)

plt.plot(t,s,'o')
plt.xlabel('Time in seconds')
plt.ylabel('Drawdown in meters')

hp.diagnostic(t,s)

import ths

p0 = ths.gss(t,s)

hp.trial(p0,t,s, name = 'ths')

p0 = [1.1,100]
hp.trial(p0,t,s,'ths')

p0 = [1.8,100]
hp.trial(p0,t,s,'ths')

p0 = [1.8,500]
hp.trial(p0,t,s,'ths')

p = hp.fit(p0,t,s,'ths')
hp.trial(p,t,s,'ths')

q = 1.3888e-2 #pumping rate in m3/s
r = 250 #radial distance in m
d = [q,r]
hp.ths.rpt(p,t,s,d,'ths',Author='Your name',ttle = 'Interference test', Rapport = 'My rapport', filetype = 'pdf')

t,s = hp.ldf('/home/pianarol/AnacondaProjects/ths_ds1.txt')
p0 = hp.ths.gss(t,s)
p = hp.fit(p0,t,s, 'ths')
hp.ths.rpt(p,t,s,d,'ths','Theis interpretation of the Fetter data', Author='name', Rapport='rapport', filetype = 'pdf')



