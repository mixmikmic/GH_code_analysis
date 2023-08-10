import hypy as hp
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

t,s = hp.ldf('blt_ds1.txt')
hp.diagnostic(t,s)

p0 = hp.blt.gss(t,s)

p = hp.fit(p0,t,s,'blt')

hp.trial(p,t,s,'blt')

q = 0.030 #Pumping rate in m3/s
r = 20
d = [q,r]

hp.blt.rpt(p,t,s,d,'blt',ttle = 'de Marsilly example')



