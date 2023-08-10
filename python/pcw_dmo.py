import hypy as hp
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

t,s = hp.ldf('pcw_ds1.txt')

hp.diagnostic(t,s)

Q=0.007997685185185;        # Pumping rate in m3/s
rw=0.1078;                  # Radius of well screen in m
rc=2.4;                     # Radius of the casing in m
d = [Q,rw,rc]

p0 = hp.pcw.gss(t,s)

hp.trial(p0,t,s,'pcw')

p1 = hp.fit(p0,t,s,'pcw')

hp.pcw.rpt(p1,t,s,d,'pcw',ttle='Rusthon example - automatic fit')



