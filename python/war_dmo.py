import hypy as hp
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

t,s = hp.ldf('war_ds1.txt')

hp.diagnostic(t,s)

e=400                   #Estimated reservoir thickness in m
rw=0.11                 #Radius of well in m
rc=0.11                 #Radius of casing in m
Q=3.58e-2               #Flow rate m3/s

p = hp.war.gss(t,s)

p = hp.fit(p,t,s,'war')

hp.trial(p,t,s,'war')

hp.war.rpt(p,t,s,[Q,rw],'war',ttle = 'Diagnostic plot')



