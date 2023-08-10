import hypy as hp

hp.edit('/home/pianarol/AnacondaProjects/thc_ds1.txt')

t,s = hp.ldf('/home/pianarol/AnacondaProjects/thc_ds1.txt')

hp.plot(t,s)

hp.diagnostic(t,s)

p0 = hp.thc.gss(t,s)
hp.trial(p0,t,s,'thc')

help(hp.trial)

p0 = [0.65,80,2e5]
hp.trial(p0,t,s,'thc')

p = hp.fit(p0,t,s,'thc')
hp.trial(p,t,s,'thc')

q = 0.030 #Pumping rate in m3/s
r = 20 #radial distance in m
d = [q,r]
hp.thc.rpt(p,t,s,d,'thc',Author='Your name',ttle = 'Interference test', Rapport = 'My rapport', filetype = 'pdf')



