import hypy as hp

hp.edit('/home/pianarol/AnacondaProjects/thn_ds1.txt')

t,s = hp.ldf('/home/pianarol/AnacondaProjects/thn_ds1.txt')
hp.plot(t,s)

hp.diagnostic(t,s)

p0 = hp.thn.gss(t,s)
hp.trial(p0,t,s,'thn')

p = hp.fit(p0,t,s,'thn')
q = 0.012 #pumping rate in m3/s
r = 20 #radial distance in m
d = [q,r]

hp.thn.rpt(p,t,s,d,'thn',Author='Your name',ttle = 'Interference test', Rapport = 'My rapport', filetype = 'pdf')



