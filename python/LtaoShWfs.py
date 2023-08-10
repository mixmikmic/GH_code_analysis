import numpy as np
import ceo
get_ipython().magic('pylab inline')

nLenslet = 60
NA = nLenslet + 1;
D = 25.5
n = 6
nPx = n*nLenslet + 1

nGS = 6
gs = ceo.Source("K",
                 zenith=np.ones(nGS)*30*math.pi/180/3600,
                 azimuth=np.linspace(0,nGS-1,nGS)*2*math.pi/nGS,
                 height = 90e3,
                 resolution=(nPx,nPx))
calib_src = ceo.Source("K", resolution=(nPx,nPx))

tel = ceo.GMT(nPx,D)
gs.masked(tel)
calib_src.masked(tel)

d = D/nLenslet
wfs = ceo.ShackHartmann(nLenslet, n, d, N_PX_IMAGE=2*(n+1))
wfs.calibrate(calib_src,0.5)
imshow(wfs.flux.host().T,interpolation='none')

px_scale = 2.179e-6/d/2
coef_med = []
px = arange(0,1,0.05)
for k in px:
    wfs.pointing(np.array(-px_scale*k,dtype=np.float32,ndmin=1),np.array(0.0,dtype=np.float32,ndmin=1))
    wfs.reset()
    wfs.analyze(calib_src)
    c = wfs.c.host()
    cx = c[0,0:c.size/2]
    m = wfs.valid_lenslet.f.host()
    cx = cx[m.flatten()>0]
    #print k, np.mean(cx/px_scale/k) , np.median(cx/px_scale/k)
    coef_med.append(np.median(cx/px_scale))
cp = np.polyfit(px, coef_med, 1)
print cp
plot(px,coef_med,px,coef_med/cp[0])
grid()

wfs = ceo.ShackHartmann(nLenslet, n, d, N_GS = nGS, N_PX_IMAGE=2*(n+1))
#wfs.slopes_gain = 1.0/cp[0]

wfs.calibrate(gs,0.5)

print wfs.valid_actuator.f.host().shape

validActuator = wfs.valid_actuator
validActuator_f = validActuator.f
imshow(validActuator_f.host(shape=((nLenslet+1)*nGS,(nLenslet+1))).T,interpolation='None')
validActuator_f.host_data.sum()

wfs.analyze(gs)

#figure(figsize=(12,12))
#imshow(wfs.frame.host().transpose(),interpolation='none')

atm =ceo.GmtAtmosphere(20e-2,30)
p = D/(nPx-1)
atm.get_phase_screen(gs,  p, nPx, p, nPx, 0.0)

#figure(figsize=(12,12))
#imshow(gs.phase.host().transpose(),interpolation='none')

wfs.reset()
wfs.analyze(gs)

#figure(figsize=(12,12))
#imshow(wfs.frame.host().transpose(),interpolation='none')

#figure(figsize=(12,12))
#imshow(wfs.c.host().reshape(2*nLenslet*nGS
#                            ,nLenslet).transpose(),interpolation='none')
#colorbar()

src = ceo.Source("K",resolution=(NA,NA))
src.masked(validActuator)

lmmse = ceo.LmmseSH(atm,gs,src,wfs,"MINRES")

lmmse.estimation(wfs)

lmmse_phase = lmmse.phase
mask_actuator = validActuator_f.host_data[0:nLenslet+1,:]
imshow(lmmse_phase.host(units='micron',zm=True,mask=mask_actuator),interpolation='none')
colorbar()

atm.get_phase_screen(src,d,NA,d,NA,0.0)

src_phase = src.phase
ps_e = src_phase.host(units='micron',zm=True,mask=mask_actuator) -     lmmse_phase.host(units='micron',zm=True,mask=mask_actuator)
wfe_rms = np.std(ps_e[mask_actuator!=0])*1e3
print "wavefront error: %6.2fnm" % wfe_rms
imshow(np.concatenate((src_phase.host_data,lmmse_phase.host_data),axis=1),interpolation='none')
colorbar()

imshow(ps_e*1e3,interpolation='None')
colorbar()

np.exp(-(2*math.pi*wfe_rms/2.2e3)**2)



