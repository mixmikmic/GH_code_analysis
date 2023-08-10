import numpy as np
import ceo
get_ipython().magic('pylab inline')

nLenslet = 60
D = 25.5
n = 6
nPx = n*nLenslet+1

nGS = 1
zen = np.random.uniform(-1,1,nGS)*5*math.pi/180/60
azi = np.random.uniform(0,2*math.pi,nGS)
gs = ceo.Source("K",zenith=zen,azimuth=azi,height=float("inf"),resolution=(nPx,nPx))

tel = ceo.GMT(nPx,D)
#tel = ceo.Telescope(nPx)
#tel = ceo.Mask(nPx)
gs.masked(tel)

imshow(tel.f.host(shape=(nPx,nPx)),interpolation='none')

wfs = ceo.ShackHartmann(nLenslet, n, D/nLenslet,N_PX_IMAGE=n,N_GS = nGS)

wfs.calibrate(gs,0.5)

wfs.analyze(gs)

wfs.frame.host().shape

figure(figsize=(12,12))
imshow(wfs.frame.host().transpose(),interpolation='none')

print wfs.frame.host()

atm =ceo.GmtAtmosphere(15e-2,30)
p = D/nPx
atm.get_phase_screen(gs,  p, nPx, p, nPx, 0.0)

figure(figsize=(12,12))
imshow(gs.phase.host(units='micron').transpose(),interpolation='none')
colorbar()

wfs.reset()
wfs.analyze(gs)

figure(figsize=(12,12))
imshow(wfs.frame.host().transpose(),interpolation='none')

figure(figsize=(12,12))
imshow(wfs.c.host(units='arcsec').reshape(2*nLenslet*nGS
                            ,nLenslet).transpose(),interpolation='none')
colorbar()





