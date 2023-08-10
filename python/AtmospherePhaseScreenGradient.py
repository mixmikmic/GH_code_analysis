import math
import numpy as np
import ceo
get_ipython().magic('pylab inline')

#atm = ceo.Atmosphere(0.15,30,altitude=10e3,wind_speed=10)
atm =  ceo.GmtAtmosphere(0.15,30)

NL = 20
NA = NL+1
src = ceo.Source("V",resolution=(NA,NA))

D = 8.0
#tel = ceo.Telescope(NL*16)
tel = ceo.Telescope(NL*16,D)
dm  = ceo.Mask(NA,D)

imshow(tel.f.host(shape=(NL*16,NL*16)),interpolation='None')

d = D/NL
cog = ceo.Centroiding(NL)
cog.fried_geometry(dm, tel, 16, 0.5)
src.masked(dm)

imshow(dm.f.host(shape=(NA,NA)),interpolation='None')

p = D/(NA-1)
atm.get_phase_screen(src,p,NA,p,NA,0.0)

dm_mask = dm.f
src_phase = src.phase
imshow(src_phase.host(units='micron',zm=True,mask=dm_mask.host()),interpolation='None')
colorbar()

atm.get_phase_screen_gradient(cog,NL,d,src,0.0)
c = cog.c.host(units='arcsec')

imshow(c.reshape(NL*2,NL).transpose(),interpolation='None')
colorbar()

src_lmmse = ceo.Lmmse(atm,src,src,d,NL,dm,"MINRES")
src_lmmse.estimation(cog)

src_lmmse_phase = src_lmmse.phase
imshow(np.concatenate((src_phase.host_data,
                            src_lmmse_phase.host(units='micron',zm=True,mask=dm_mask.host())),axis=1),
           interpolation='none')
colorbar()

ps_e = src_lmmse_phase.host_data - src_phase.host_data
imshow(ps_e*1e3,interpolation='none')
colorbar()

print "wavefront error: %5.2fnm" % (np.std(ps_e[dm_mask.host_data!=0])*1e3)

n = 6
nPx = n*NL+1

gs = ceo.Source("K",resolution=(nPx,nPx))
tel_osf = ceo.Telescope(nPx,D)
gs.masked(tel_osf)

gs_lmmse = ceo.Lmmse(atm,gs,gs,d,NL,tel_osf,"MINRES",osf=n)
gs_lmmse.estimation(cog)

p = D/(nPx-1)
atm.get_phase_screen(gs,p,nPx,p,nPx,0.0)

tel_osf_mask = tel_osf.f
gs_phase = gs.phase
gs_lmmse_phase = gs_lmmse.phase
fig = figure(figsize=(10,5))
imshow(np.concatenate((gs_phase.host(units='micron',zm=True,mask=tel_osf_mask.host()),
                            gs_lmmse_phase.host(units='micron',zm=True,mask=tel_osf_mask.host())),axis=1),
           interpolation='none')
colorbar()

gs_ps_e = gs_lmmse_phase.host_data - gs_phase.host_data
imshow(gs_ps_e*1e3,interpolation='none')
colorbar()
print "wavefront error: %5.2fnm" % (np.std(gs_ps_e.ravel()[tel_osf_mask.host_data.ravel()!=0])*1e3)



