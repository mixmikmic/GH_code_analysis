import math
import numpy as np
import ceo
get_ipython().magic('pylab inline')

atm =  ceo.GmtAtmosphere(0.15,60)

NL = 60
NA = NL+1
lgs = ceo.Source("V",
                 zenith=np.ones(6)*30*math.pi/180/3600,
                 azimuth=np.linspace(0,5,6)*2*math.pi/6,
                 height = 90e3,
                 resolution=(NA,NA))

D = 25.5
#tel = ceo.Telescope(NL*16)
#dm  = ceo.Telescope(NA)
tel = ceo.GMT(NL*16,D)
dm  = ceo.Mask(NA,D)

d = D/NL
cog = ceo.Centroiding(NL,N_SOURCE=lgs.size)
cog.fried_geometry(dm, tel, 16, 0.5)
lgs.masked(dm)

dm_mask = dm.f

atm.get_phase_screen_gradient(cog,NL,d,lgs,0.0)
c = cog.c.host(units='arcsec')

imshow(c.reshape(NL*lgs.size*2,NL).transpose(),interpolation='none')
#ceog.heatmap(c.reshape(NL*6*2,NL).transpose(), filename=PLOTLY_PATH+"wavefront gradient")

src = ceo.Source("K",resolution=(NA,NA))
src.masked(dm)
atm.get_phase_screen(src,d,NA,d,NA,0.0)

et = ceo.StopWatch()
et.tic()
src_lmmse = ceo.Lmmse(atm,lgs,src,d,NL,dm,"MINRES")
et.toc()
print "ET = %.2fms"%et.elapsedTime

et.tic()
src_lmmse.reset()
src_lmmse.estimation(cog)
et.toc()
src_phase = src.phase
src_lmmse_phase = src_lmmse.phase
print "ET = %.2fms"%et.elapsedTime

ps_e = src_lmmse_phase.host(units='micron',
                            zm=True,mask=dm_mask.host()) - src_phase.host(units='micron',zm=True,mask=dm_mask.host_data)
print "wavefront error: %6.2fnm" % (np.std(ps_e[dm_mask.host_data.reshape((NA,NA))!=0])*1e3)

imshow(np.concatenate((src_phase.host_data, src_lmmse_phase.host_data),axis=1),
             interpolation='none')
colorbar()

imshow(ps_e*1e3,interpolation='none')
colorbar()



