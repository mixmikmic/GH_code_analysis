import math
import numpy as np
import ceo
import plotly.plotly as py       
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
import plotly.graph_objs as go
from plotly import tools
init_notebook_mode()

get_ipython().magic('pylab inline')

N = 401
L = 25.0
tid = ceo.StopWatch()
delta = L/(N-1)
print delta

atm = ceo.Atmosphere(15e-2,30,N_LAYER=1,altitude=[0e3],xi0=[1.0],
                     wind_speed=[10.0],wind_direction=[0],
                     L=L,NXY_PUPIL=N,fov=60.0*ceo.constants.ARCSEC2RAD,duration=30,
                     filename='singleLayer.bin')

src =ceo.Source('R',resolution=(N,N))

tel = ceo.Mask(N*N)
src.masked(tel)

tau = 0.0
src.reset()
tid.tic()
atm.get_phase_screen(src,delta,N,delta,N,tau)
tid.toc()
print "ET=%.4fms"%tid.elapsedTime
print src.wavefront.rms(-6)
ps1 = src.phase.host(units='micron')
trace1 = go.Heatmap(z=ps1)

tau = 0.0
src.reset()
tid.tic()
atm.ray_tracing(src,delta,N,delta,N,tau)
tid.toc()
print "ET=%.4fms"%tid.elapsedTime
print src.wavefront.rms(-6)
ps2 = src.phase.host(units='micron')
trace2 = go.Heatmap(z=ps2)

fig = tools.make_subplots(rows=1, cols=2)
fig.append_trace(trace1,1,1)
fig.append_trace(trace2,1,2)
py.image.ishow(fig,width=650,height=400)
#iplot(fig)

layer_phase_screen = atm.layers[0].phase_screen.host(units='micron')
imshow(layer_phase_screen)
colorbar()

fig,(ax1,ax2) = subplots(nrows=1,ncols=2)
ax1.imshow(ps1,vmin=layer_phase_screen.min(),vmax=layer_phase_screen.max())
ax2.imshow(ps2,vmin=layer_phase_screen.min(),vmax=layer_phase_screen.max())

src.reset()
tid.tic()
atm.ray_tracing(src,delta,N,delta,N,1.0)
tid.toc()
ps = src.phase.host(units='micron')
imshow(ps,vmin=layer_phase_screen.min(),vmax=layer_phase_screen.max())

src.reset()
tid.tic()
atm.ray_tracing(src,delta,N,delta,N,2.1)
tid.toc()
ps = src.phase.host(units='micron')
imshow(ps,vmin=layer_phase_screen.min(),vmax=layer_phase_screen.max())

layer_phase_screen_1 = atm.layers[0].phase_screen.host(units='micron')
imshow(layer_phase_screen_1,vmin=layer_phase_screen.min(),vmax=layer_phase_screen.max())

nTau = 300
wfe_rms = np.zeros((2,nTau))
tid = ceo.StopWatch()
tid.tic()
for k in range(nTau):
    tau = k*1e-2
    src.reset()
    atm.get_phase_screen(src,delta,N,delta,N,tau)
    wfe_rms[0,k] = src.wavefront.rms(-6)
    src.reset()
    atm.ray_tracing(src,delta,N,delta,N,tau)
    wfe_rms[1,k] = src.wavefront.rms(-6)
tid.toc()
print "ET: %.2fs"%(1e-3*tid.elapsedTime)

traces = []
traces.append( go.Trace(y=wfe_rms[0,:],name='Polar-log') )
traces.append( go.Trace(y=wfe_rms[1,:],name='Ray tracing') )
iplot(go.Data(traces))

trace = go.Trace(y=wfe_rms[0,:]-wfe_rms[1,:])
iplot(go.Data([trace]))

atm = ceo.Atmosphere(15e-2,30,N_LAYER=1,altitude=[10e3],xi0=[1.0],
                     wind_speed=[10.0],wind_direction=[0*math.pi/4],
                     L=L,NXY_PUPIL=N,fov=60.0*2*ceo.constants.ARCSEC2RAD,
                     filename='threeLayers.bin',N_DURATION=10)

src =ceo.Source('R',zenith=np.ones((1,3))*0.5*ceo.constants.ARCMIN2RAD,
                azimuth=np.arange(3)*2*math.pi/3,resolution=(N,N))
#src =ceo.Source('R',resolution=(N,N))
tel = ceo.Mask(N*N*src.N_SRC)
src.masked(tel)

tau = 0.0
src.reset()
tid.tic()
atm.get_phase_screen(src,delta,N,delta,N,tau)
tid.toc()
print "ET=%.4fms"%tid.elapsedTime
ps1 = src.phase.host(units='micron')
#trace1 = go.Heatmap(z=src.phase.host(units='micron'))
src.wavefront.rms(-6)

src.reset()
tid.tic()
atm.ray_tracing(src,delta,N,delta,N,tau)
tid.toc()
print "ET=%.4fms"%tid.elapsedTime
ps2 = src.phase.host(units='micron')
#trace2 = go.Heatmap(z=src.phase.host(units='micron'))
src.wavefront.rms(-6)

fig,(ax1,ax2) = subplots(nrows=1,ncols=2)
ax1.imshow(ps1)
ax2.imshow(ps2)

nTau = 300
wfe_rms = np.zeros((src.N_SRC,nTau))
for k in range(nTau):
    tau = k*1e-2
    src.reset()
    atm.ray_tracing(src,delta,N,delta,N,tau)
    wfe_rms[:,k] = src.wavefront.rms(-6)

traces = []
for k in range(src.N_SRC):
    traces.append( go.Trace(y=wfe_rms[k,:],name='Ray tracing #%d'%k) )
iplot(go.Data(traces))

xc = np.ravel(10e3*src.zenith*np.cos(src.azimuth))
yc = np.ravel(10e3*src.zenith*np.sin(src.azimuth))
x0 = -L/2+xc
y0 = -L/2+yc
tau = 0
print atm.layers[0].WIDTH
layer_phase_screen = atm.layers[0].phase_screen.host(units='micron')

imshow(layer_phase_screen)

atm = ceo.Atmosphere(15e-2,30,N_LAYER=2,altitude=[0,10e3],xi0=[0.7,0.3],
                     wind_speed=[10.0,20.0],wind_direction=[0,4*math.pi/3],
                     L=L,NXY_PUPIL=N,fov=60.0*2*ceo.constants.ARCSEC2RAD)

tau = 0.0
src.reset()
tid.tic()
atm.get_phase_screen(src,delta,N,delta,N,tau)
tid.toc()
print "ET=%.4fms"%tid.elapsedTime
src.wavefront.rms(-6)

tau = 0.0
src.reset()
tid.tic()
atm.ray_tracing(src,delta,N,delta,N,tau)
tid.toc()
print "ET=%.4fms"%tid.elapsedTime
src.wavefront.rms(-6)

nTau = 300
wfe_rms = np.zeros((src.N_SRC,nTau))
for k in range(nTau):
    tau = k*1e-2
    src.reset()
    atm.ray_tracing(src,delta,N,delta,N,tau)
    wfe_rms[:,k] = src.wavefront.rms(-6)

traces = []
for k in range(src.N_SRC):
    traces.append( go.Trace(y=wfe_rms[k,:],name='Ray tracing #%d'%k) )
iplot(go.Data(traces))

layer_phase_screen = atm.layers[0].phase_screen.host(units='micron')
imshow(layer_phase_screen)

layer_phase_screen = atm.layers[1].phase_screen.host(units='micron')
imshow(layer_phase_screen)

src.reset()
atm.ray_tracing(src,delta,N,delta,N,tau)
ps = src.phase.host(units='micron')
imshow(ps)

atm= ceo.GmtAtmosphere(15e-2,30,L=L,NXY_PUPIL=N,fov=2*ceo.constants.ARCMIN2RAD,
                       duration=30,filename='/home/ubuntu/phaseScreens.bin')

nTau = 3000
wfe_rms = np.zeros((src.N_SRC,nTau))
tid.tic()
for k in range(nTau):
    tau = k*1e-2
    src.reset()
    atm.ray_tracing(src,delta,N,delta,N,tau)
    wfe_rms[:,k] = src.wavefront.rms(-6)
tid.toc()
print "ET=%.4fms"%tid.elapsedTime

traces = []
for k in range(src.N_SRC):
    traces.append( go.Trace(y=wfe_rms[k,:],name='Ray tracing #%d'%k) )
iplot(go.Data(traces))

print "Theoretical rms : %.2fmicron"%(np.sqrt(ceo.phaseStats.variance(atmosphere=atm))*550e-3*0.5/math.pi)
print "Experimental rms: %.2fmicron"%np.sqrt(np.mean(wfe_rms**2))

src.reset()
atm.ray_tracing(src,delta,N,delta,N,tau)
ps = src.phase.host(units='micron')
imshow(ps)



