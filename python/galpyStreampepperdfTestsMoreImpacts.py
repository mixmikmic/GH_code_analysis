from IPython.display import HTML

HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
<form action="javascript:code_toggle()"><input type="submit" value="Click here to toggle on/off the raw code."></form>''')

try:
    reload(galpy.df_src.streampepperdf)
    reload(galpy.df_src.streampepperdf.galpy.df_src.streamgapdf)
    reload(galpy.df_src.streampepperdf.galpy.df_src.streamdf)
except NameError:
    import galpy.df_src.streampepperdf
import time
import numpy
from scipy import signal, ndimage
import statsmodels.api as sm
from galpy.potential import LogarithmicHaloPotential
from galpy.orbit import Orbit
from galpy.actionAngle import actionAngleIsochroneApprox
from galpy.util import bovy_conversion, bovy_coords
from galpy.util import bovy_plot
get_ipython().magic('pylab inline')
import seaborn as sns
R0, V0= 8., 220.

lp= LogarithmicHaloPotential(normalize=1.,q=0.9)
xv_prog_init= numpy.array([30.,0.,0.,0., 105.74895, 105.74895])
def convert_to_cylindrical(xv):
    R,phi,Z= bovy_coords.rect_to_cyl(xv[:,0],xv[:,1],xv[:,2])
    vR,vT,vZ= bovy_coords.rect_to_cyl_vec(xv[:,3],xv[:,4],xv[:,5],R,phi,Z,cyl=True)
    out= numpy.empty_like(xv)
    # Preferred galpy arrangement of cylindrical coordinates
    out[:,0]= R
    out[:,1]= vR
    out[:,2]= vT
    out[:,3]= Z
    out[:,4]= vZ
    out[:,5]= phi
    return out
sigv= 0.365*(10./2.)**(1./3.)
RvR_prog_init= convert_to_cylindrical(xv_prog_init[:,numpy.newaxis].T)[0,:]
prog_init= Orbit([RvR_prog_init[0]/R0,RvR_prog_init[1]/V0,RvR_prog_init[2]/V0,
                  RvR_prog_init[3]/R0,RvR_prog_init[4]/V0,RvR_prog_init[5]])
times= numpy.linspace(0.,10.88/bovy_conversion.time_in_Gyr(V0,R0),10001)
prog_init.integrate(times,lp)
xv_unp_peri_prog= [prog_init.x(times[-1]),prog_init.y(times[-1]),prog_init.z(times[-1]),
              prog_init.vx(times[-1]),prog_init.vy(times[-1]),prog_init.vz(times[-1])]
RvR_unp_peri_prog= convert_to_cylindrical(numpy.array(xv_unp_peri_prog)[:,numpy.newaxis].T)[0,:]
prog_unp_peri= Orbit([RvR_unp_peri_prog[0],RvR_unp_peri_prog[1],RvR_unp_peri_prog[2],
                      RvR_unp_peri_prog[3]+0.0,RvR_unp_peri_prog[4],RvR_unp_peri_prog[5]])
aAI= actionAngleIsochroneApprox(pot=lp,b=0.8)

sdf_pepper_5impacts= galpy.df_src.streampepperdf.streampepperdf(    sigv/V0,progenitor=prog_unp_peri,pot=lp,aA=aAI,
    leading=False,nTrackChunks=26,nTrackIterations=1,
    sigMeanOffset=4.5,
    tdisrupt=10.88/bovy_conversion.time_in_Gyr(V0,R0),
    Vnorm=V0,Rnorm=R0,
    impactb=[0.1/R0 for ii in range(5)],
    subhalovel=numpy.array([[36.82200571,102.7700529,169.4174464],
                            [6.82200571,132.7700529,149.4174464],
                            [126.82200571,32.7700529,89.4174464],
                            [56.82200571,32.7700529,89.4174464],
                            [-36.82200571,22.7700529,-149.4174464]])/V0,
    timpact=[2.88/bovy_conversion.time_in_Gyr(V0,R0),1.88/bovy_conversion.time_in_Gyr(V0,R0),
             2.38/bovy_conversion.time_in_Gyr(V0,R0),3.38/bovy_conversion.time_in_Gyr(V0,R0),
            4.44/bovy_conversion.time_in_Gyr(V0,R0)],
    impact_angle=[-1.34,-0.34,-1.,-2.,-3.2],
    GM=[10.**-2.7/bovy_conversion.mass_in_1010msol(V0,R0) for ii in range(5)],
    rs=[0.35/R0 for ii in range(5)],
    spline_order=1) 

sdf_pepper_6impacts= galpy.df_src.streampepperdf.streampepperdf(    sigv/V0,progenitor=prog_unp_peri,pot=lp,aA=aAI,
    leading=False,nTrackChunks=26,nTrackIterations=1,
    sigMeanOffset=4.5,
    tdisrupt=10.88/bovy_conversion.time_in_Gyr(V0,R0),
    Vnorm=V0,Rnorm=R0,
    impactb=[0.1/R0 for ii in range(6)],
    subhalovel=numpy.array([[36.82200571,102.7700529,169.4174464],
                            [6.82200571,132.7700529,149.4174464],
                            [126.82200571,32.7700529,89.4174464],
                            [56.82200571,32.7700529,89.4174464],
                            [-36.82200571,22.7700529,-149.4174464],
                           [-136.82200571,22.7700529,-149.4174464]])/V0,
    timpact=[2.88/bovy_conversion.time_in_Gyr(V0,R0),1.88/bovy_conversion.time_in_Gyr(V0,R0),
             2.38/bovy_conversion.time_in_Gyr(V0,R0),3.38/bovy_conversion.time_in_Gyr(V0,R0),
             4.44/bovy_conversion.time_in_Gyr(V0,R0),3.44/bovy_conversion.time_in_Gyr(V0,R0)],
    impact_angle=[-1.34,-0.34,-1.,-2.,-3.2,-2.5],
    GM=[10.**-2.7/bovy_conversion.mass_in_1010msol(V0,R0) for ii in range(6)],
    rs=[0.35/R0 for ii in range(6)],
    spline_order=1)

sdf_pepper_7impacts= galpy.df_src.streampepperdf.streampepperdf(    sigv/V0,progenitor=prog_unp_peri,pot=lp,aA=aAI,
    leading=False,nTrackChunks=26,nTrackIterations=1,
    sigMeanOffset=4.5,
    tdisrupt=10.88/bovy_conversion.time_in_Gyr(V0,R0),
    Vnorm=V0,Rnorm=R0,
    impactb=[0.1/R0 for ii in range(7)],
    subhalovel=numpy.array([[36.82200571,102.7700529,169.4174464],
                            [6.82200571,132.7700529,149.4174464],
                            [126.82200571,32.7700529,89.4174464],
                            [86.82200571,-32.7700529,49.4174464],
                            [26.82200571,-132.7700529,29.4174464],
                            [56.82200571,32.7700529,89.4174464],
                            [-36.82200571,22.7700529,-149.4174464]])/V0,
    timpact=[2.88/bovy_conversion.time_in_Gyr(V0,R0),1.88/bovy_conversion.time_in_Gyr(V0,R0),
             2.38/bovy_conversion.time_in_Gyr(V0,R0),3.38/bovy_conversion.time_in_Gyr(V0,R0),
            4.44/bovy_conversion.time_in_Gyr(V0,R0),3.44/bovy_conversion.time_in_Gyr(V0,R0),
            2.24/bovy_conversion.time_in_Gyr(V0,R0)],
    impact_angle=[-1.34,-0.34,-1.,-2.,-3.2,-2.5,-2.2],
    GM=[10.**-2.7/bovy_conversion.mass_in_1010msol(V0,R0) for ii in range(7)],
    rs=[0.35/R0 for ii in range(7)],
    spline_order=1) 

sdf_pepper_8impacts= galpy.df_src.streampepperdf.streampepperdf(    sigv/V0,progenitor=prog_unp_peri,pot=lp,aA=aAI,
    leading=False,nTrackChunks=26,nTrackIterations=1,
    sigMeanOffset=4.5,
    tdisrupt=10.88/bovy_conversion.time_in_Gyr(V0,R0),
    Vnorm=V0,Rnorm=R0,
    impactb=[0.1/R0 for ii in range(8)],
    subhalovel=numpy.array([[36.82200571,102.7700529,169.4174464],
                            [6.82200571,132.7700529,149.4174464],
                            [126.82200571,32.7700529,89.4174464],
                            [86.82200571,-32.7700529,49.4174464],
                            [186.82200571,-2.7700529,19.4174464],
                            [26.82200571,-132.7700529,29.4174464],
                            [56.82200571,32.7700529,89.4174464],
                            [-36.82200571,22.7700529,-149.4174464]])/V0,
    timpact=[2.88/bovy_conversion.time_in_Gyr(V0,R0),1.88/bovy_conversion.time_in_Gyr(V0,R0),
             2.38/bovy_conversion.time_in_Gyr(V0,R0),3.38/bovy_conversion.time_in_Gyr(V0,R0),
            4.44/bovy_conversion.time_in_Gyr(V0,R0),3.44/bovy_conversion.time_in_Gyr(V0,R0),
            2.24/bovy_conversion.time_in_Gyr(V0,R0),3.04/bovy_conversion.time_in_Gyr(V0,R0)],
    impact_angle=[-1.34,-0.34,-1.,-2.,-3.2,-2.5,-2.2,-0.8],
    GM=[10.**-2.7/bovy_conversion.mass_in_1010msol(V0,R0) for ii in range(8)],
    rs=[0.35/R0 for ii in range(8)],
    spline_order=1) 

Opars= numpy.linspace(-0.1,0.5,101)/bovy_conversion.freq_in_Gyr(V0,R0)
apars= numpy.linspace(0.,5.,101)
lowlim= numpy.array([sdf_pepper_5impacts.minOpar(da) for da in apars])
y= numpy.array([sdf_pepper_5impacts.pOparapar(Opars,a) for a in apars])
bovy_plot.bovy_dens2d(y.T,
                      origin='lower',
                      cmap='afmhot_r',colorbar=True,
                      vmin=0.,
                      xrange=[apars[0],apars[-1]],
                      yrange=[Opars[0]*bovy_conversion.freq_in_Gyr(V0,R0),
                              Opars[-1]*bovy_conversion.freq_in_Gyr(V0,R0)],
                     zlabel=r'$p(\Omega\parallel,\theta_\parallel)$')
plot(apars,lowlim*bovy_conversion.freq_in_Gyr(V0,R0),lw=4.,zorder=2)
plot(apars,apars/sdf_pepper_5impacts._tdisrupt*bovy_conversion.freq_in_Gyr(V0,R0),lw=4.,zorder=1)
xlabel(r'$\theta_\parallel$')
ylabel(r'$\Omega\parallel\,(\mathrm{Gyr}^{-1})$')

Opars= numpy.linspace(-0.1,0.5,101)/bovy_conversion.freq_in_Gyr(V0,R0)
apars= numpy.linspace(0.,5.,101)
lowlim= numpy.array([sdf_pepper_6impacts.minOpar(da) for da in apars])
y= numpy.array([sdf_pepper_6impacts.pOparapar(Opars,a) for a in apars])
bovy_plot.bovy_dens2d(y.T,
                      origin='lower',
                      cmap='afmhot_r',colorbar=True,
                      vmin=0.,
                      xrange=[apars[0],apars[-1]],
                      yrange=[Opars[0]*bovy_conversion.freq_in_Gyr(V0,R0),
                              Opars[-1]*bovy_conversion.freq_in_Gyr(V0,R0)],
                     zlabel=r'$p(\Omega\parallel,\theta_\parallel)$')
plot(apars,lowlim*bovy_conversion.freq_in_Gyr(V0,R0),lw=4.,zorder=2)
plot(apars,apars/sdf_pepper_6impacts._tdisrupt*bovy_conversion.freq_in_Gyr(V0,R0),lw=4.,zorder=1)
xlabel(r'$\theta_\parallel$')
ylabel(r'$\Omega\parallel\,(\mathrm{Gyr}^{-1})$')

Opars= numpy.linspace(-0.1,0.5,101)/bovy_conversion.freq_in_Gyr(V0,R0)
apars= numpy.linspace(0.,5.,101)
lowlim= numpy.array([sdf_pepper_7impacts.minOpar(da) for da in apars])
y= numpy.array([sdf_pepper_7impacts.pOparapar(Opars,a) for a in apars])
bovy_plot.bovy_dens2d(y.T,
                      origin='lower',
                      cmap='afmhot_r',colorbar=True,
                      vmin=0.,
                      xrange=[apars[0],apars[-1]],
                      yrange=[Opars[0]*bovy_conversion.freq_in_Gyr(V0,R0),
                              Opars[-1]*bovy_conversion.freq_in_Gyr(V0,R0)],
                     zlabel=r'$p(\Omega\parallel,\theta_\parallel)$')
plot(apars,lowlim*bovy_conversion.freq_in_Gyr(V0,R0),lw=4.,zorder=2)
plot(apars,apars/sdf_pepper_7impacts._tdisrupt*bovy_conversion.freq_in_Gyr(V0,R0),lw=4.,zorder=1)
xlabel(r'$\theta_\parallel$')
ylabel(r'$\Omega\parallel\,(\mathrm{Gyr}^{-1})$')

Opars= numpy.linspace(-0.1,0.5,101)/bovy_conversion.freq_in_Gyr(V0,R0)
apars= numpy.linspace(0.,5.,101)
lowlim= numpy.array([sdf_pepper_8impacts.minOpar(da,force_indiv_impacts=True) for da in apars])
y= numpy.array([sdf_pepper_8impacts.pOparapar(Opars,a) for a in apars])
bovy_plot.bovy_dens2d(y.T,
                      origin='lower',
                      cmap='afmhot_r',colorbar=True,
                      vmin=0.,
                      xrange=[apars[0],apars[-1]],
                      yrange=[Opars[0]*bovy_conversion.freq_in_Gyr(V0,R0),
                              Opars[-1]*bovy_conversion.freq_in_Gyr(V0,R0)],
                     zlabel=r'$p(\Omega\parallel,\theta_\parallel)$')
plot(apars,lowlim*bovy_conversion.freq_in_Gyr(V0,R0),lw=4.,zorder=2)
plot(apars,apars/sdf_pepper_8impacts._tdisrupt*bovy_conversion.freq_in_Gyr(V0,R0),lw=4.,zorder=1)
xlabel(r'$\theta_\parallel$')
ylabel(r'$\Omega\parallel\,(\mathrm{Gyr}^{-1})$')

xs= numpy.linspace(0.,4.5,101)
# Compute
dens_unp= numpy.array([super(galpy.df_src.streampepperdf.streampepperdf,sdf_pepper_5impacts)._density_par(x) for x in xs])
dens_approx_5impacts= numpy.array([sdf_pepper_5impacts.density_par(x,approx=True) for x in xs])
dens_5impacts= numpy.array([sdf_pepper_5impacts.density_par(x,approx=False) for x in xs])

figsize(12,4)
subplot(1,2,1)
plot(xs,dens_5impacts/numpy.sum(dens_5impacts)/(xs[1]-xs[0]),lw=4.)
plot(xs,dens_approx_5impacts/numpy.sum(dens_approx_5impacts)/(xs[1]-xs[0]),lw=4.)
plot(xs,dens_unp/numpy.sum(dens_unp)/(xs[1]-xs[0]),lw=4.)
#dum= hist(apar,bins=101,normed=True,histtype='step',color='k',zorder=0,lw=5.)
xlabel(r'$\theta_\parallel$')
subplot(1,2,2)
plot(xs,(dens_5impacts/numpy.sum(dens_5impacts)/(xs[1]-xs[0])         -dens_approx_5impacts/numpy.sum(dens_approx_5impacts)/(xs[1]-xs[0]))     /(dens_5impacts/numpy.sum(dens_5impacts)/(xs[1]-xs[0])),
     lw=4.)
xlabel(r'$\theta_\parallel$')
ylim(-0.05,0.05)

timexs= xs
start= time.time()
dum= numpy.array([sdf_pepper_5impacts.density_par(x,approx=True) for x in timexs])
approxTime= time.time()-start
start= time.time()
dum= numpy.array([sdf_pepper_5impacts.density_par(x,approx=False) for x in timexs])
print "Speed-up with approximation: %f; time / angle: %f ms" % ((time.time()-start)/approxTime,approxTime/len(timexs)*1000.)

xs= numpy.linspace(0.,4.5,101)
# Compute
dens_unp= numpy.array([super(galpy.df_src.streampepperdf.streampepperdf,sdf_pepper_6impacts)._density_par(x) for x in xs])
dens_approx_6impacts= numpy.array([sdf_pepper_6impacts.density_par(x,approx=True) for x in xs])
dens_6impacts= numpy.array([sdf_pepper_6impacts.density_par(x,approx=False) for x in xs])

figsize(12,4)
subplot(1,2,1)
plot(xs,dens_6impacts/numpy.sum(dens_6impacts)/(xs[1]-xs[0]),lw=4.)
plot(xs,dens_approx_6impacts/numpy.sum(dens_approx_6impacts)/(xs[1]-xs[0]),lw=4.)
plot(xs,dens_unp/numpy.sum(dens_unp)/(xs[1]-xs[0]),lw=4.)
#dum= hist(apar,bins=101,normed=True,histtype='step',color='k',zorder=0,lw=5.)
xlabel(r'$\theta_\parallel$')
subplot(1,2,2)
plot(xs,(dens_6impacts/numpy.sum(dens_6impacts)/(xs[1]-xs[0])         -dens_approx_6impacts/numpy.sum(dens_approx_6impacts)/(xs[1]-xs[0]))     /(dens_6impacts/numpy.sum(dens_6impacts)/(xs[1]-xs[0])),
     lw=4.)
xlabel(r'$\theta_\parallel$')
ylim(-0.05,0.05)

timexs= xs
start= time.time()
dum= numpy.array([sdf_pepper_6impacts.density_par(x,approx=True) for x in timexs])
approxTime= time.time()-start
start= time.time()
dum= numpy.array([sdf_pepper_6impacts.density_par(x,approx=False) for x in timexs])
print "Speed-up with approximation: %f; time / angle: %f ms" % ((time.time()-start)/approxTime,approxTime/len(timexs)*1000.)

xs= numpy.linspace(0.,4.5,101)
# Compute
dens_unp= numpy.array([super(galpy.df_src.streampepperdf.streampepperdf,sdf_pepper_7impacts)._density_par(x) for x in xs])
dens_approx_7impacts= numpy.array([sdf_pepper_7impacts.density_par(x,approx=True) for x in xs])
dens_7impacts= numpy.array([sdf_pepper_7impacts.density_par(x,approx=False) for x in xs])

figsize(12,4)
subplot(1,2,1)
plot(xs,dens_7impacts/numpy.sum(dens_7impacts)/(xs[1]-xs[0]),lw=4.)
plot(xs,dens_approx_7impacts/numpy.sum(dens_approx_7impacts)/(xs[1]-xs[0]),lw=4.)
plot(xs,dens_unp/numpy.sum(dens_unp)/(xs[1]-xs[0]),lw=4.)
#dum= hist(apar,bins=101,normed=True,histtype='step',color='k',zorder=0,lw=5.)
xlabel(r'$\theta_\parallel$')
subplot(1,2,2)
plot(xs,(dens_7impacts/numpy.sum(dens_7impacts)/(xs[1]-xs[0])         -dens_approx_7impacts/numpy.sum(dens_approx_7impacts)/(xs[1]-xs[0]))     /(dens_7impacts/numpy.sum(dens_7impacts)/(xs[1]-xs[0])),
     lw=4.)
xlabel(r'$\theta_\parallel$')
ylim(-0.05,0.05)

timexs= xs
start= time.time()
dum= numpy.array([sdf_pepper_7impacts.density_par(x,approx=True) for x in timexs])
approxTime= time.time()-start
start= time.time()
dum= numpy.array([sdf_pepper_7impacts.density_par(x,approx=False) for x in timexs])
print "Speed-up with approximation: %f; time / angle: %f ms" % ((time.time()-start)/approxTime,approxTime/len(timexs)*1000.)

xs= numpy.linspace(0.,4.5,101)
# Compute
dens_unp= numpy.array([super(galpy.df_src.streampepperdf.streampepperdf,sdf_pepper_8impacts)._density_par(x) for x in xs])
dens_approx_8impacts= numpy.array([sdf_pepper_8impacts.density_par(x,approx=True) for x in xs])
dens_8impacts= numpy.array([sdf_pepper_8impacts.density_par(x,approx=False) for x in xs])

figsize(12,4)
subplot(1,2,1)
plot(xs,dens_8impacts/numpy.sum(dens_8impacts)/(xs[1]-xs[0]),lw=4.)
plot(xs,dens_approx_8impacts/numpy.sum(dens_approx_8impacts)/(xs[1]-xs[0]),lw=4.)
plot(xs,dens_unp/numpy.sum(dens_unp)/(xs[1]-xs[0]),lw=4.)
#dum= hist(apar,bins=101,normed=True,histtype='step',color='k',zorder=0,lw=5.)
xlabel(r'$\theta_\parallel$')
subplot(1,2,2)
plot(xs,(dens_8impacts/numpy.sum(dens_8impacts)/(xs[1]-xs[0])         -dens_approx_8impacts/numpy.sum(dens_approx_8impacts)/(xs[1]-xs[0]))     /(dens_8impacts/numpy.sum(dens_8impacts)/(xs[1]-xs[0])),
     lw=4.)
xlabel(r'$\theta_\parallel$')
ylim(-0.05,0.05)

timexs= xs
start= time.time()
dum= numpy.array([sdf_pepper_8impacts.density_par(x,approx=True) for x in timexs])
approxTime= time.time()-start
start= time.time()
dum= numpy.array([sdf_pepper_8impacts.density_par(x,approx=False) for x in timexs])
print "Speed-up with approximation: %f; time / angle: %f ms" % ((time.time()-start)/approxTime,approxTime/len(timexs)*1000.)

xs= numpy.linspace(0.,4.5,101)
# Compute
mO_unp= numpy.array([super(galpy.df_src.streampepperdf.streampepperdf,sdf_pepper_5impacts).meanOmega(x,oned=True) for x in xs])
mO_approx_5impacts= numpy.array([sdf_pepper_5impacts.meanOmega(x,oned=True,approx=True) for x in xs])
mO_5impacts= numpy.array([sdf_pepper_5impacts.meanOmega(x,oned=True,approx=False) for x in xs])

figsize(12,4)
subplot(1,2,1)
plot(xs,mO_5impacts*bovy_conversion.freq_in_Gyr(V0,R0),lw=4.)
plot(xs,mO_approx_5impacts*bovy_conversion.freq_in_Gyr(V0,R0),lw=4.)
plot(xs,mO_unp*bovy_conversion.freq_in_Gyr(V0,R0),lw=4.)
ylim(0.1,0.5)
xlabel(r'$\theta_\parallel$')
ylabel(r'$\Omega_\parallel\,(\mathrm{Gyr}^{-1})$')
subplot(1,2,2)
plot(xs,(mO_5impacts-mO_approx_5impacts)/mO_5impacts,lw=4.)
xlabel(r'$\theta_\parallel$')
ylim(-0.01,0.01)

timexs= xs[::10]
start= time.time()
dum= numpy.array([sdf_pepper_5impacts.meanOmega(x,oned=True,approx=True) for x in timexs])
approxTime= time.time()-start
start= time.time()
dum= numpy.array([sdf_pepper_5impacts.meanOmega(x,oned=True,approx=False) for x in timexs])
print "Speed-up with approximation: %f; time / angle: %f ms" % ((time.time()-start)/approxTime,approxTime/len(timexs)*1000.)

xs= numpy.linspace(0.,4.5,101)
# Compute
mO_unp= numpy.array([super(galpy.df_src.streampepperdf.streampepperdf,sdf_pepper_6impacts).meanOmega(x,oned=True) for x in xs])
mO_approx_6impacts= numpy.array([sdf_pepper_6impacts.meanOmega(x,oned=True,approx=True) for x in xs])
mO_6impacts= numpy.array([sdf_pepper_6impacts.meanOmega(x,oned=True,approx=False) for x in xs])

figsize(12,4)
subplot(1,2,1)
plot(xs,mO_6impacts*bovy_conversion.freq_in_Gyr(V0,R0),lw=4.)
plot(xs,mO_approx_6impacts*bovy_conversion.freq_in_Gyr(V0,R0),lw=4.)
plot(xs,mO_unp*bovy_conversion.freq_in_Gyr(V0,R0),lw=4.)
ylim(0.1,0.5)
xlabel(r'$\theta_\parallel$')
ylabel(r'$\Omega_\parallel\,(\mathrm{Gyr}^{-1})$')
subplot(1,2,2)
plot(xs,(mO_6impacts-mO_approx_6impacts)/mO_6impacts,lw=4.)
xlabel(r'$\theta_\parallel$')
ylim(-0.01,0.01)

timexs= xs[::10]
start= time.time()
dum= numpy.array([sdf_pepper_6impacts.meanOmega(x,oned=True,approx=True) for x in timexs])
approxTime= time.time()-start
start= time.time()
dum= numpy.array([sdf_pepper_6impacts.meanOmega(x,oned=True,approx=False) for x in timexs])
print "Speed-up with approximation: %f; time / angle: %f ms" % ((time.time()-start)/approxTime,approxTime/len(timexs)*1000.)

xs= numpy.linspace(0.,4.5,101)
# Compute
mO_unp= numpy.array([super(galpy.df_src.streampepperdf.streampepperdf,sdf_pepper_7impacts).meanOmega(x,oned=True) for x in xs])
mO_approx_7impacts= numpy.array([sdf_pepper_7impacts.meanOmega(x,oned=True,approx=True) for x in xs])
mO_7impacts= numpy.array([sdf_pepper_7impacts.meanOmega(x,oned=True,approx=False) for x in xs])

figsize(12,4)
subplot(1,2,1)
plot(xs,mO_7impacts*bovy_conversion.freq_in_Gyr(V0,R0),lw=4.)
plot(xs,mO_approx_7impacts*bovy_conversion.freq_in_Gyr(V0,R0),lw=4.)
plot(xs,mO_unp*bovy_conversion.freq_in_Gyr(V0,R0),lw=4.)
ylim(0.1,0.5)
xlabel(r'$\theta_\parallel$')
ylabel(r'$\Omega_\parallel\,(\mathrm{Gyr}^{-1})$')
subplot(1,2,2)
plot(xs,(mO_7impacts-mO_approx_7impacts)/mO_7impacts,lw=4.)
xlabel(r'$\theta_\parallel$')
ylim(-0.01,0.01)

timexs= xs[::10]
start= time.time()
dum= numpy.array([sdf_pepper_7impacts.meanOmega(x,oned=True,approx=True) for x in timexs])
approxTime= time.time()-start
start= time.time()
dum= numpy.array([sdf_pepper_7impacts.meanOmega(x,oned=True,approx=False) for x in timexs])
print "Speed-up with approximation: %f; time / angle: %f ms" % ((time.time()-start)/approxTime,approxTime/len(timexs)*1000.)

xs= numpy.linspace(0.,4.5,101)
# Compute
mO_unp= numpy.array([super(galpy.df_src.streampepperdf.streampepperdf,sdf_pepper_8impacts).meanOmega(x,oned=True) for x in xs])
mO_approx_8impacts= numpy.array([sdf_pepper_8impacts.meanOmega(x,oned=True,approx=True) for x in xs])
mO_8impacts= numpy.array([sdf_pepper_8impacts.meanOmega(x,oned=True,approx=False) for x in xs])

figsize(12,4)
subplot(1,2,1)
plot(xs,mO_8impacts*bovy_conversion.freq_in_Gyr(V0,R0),lw=4.)
plot(xs,mO_approx_8impacts*bovy_conversion.freq_in_Gyr(V0,R0),lw=4.)
plot(xs,mO_unp*bovy_conversion.freq_in_Gyr(V0,R0),lw=4.)
ylim(0.1,0.5)
xlabel(r'$\theta_\parallel$')
ylabel(r'$\Omega_\parallel\,(\mathrm{Gyr}^{-1})$')
subplot(1,2,2)
plot(xs,(mO_8impacts-mO_approx_8impacts)/mO_8impacts,lw=4.)
xlabel(r'$\theta_\parallel$')
ylim(-0.01,0.01)

timexs= xs[::10]
start= time.time()
dum= numpy.array([sdf_pepper_8impacts.meanOmega(x,oned=True,approx=True) for x in timexs])
approxTime= time.time()-start
start= time.time()
dum= numpy.array([sdf_pepper_8impacts.meanOmega(x,oned=True,approx=False) for x in timexs])
print "Speed-up with approximation: %f; time / angle: %f ms" % ((time.time()-start)/approxTime,approxTime/len(timexs)*1000.)



