import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, interactive, fixed
get_ipython().magic('matplotlib notebook')

def sigmaXX(x_axis,y_axis,Ox,Oy,D,sign):
    x,y=np.meshgrid(x_axis,y_axis)
    sXX=(-D*(y-Oy)*(3*(x-Ox)**2+(y-Oy)**2))/(((x-Ox)**2+(y-Oy)**2)**2)*sign
    return sXX

def sigmaYY(x_axis,y_axis,Ox,Oy,D,sign):
    x,y=np.meshgrid(x_axis,y_axis)
    sYY=(D*(y-Oy)*((x-Ox)**2-(y-Oy)**2))/(((x-Ox)**2+(y-Oy)**2)**2)*sign
    return sYY

def sigmaXY(x_axis,y_axis,Ox,Oy,D,sign):
    x,y=np.meshgrid(x_axis,y_axis)
    sXY=D*(x-Ox)*((x-Ox)**2-(y-Oy)**2)/(((x-Ox)**2+(y-Oy)**2)**2)*sign
    return sXY

def screw_sigmaZX(x_axis,y_axis,Ox,Oy,D,sign):
    x,y=np.meshgrid(x_axis,y_axis)
    sZX=-D*(y-Oy)/((x-Ox)**2+(y-Oy)**2)*sign
    return sZX

def screw_sigmaZY(x_axis,y_axis,Ox,Oy,D,sign):
    x,y=np.meshgrid(x_axis,y_axis)
    sZY=-D*(x-Ox)/((x-Ox)**2+(y-Oy)**2)*sign
    return sZY

def screw_sigmaTZ(x_axis,y_axis,Ox,Oy,D,sign):
    x,y=np.meshgrid(x_axis,y_axis)
    sTZ=D/(np.sqrt(x**2+y**2))*sign
    return sTZ

def sigmaZZ(x_axis,y_axis,Ox,Oy,D,sign):
    x,y=np.meshgrid(x_axis,y_axis)
    sZZ=-2*nu*D*((y-Oy)/(x-Ox)**2+(y-Oy)**2)
    return sZZ
    
def press(x_axis,y_axis,Ox,Oy,D,sign):
    x,y=np.meshgrid(x_axis,y_axis)
    p=(sigmaXX(x_axis,y_axis,Ox,Oy,D,sign)+
       sigmaYY(x_axis,y_axis,Ox,Oy,D,sign)+
       sigmaZZ(x_axis,y_axis,Ox,Oy,D,sign))/3
    return p


def single_dislocation(stress,character='edge'):    
    plt.figure(figsize=(7,5))
    plt.pcolormesh(X_axis,Y_axis,stress,vmin=-1e7,vmax=1e7,cmap='Spectral_r',shading='gouraud')
    if character=='edge':
        plt.plot(0,0,marker=mtop,markeredgecolor='white',markersize=10,markeredgewidth=2)
    if character!='edge':
        plt.plot(0,0,marker='+',markeredgecolor='white',markersize=10,markeredgewidth=2)
    plt.axis('Off')
    plt.axis('Equal')
    plt.colorbar()
    
def two_dislocations(component='xx'):
    if component=='xx':
        stress=sigmaXX(X_axis,Y_axis,-0.5e-6,-0.5e-6,D,1)+sigmaXX(
                      X_axis,Y_axis,0.5e-6,0.5e-6,D,1)
    if component=='xy':
        stress=sigmaXY(X_axis,Y_axis,-0.5e-6,-0.5e-6,D,1)+sigmaXY(
                      X_axis,Y_axis,0.5e-6,0.5e-6,D,1)
    stress_fig,ax=plt.subplots(figsize=(7,5))
    fig=ax.imshow(stress,vmin=-5e7,vmax=5e7,cmap='Spectral_r',origin='lower')
    dis1,=ax.plot(50,50,marker=mtop,mec='white',ms=10,mew=2,lw=0)
    dis2,=ax.plot(150,150,marker=mbot,mec='white',ms=10,mew=2,lw=0)
    #ax.axis('Off')
    ax.axis('Equal')
    ax.axis('Tight')
    if component=='xx':
        ax.set_title(r'$\sigma_{xx}$')
    if component=='xy':
        ax.set_title(r'$\sigma_{xy}$')
    stress_fig.colorbar(fig,label='Pa')
    return fig, dis1,dis2

def add_dislocation(fig,dis1,dis2,Ox,Oy,sign,component='xx'):
    Ox_image=(Ox)*(2*a)/200-a
    Oy_image=(Oy)*(2*a)/200-a
    
    if component=='xx':
        stress_xx1=sigmaXX(X_axis,Y_axis,0.0,0.0,D,sign)
        stress_xx2=sigmaXX(X_axis,Y_axis,Ox_image,Oy_image,D,1)
        stress=stress_xx1+stress_xx2
    if component=='xy':
        stress_xy1=sigmaXY(X_axis,Y_axis,0.0,0.0,D,sign)
        stress_xy2=sigmaXY(X_axis,Y_axis,Ox_image,Oy_image,D,1)
        stress=stress_xy1+stress_xy2
        
    fig.set_data(stress)
    if sign==1:
        dis1.set_ydata([100,Oy])
        dis1.set_xdata([100,Ox])
        dis2.set_ydata([0])
        dis2.set_xdata([0])
    if sign==-1:
        dis1.set_ydata([100])
        dis1.set_xdata([100])
        dis2.set_ydata([Oy])
        dis2.set_xdata([Ox])

nu=0.3
G=30e9
b=2e-10
D=G*b/(1*np.pi*(1-nu))
D_screw=G*b/(2*np.pi)
a=1.0e-6
X_axis=np.linspace(-a,a,num=200)
Y_axis=np.linspace(-a,a,num=200)

mbot = ((-7, 0), (0, 0), (0, -8), (0, 0), (7, 0))
mtop = ((-7, 0), (0, 0), (0, 8), (0, 0), (7, 0))

stress_xx=sigmaXX(X_axis,Y_axis,0.0,0.0,D,1)
single_dislocation(stress_xx)
plt.title(r'$\sigma_{xx}$')

stress_yy=sigmaYY(X_axis,Y_axis,0.0,0.0,D,1)
single_dislocation(stress_yy)
plt.title(r'$\sigma_{yy}$')

stress_xy=sigmaXY(X_axis,Y_axis,0.0,0.0,D,1)
single_dislocation(stress_xy)
plt.title(r'$\sigma_{xy}$')

stress_zz=sigmaZZ(X_axis,Y_axis,0.0,0.0,D,1)
single_dislocation(stress_zz)
plt.title(r'$\sigma_{zz}$')

pressure=press(X_axis,Y_axis,0.0,0.0,D,1)
single_dislocation(pressure)
plt.title(r'$\sigma_{H}$')

stress_zx_screw=screw_sigmaZX(X_axis,Y_axis,0.0,0.0,D_screw,1)
single_dislocation(stress_zx_screw,character='screw')
plt.title(r'Screw $\sigma_{zx}$')

stress_zy_screw=screw_sigmaZY(X_axis,Y_axis,0.0,0.0,D_screw,1)
single_dislocation(stress_zy_screw,character='screw')
plt.title(r'Screw $\sigma_{zy}$')

stress_tz_screw=screw_sigmaTZ(X_axis,Y_axis,0.0,0.0,D_screw,1)
single_dislocation(stress_tz_screw,character='screw')
plt.title(r'RH Screw $\sigma_{\theta z}$')

stress_tz_screw=screw_sigmaTZ(X_axis,Y_axis,0.0,0.0,D_screw,-1)
single_dislocation(stress_tz_screw,character='screw')
plt.title(r'LH Screw $\sigma_{\theta z}$')

stress_xx1=sigmaXX(X_axis,Y_axis,0.0,0.0,D,1)
o_x=0.0
o_y=0.5e-6
stress_xx2=sigmaXX(X_axis,Y_axis,o_x,o_y,D,1)
stress_xx=stress_xx1+stress_xx2
plt.figure(figsize=(7,5))
plt.pcolormesh(X_axis,Y_axis,stress_xx,vmin=-1e7,vmax=1e7,cmap='Spectral_r',shading='gouraud')
plt.plot(0,0,marker=mtop,markeredgecolor='white',markersize=10,markeredgewidth=2)
plt.plot(o_x,o_y,marker=mtop,markeredgecolor='white',markersize=10,markeredgewidth=2)
plt.axis('Off')
plt.axis('Equal')
plt.title(r'$\sigma_{xx}$')
plt.colorbar()

fig,dis1,dis2=two_dislocations()

interactive(add_dislocation, fig=fixed(fig), dis1=fixed(dis1),dis2=fixed(dis2),
            Ox=(0,200),Oy=(0,200),
            sign={"Opposite sign":-1,"Same sign":1})

fig,dis1,dis2=two_dislocations(component='xy')

interactive(add_dislocation, fig=fixed(fig), dis1=fixed(dis1),dis2=fixed(dis2),
            Ox=(0,200),Oy=(0,200),sign={"Opposite sign":-1,"Same sign":1},
            component=fixed('xy'))

